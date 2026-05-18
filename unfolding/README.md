# Unfolding Package

The `unfolding` package provides tools for performing unfolding/deconvolution of detector-level measurements to generator-level distributions using iterative methods combined with a parametric detector model.

## Package Structure

```
unfolding/
├── histogram.py           # Histogram class for storing values with covariance matrices
├── detectormodel.py       # Detector model parameterization (transfer matrix, backgrounds)
├── loss.py                # Loss function for minimization
├── minimizer.py           # Optimizer wrapper using torchmin
├── forward.py             # Forward propagation and covariance propagation
├── io.py                  # Input/output utilities for histograms
├── specs.py               # Type definitions and protocols
├── unfolding_workspace.py # Setup utilities for unfolding workspace
├── scripts/               # Command-line tools
│   ├── setup_unfolding_workspace.py   # Create workspace from config
│   ├── run_unfolding.py               # Run the minimization
│   ├── build_unfolded_histogram.py    # Build final histogram from minimization
│   ├── run_forward.py                 # Test forward propagation
│   ├── plot_histogram_from_disk.py    # Visualize histogram
│   └── plot_model_from_disk.py        # Visualize detector model
└── test*.py               # Example scripts and tests
```

## Workflow Overview

A typical unfolding analysis follows these steps:

1. **Setup Workspace** (`setup_unfolding_workspace.py`)
   - Reads data (reco histogram) and MC (gen histogram + detector model)
   - Writes standardized directory structure for minimization

2. **Run Minimization** (`run_unfolding.py`)
   - Loads reco, gen baseline, and detector model
   - Constructs loss function (chi-squared + systematics)
   - Runs minimization to find optimal unfolded spectrum
   - Saves minimization artifacts (parameters, Hessian, checkpoints)

3. **Build Result Histogram** (`build_unfolded_histogram.py`)
   - Loads minimization result and Hessian matrix
   - Inverts Hessian to get covariance of unfolded spectrum
   - Propagates uncertainties through baseline template
   - Saves final `Histogram` object to disk

## Core Classes

### Histogram

Stores 1D binned data with full covariance matrix support.

**Constructor:**
```python
Histogram(
    values: np.ndarray,      # 1D array of bin values
    covmat: np.ndarray,      # 2D covariance matrix
    binning: ArbitraryBinning,  # Bin structure
    invcov: Optional[np.ndarray] = None,  # Pre-computed inverse (optional)
    L: Optional[np.ndarray] = None,       # Cholesky factor (optional)
    Linv: Optional[np.ndarray] = None,    # Inverse Cholesky (optional)
    eigvals: Optional[np.ndarray] = None, # Eigenvalues (optional)
    eigvecs: Optional[np.ndarray] = None  # Eigenvectors (optional)
)
```

**Key Methods:**
- `from_dataset(cfg, hist, whichhist)`: Load from skim data
- `from_disk(path)`: Load from saved files
- `dump_to_disk(path)`: Save to disk
- `compute_invcov()`: Compute and cache inverse covariance
- `compute_sqrt()`: Compute and cache Cholesky decomposition
- `plot(output_folder)`: Generate diagnostic plots

**Properties:**
- `values`: Bin values (numpy or torch tensor)
- `covmat`: Covariance matrix
- `invcov`: Inverse covariance matrix (lazy-computed)
- `L`: Cholesky decomposition (lazy-computed)
- `binning`: Bin structure

### DetectorModel

Encapsulates the detector response, consisting of:
- **Transfer matrix** T: Maps gen-level to reco-level signal (nReco × nGen)
- **Gen background** γ: Fraction of events that don't reach reco (nGen,)
- **Reco background** ρ: Fraction of reco events that are background (nReco,)

**Detector Response (Forward Model):**
```
reco = T @ [gen × (1 - γ)] × (1 + ρ)
```

Supports systematic variations for transfer matrix, backgrounds, and nuisance parameters.

**Constructor:**
```python
DetectorModel(
    transfer0: np.ndarray,       # Nominal transfer matrix
    gamma0: np.ndarray,          # Nominal gen background
    rho0: np.ndarray,            # Nominal reco background
    transferVariations: np.ndarray,      # Shape (nSyst, nReco, nGen)
    transferVarIndices: np.ndarray,      # Which systematics affect transfer
    gammaVariations: np.ndarray,         # Shape (nSyst, nGen)
    rhoVariations: np.ndarray,           # Shape (nSyst, nReco)
    binning: ArbitraryGenRecoBinning
)
```

**Key Methods:**
- `from_dataset(cfg)`: Construct from data/MC using systematic variations
- `from_disk(path)`: Load from saved files
- `dump_to_disk(path)`: Save to disk
- `forward(beta, theta)`: Compute reco-level prediction given gen spectrum β and nuisances θ

**Properties:**
- `nGen`: Number of generator-level bins
- `nReco`: Number of reco-level bins
- `nSyst`: Number of nuisance parameters
- `binning`: Gen/reco binning structure

## Loss Function

The loss function is chi-squared in reco space plus L2 regularization on nuisances:

```
L(β, θ) = (reco_pred - reco_data)ᵀ × Σ_reco⁻¹ × (reco_pred - reco_data) + Σ_i θ_i²
```

where:
- **β** is the gen-level unfolded spectrum (shape: nGen)
- **θ** is the nuisance parameter vector (shape: nSyst)
- **reco_pred = model.forward(β, θ)** is the predicted reco distribution
- **Σ_reco** is the reco covariance matrix
- Optional negative-bin penalty added to encourage physical (non-negative) solutions

**Constructor:**
```python
Loss(
    reco: Histogram,                    # Reco data to unfold
    genbaseline: np.ndarray | torch.Tensor,  # Template for gen scaling
    model: DetectorModel,               # Detector model
    negativePenalty: float = 0          # Penalty strength for negative bins
)
```

**Usage:**
```python
loss = Loss(reco, gen_template.values, model, negativePenalty=1e6)
chi2 = loss(gen_params, nuisance_params)  # Returns scalar loss
```

The generator distribution is always parametrized as a scaling of the baseline:
```
gen = genbaseline × β
```
where β is the optimization variable (shape: nGen). This helps stabilize fits and reduces degeneracies.

## Minimization Options

Minimization is performed using [torchmin](https://github.com/jettify/torchmin) optimizers via the `Minimizer` class.

### Configuration Dictionary

```python
mincfg = {
    "logpath": "logs",           # Directory for logs, checkpoints, result
    "method": "l-bfgs",          # Optimizer method (see below)
    "cpt_interval": 10,          # Save checkpoint every N iterations
    "cpt_start": 0,              # Start checkpoint numbering
    "method_options": {}         # Optimizer-specific options
}
```

### Available Methods

All methods from `torchmin.minimize()`:
- **l-bfgs** (default): Limited-memory BFGS, good for medium-size problems
- **bfgs**: Full BFGS, more memory-intensive
- **cg**: Conjugate gradient, fast for well-conditioned problems
- **gd**: Steepest descent, slow but very robust
- **newton**: Newton's method (requires Hessian)
- And others (see torchmin documentation)

### Method Options

Pass optimizer-specific parameters via `method_options`:

```python
mincfg["method_options"] = {
    "max_iter": 1000,        # Maximum iterations
    "line_search": "strong_wolfe",  # Line search strategy
    "max_eval": 5000,        # Max function evaluations
}
```

### Device Selection

- **CPU (default)**: `--device cpu` — reliable, no GPU required
- **GPU**: `--device cuda:0` — faster for large problems, requires valid CUDA setup

Note: If you encounter CUDA errors (e.g., "invalid device ordinal"), ensure:
1. You have allocated GPU resources (via Slurm or local system)
2. CUDA is properly installed and visible
3. Use `--device cpu` as fallback

### Checkpointing and Resuming

The minimizer saves:
- `minimizer_cfg.json`: Configuration used
- `x0.npy`: Initial parameters
- `checkpoints/cpt_NNN.npy`: Periodic snapshots
- `result/`: Final minimization result (x, hessian, fun, etc.)

Resume an interrupted minimization:
```bash
run_unfolding.py --continue-from --output-dir minimization
```

## Configuration File Format

The `config.json` file defines the data and detector model for an unfolding analysis. It is used by `setup_unfolding_workspace.py` to construct histograms and the detector model.

### Full Example

```json
{
    "data": {
        "dset": {
            "location": "scratch-submit",
            "config_suite": "BasicConfig",
            "runtag": "Mar_01_2026",
            "dataset": "Pythia_HTsum",
            "isStack": true,
            "isMC": true,
            "target_lumi": 1.0,
            "statN": 2,
            "statK": 0,
            "what": "res4tee"
        },
        "hist": {
            "wtsyst": "nominal",
            "objsyst": "nominal"
        }
    },
    "model": {
        "dset": {
            "location": "scratch-submit",
            "config_suite": "BasicConfig",
            "runtag": "Mar_01_2026",
            "dataset": "Pythia_HTsum",
            "isStack": true,
            "isMC": true,
            "target_lumi": 1.0,
            "statN": 2,
            "statK": 0,
            "what": "res4tee"
        },
        "systematics": [],
        "what": "res4tee"
    }
}
```

### Schema

#### `data` section

Specifies the reconstructed-level (reco) data histogram and optionally a generated-level (gen) reference.

**`data.dset`** — Dataset specification:
- `location` (string): Storage location identifier (e.g., `"scratch-submit"`, `"local-submit"`)
- `config_suite` (string): Configuration suite name (e.g., `"BasicConfig"`)
- `runtag` (string): Processing tag/version (e.g., `"Mar_01_2026"`)
- `dataset` (string): Dataset name (e.g., `"Pythia_HTsum"`, `"Data_2018A"`)
- `isStack` (boolean): If `true`, treat as a stack of datasets; `dataset` is a stack name
- `isMC` (boolean): `true` for Monte Carlo, `false` for data
- `target_lumi` (float): Target luminosity (for MC reweighting)
- `statN` (int): Number of statistically independent subsets to divide the dataset into (0 = use full dataset)
- `statK` (int): Which subset index to load (when `statN > 0`, ranges from 0 to `statN-1`)
- `what` (string): Physics variable/channel name (e.g., `"res4tee"`)

**`data.hist`** — Histogram systematics:
- `wtsyst` (string): Weight (theoretical) systematic variation (e.g., `"nominal"`, `"scale_up"`)
- `objsyst` (string): Object (reconstruction) systematic variation (e.g., `"nominal"`, `"jec_up"`)

#### `model` section

Specifies the Monte Carlo used to construct the detector model (transfer matrix, backgrounds).

**`model.dset`** — Dataset specification (same schema as `data.dset`):
- Usually the same MC sample as data, or a reference MC sample
- Used to compute transfer matrix and background fractions

**`model.systematics`** — Array of systematic variations:
```json
"systematics": [
    {
        "name": "scale",
        "isobjsyst": false,
        "onesided": false,
        "varytransfer": true
    },
    {
        "name": "pdf",
        "isobjsyst": false,
        "onesided": true,
        "varytransfer": false
    }
]
```

Each systematic object:
- `name` (string): Systematic name
- `isobjsyst` (boolean): `true` if object systematic (reconstruction), `false` if theoretical
- `onesided` (boolean): `true` if only one variation, `false` if symmetric up/down
- `varytransfer` (boolean): `true` if affects the transfer matrix, `false` if only backgrounds

For two-sided systematics, `"Up"` and `"Down"` suffixes are appended to the name to locate histogram variations.

**`model.what`** (string): Physics variable (must match `data.what`)

### Notes

- **Stacks**: If `isStack: true`, the dataset name should be a stack definition; data will be loaded from multiple constituent datasets and summed.
- **Statistically Independent Subsets**: Set `statN > 0` to divide the dataset into independent subsets. Use different `statK` values to load different subsets for ensemble studies.
- **Systematics**: Only list systematics that are available in the skimmed histograms. Missing systematics will cause errors.
- **Matching**: `data.what` and `model.what` must refer to the same physics variable and binning structure.

## Command-Line Scripts

### setup_unfolding_workspace.py

Create a workspace from a configuration file.

**Usage:**
```bash
# First, create a directory and add a config.json file
mkdir my_workspace
cd my_workspace
cp /path/to/config.json .

# Then run the setup script from inside the workspace directory
python ../unfolding/scripts/setup_unfolding_workspace.py
```

The script reads `config.json` from the current working directory and creates:
- `reco/`: Reco histogram
- `mcgen/`: Gen histogram (for baseline)
- `detectormodel/`: Detector model

### run_unfolding.py

Run the unfolding minimization.

**Usage:**
```bash
python unfolding/scripts/run_unfolding.py \
    --reco-path reco \
    --baseline-path mcgen \
    --model-path model \
    --output-dir minimization \
    --device cuda:0 \
    --method l-bfgs \
    --negative-penalty 1e6
```

**Options:**
- `--reco-path`: Path to reco histogram (default: `reco`)
- `--baseline-path`: Path to gen baseline histogram (default: `mcgen`)
- `--model-path`: Path to detector model (default: `model`)
- `--output-dir`: Where to save minimization artifacts (default: `minimization`)
- `--device`: Compute device (default: `cpu`)
- `--method`: Optimizer method (default: `l-bfgs`)
- `--cpt-interval`: Checkpoint interval (default: `10`)
- `--negative-penalty`: Penalty for negative bins (default: `1e6`)
- `--beta0`: Initial gen spectrum (`ones`, `random`, or path to `.npy` file)
- `--theta0`: Initial nuisances (`zeros`, `random`, or path to `.npy` file`)
- `--continue-from`: Resume from existing minimization

**Output:**
- `minimization/logs/`: Optimizer logs and checkpoints
- `minimization/result/`: Final parameters and Hessian

### build_unfolded_histogram.py

Build the final histogram from minimization results.

**Usage:**
```bash
python unfolding/scripts/build_unfolded_histogram.py \
    --minimization-dir minimization \
    --baseline-path mcgen \
    --output unfolded
```

**Options:**
- `--minimization-dir`: Path to minimization artifacts (default: `minimization`)
- `--baseline-path`: Path to gen baseline histogram (default: `mcgen`)
- `--output`: Where to save the unfolded histogram (default: `unfolded`)

**Output:**
- `unfolded/`: Final `Histogram` object (values, covariance, binning)

**Note:** This script requires `fasteigenpy` for Hessian inversion. It must run in a separate Python session from `run_unfolding.py` if both scipy and pytorch are used, due to environment conflicts.

### run_forward.py

Test forward propagation without minimization.

**Usage:**
```bash
python unfolding/scripts/run_forward.py \
    --gen-path gen \
    --model-path model \
    --nuisances nuisances.npy \
    --output forward \
    --nboot 1000 \
    --seed 12345
```

**Options:**
- `--gen-path`: Path to gen histogram (default: `gen`)
- `--model-path`: Path to detector model (default: `model`)
- `--output`: Where to save the forward histogram (default: `forward`)
- `--nuisances`: Path to nuisance parameters (default: zeros)
- `--nboot`: Bootstrap samples for covariance (default: `1000`)
- `--seed`: Random seed (default: `12345`)

### plot_histogram_from_disk.py

Visualize a saved histogram.

**Usage:**
```bash
python unfolding/scripts/plot_histogram_from_disk.py histogram_dir
```

Generates diagnostic plots in `histogram_dir/plots/`.

### plot_model_from_disk.py

Visualize a saved detector model.

**Usage:**
```bash
python unfolding/scripts/plot_model_from_disk.py model_dir
```

Generates diagnostic plots in `model_dir/plots/`.

## Example Workflow

```bash
# 1. Set up workspace
mkdir my_workspace
cd my_workspace
cp /path/to/config.json .
python ../../unfolding/scripts/setup_unfolding_workspace.py

# 2. Run minimization on GPU
python ../../unfolding/scripts/run_unfolding.py \
    --device cuda:0 \
    --negative-penalty 1e6 \
    --beta0 ones \
    --theta0 zeros

# 3. Build final histogram
python ../unfolding/scripts/build_unfolded_histogram.py

# 4. Inspect results
python ../unfolding/scripts/plot_histogram_from_disk.py unfolded/
```

## Systematic Uncertainties

Systematic variations in the detector model are parametrized via nuisance parameters θ. Each systematic can affect:
- The transfer matrix (shape distortions)
- Gen-level backgrounds (efficiency variations)
- Reco-level backgrounds (fake-rate variations)

Systematics are defined as either:
- **One-sided**: A single parameter variation
- **Two-sided**: Symmetric up/down variations (averaged)

The loss function includes an L2 penalty on nuisance parameters:
```
penalty = Σ_i θ_i²
```

This acts as a Gaussian prior centered at 0, pulling nuisance parameters toward their nominal values.

## Covariance and Uncertainties

### Reco-level covariance
Stored in `Histogram.covmat`. Usually dominated by statistical uncertainties from data/MC.

### Gen-level (unfolded) covariance
Computed from the inverse Hessian of the loss function at the minimum. The Hessian encodes how sensitive the solution is to variations in the input.

High covariance indicates strong correlations between bins, common when neighboring bins are folded together by the detector response.

## Advanced Topics

### Custom Initial Guesses

Provide custom initial parameters:
```bash
python run_unfolding.py --beta0 my_beta.npy --theta0 my_theta.npy
```

Useful for:
- Warm-starting from a previous fit
- Testing sensitivity to initial conditions

### Continuing Interrupted Minimizations

If minimization is interrupted, resume where it left off:
```bash
python run_unfolding.py --continue-from --output-dir minimization
```

The minimizer automatically loads the last checkpoint and continues.

### Negative Penalty Tuning

The `negativePenalty` parameter discourages unphysical (negative) solutions:
- **No penalty** (`0`): Allows negative bins, fast but physically questionable
- **Moderate penalty** (`1e4 - 1e6`): Good for typical analyses
- **High penalty** (`1e7+`): Strongly enforces non-negative, may slow convergence

Tune based on your data quality and physics expectations.

## Troubleshooting

**CUDA errors (invalid device ordinal)**
- Ensure GPU allocation: Use `srun --gres=gpu:1` in Slurm environments
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Fall back to CPU: `--device cpu`

**High chi-squared in result**
- Check reco covariance matrix conditioning
- Verify detector model is well-measured
- Reduce negative penalty if solution is biased

**Unfolded spectrum is noisy**
- Increase negative penalty
- Check number of iterations reached
- Verify baseline template is appropriate

**Minimization doesn't converge**
- Try a gentler method: `--method cg` or `--method gd`
- Increase max iterations via `method_options`
- Simplify the model (fewer nuisance parameters)

## References

- [torchmin documentation](https://github.com/jettify/torchmin)
- Unfolding methods in high-energy physics (typical references: D'Agostini 1994, Schmitt et al.)
- Detector simulation and response matrices
