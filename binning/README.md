
# Binning Subpackage

The `binning` subpackage turns skimmed parquet tables into binned histograms (and optional covariance matrices) for EEC analysis. It supports gen-level, reco-level, and transfer binning, and can operate on large datasets via Arrow batch iteration.

## Overview

This package provides:

- Histogram construction from JSON binning configs
- Optional rebinning and narrowing of prebinned axes
- Event-weighted histogram filling from parquet datasets
- Covariance matrix computation via event-wise accumulation
- A CLI for scaleout processing

## Key Modules

- `main.py`: Core histogram building and filling utilities
- `config/`: JSON binning configurations (e.g., `res4tee.json`)
- `scripts/bin.py`: CLI wrapper used in scaleout workflows

## Core API

### Histogram construction

```python
from binning.main import build_hist, build_transfer_config

H, prebinned = build_hist(cfg)  # cfg is a list of axis dicts
transfer_cfg = build_transfer_config(gen_cfg, reco_cfg)
```

`build_hist()` returns a `hist.Hist` plus a `prebinned` map for axes tagged as `Variable-Prebinned` or `Regular-Prebinned`. Those axes can be rebinned and/or narrowed while preserving original bin edges for lookup.

### Filling histograms

```python
from binning.main import fill_hist, fill_cov
import pyarrow.dataset as ds

dataset = ds.dataset("/path/to/parquet", format="parquet")
H = fill_hist(H, prebinned, dataset, weightname="wt_nominal", itemwt="wt")

# Optional covariance
cov = fill_cov(H, prebinned, dataset, weightname="wt_nominal", itemwt="wt")
```

Supported features:

- `itemwt`: optional per-row weight column (e.g., `wt` or `wt_reco`)
- `reweight`: correctionlib `Correction` for reweighting
- `statN/statK`: statistical splitting for distributed processing

## CLI Usage

The `scripts/bin.py` entry point is used by scaleout jobs:

```bash
python binning/scripts/bin.py \
	<runtag> <dataset> <objsyst> <table> \
	--location local-submit \
	--config-suite BasicConfig \
	--bincfg res4tee \
	--wtsyst nominal \
	--statN -1 --statK -1
```

Notes:

- `table` must end with `Reco`, `Gen`, or `transfer`.
- `bincfg` defaults to the prefix of `table` (e.g., `res4tee` for `res4tee_Reco`).
- `--cov` produces a covariance matrix instead of a histogram.

Outputs:

- Histogram arrays are saved as `.npy` in a `*_BINNED` directory next to the input tables.
- A `*_bincfg.json` with the serialized binning is stored alongside output to ensure compatibility.

Scaleout workspace contents:

- `commands.txt`: one low-level `bin.py` command per line
- `binscript.py`: command-by-index runner used by schedulers

## Scaleout Setup

Use `setup_binning_workspace.py` as the single workspace-generation entry point. It handles the bulk-style dataset/systematic expansion directly.

Example with MC expanded over `--objsysts` and `--wtsysts`:

```bash
python binning/scripts/setup_binning_workspace.py \
	/path/to/workspace \
	Apr_19_2026 \
	--mc Pythia_inclusive Herwig_inclusive \
	--objsysts nominal JES_UP \
	--tables res4tee_totalReco \
	--wtsysts nominal wtvar_up
```

Example with both MC and data:

```bash
python binning/scripts/setup_binning_workspace.py \
	/path/to/workspace \
	Apr_19_2026 \
	--mc Pythia_inclusive WW WZ \
	--data DATA_2018A DATA_2018B \
	--objsysts nominal JES_UP \
	--tables res4tee_totalReco \
	--wtsysts nominal
```

Expansion behavior:

- `--mc`: expanded over all `--objsysts` and all `--wtsysts`
- `--data`: always uses `objsyst=DATA` and `wtsyst=nominal`

Internally, the workspace is built from explicit `(dataset, objsyst, wtsyst)` triples, so each command line is fully specified before it is written to `commands.txt`.

Run/stage with:

- `binning/scripts/run_binning_workspace_local.py`
- `binning/scripts/stage_binning_to_slurm.py`
- `binning/scripts/stage_binning_to_condor.py`

When running locally, each command writes combined stdout/stderr to `logs/command_<index>.log` inside the workspace.

## Binning Configuration

Configs live in `binning/config/*.json` and contain `gen` and `reco` axis lists. Each axis entry supports:

- `type`: `Regular` or `Variable`, optionally suffixed with `-Prebinned`
- `name`: axis name matching the parquet column
- `bins`, `start`, `stop` for `Regular`
- `edges` for `Variable`
- Optional `rebin` and `narrow_to` for `*-Prebinned`
- Optional `underflow`/`overflow` flags

Transfer binning concatenates reco axes with `_reco` suffix and gen axes with `_gen` suffix.

## Dependencies

- `hist`
- `numpy`
- `pyarrow`
- `tqdm`
- `directcov`
- `correctionlib`

## Testing

The file `binning/test.py` demonstrates a local end-to-end workflow using the config in `binning/config/res4tee.json` and direct parquet paths.
