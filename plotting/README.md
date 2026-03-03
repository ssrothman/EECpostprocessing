# Plotting Submodule

The `plotting` submodule provides lightweight drivers and dataset builders for producing analysis plots from skimmed parquet tables using `simonplot`. It is designed for quick, reproducible plotting via JSON configs or simple Python scripts.

## Overview

This module:

- Builds `simonplot` datasets directly from skimmed parquet outputs
- Supports dataset stacks (data/MC groups) with shared styling
- Runs plot jobs from JSON configs with a simple CLI

## Structure

- [plotting/load_datasets.py](plotting/load_datasets.py): Dataset builders for parquet inputs and dataset stacks
- [plotting/plotdriver.py](plotting/plotdriver.py): Plot execution driver (variables, cuts, binning, output)
- [plotting/run_plots.py](plotting/run_plots.py): CLI entry point for JSON plot configs
- [plotting/AN_plot_configs](plotting/AN_plot_configs): Example analysis plot configs

## Main API

### Build datasets

Use `build_pq_dataset()` for a single dataset or `build_pq_dataset_stack()` for stacked collections:

- `build_pq_dataset(configsuite, runtag, dataset, objsyst, table, location='local-submit')`
- `build_pq_dataset_stack(configsuite, runtag, stackname, objsyst, table, location='local-submit')`

These functions:

- Resolve parquet paths from skim outputs
- Load dataset metadata (labels/colors/xsec or lumi)
- Attach event counts from the `count` table

## JSON-driven plotting

The CLI reads a config file with a `TO DO` list of plot jobs:

```bash
python plotting/run_plots.py plotting/AN_plot_configs/htsum_check.json
```

Each job supports:

- `datasets`: list of dataset or stack definitions
- `variables`: list of variables, optionally with `var::`/`cut::` prefixes
- `weights`: list of weight expressions
- `cut`: list of cut expressions (empty for no cut)
- `binning`: `auto` or `autoint:<labelkey>`
- `plotspath` / `plotsprefix`: output controls
- `driver`: currently `plot_histogram`

See [plotting/AN_plot_configs/htsum_check.json](plotting/AN_plot_configs/htsum_check.json) for a working example.

## Binning

`plotdriver.py` supports:

- `auto`: `simonplot.binning.AutoBinning()`
- `autoint:<labelkey>`: `simonplot.binning.AutoIntCategoryBinning()` with labels from [plotting/autoint_lookups.json](plotting/autoint_lookups.json)

## Usage Example (Python)

`test.py` demonstrates interactive plotting from Python. See [plotting/test.py](plotting/test.py).

## Dependencies

- `simonplot`
- `pyarrow` (via dataset readers)
- `json`

## Notes

- Dataset metadata comes from `general.datasets` and must define `color`, `label`, and either `lumi` or `xsec`.
