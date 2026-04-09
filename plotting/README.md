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

## JSON-driven yield tables

The plotting package also includes a lightweight yield-table driver that reads one JSON config and writes a plain-text table.

Run it with:

```bash
python plotting/run_yields.py plotting/yield_config_example.json
```

For LaTeX-formatted output tables:

```bash
python plotting/run_yields.py --tex plotting/yield_config_example.json
```

When `--tex` is used, output files are written with a `.tex` extension.

The config schema is single-task (no `TO DO` wrapper) and expects:

- `meta`: output and formatting options
- `datasets`: datasets/stacks to evaluate
- `base_cuts`: cuts applied to all bins
- `bins`: list of bins, where each bin is either:
	- a list of cut expressions (with optional `bin_labels`), or
	- an object with `label` and `cuts`
- `weight_variable`: weight expression (for example `wt_nominal`)
- `alternative_weight` (optional): alternate weight expression available for selective dataset use
- `datasets[].use_alternative_weight` (optional): boolean flag to use `alternative_weight` for that dataset
	- defaults to `false` when omitted
	- requires top-level `alternative_weight` when any dataset enables it
- `table_metric` (optional):
	- `yield` (default): print raw yields in each bin
	- `efficiency`: print efficiency percentages in each bin
- `alternative_cut` (optional, efficiency mode only): shared cut list applied to the numerator in every bin
	- Formula per bin becomes `(alternative_cut + bins[bin]) / (base_cuts + bins[bin])`
- `totals` (optional): list of subtotal rows, each with `label` and dataset key list
- `percent_contribution` (optional): print dataset percentage rows relative to one totals row
	- `reference_total`: totals label to use as denominator
	- `format` (optional): printf-style number format (default `%.1f`)
- `row_order` (optional): customize row order and separator lines using a compact DSL
	- tokens are dataset `key` values and total labels
	- `|` inserts a single horizontal separator
	- `||` inserts a double horizontal separator
	- example: `"WW WZ ZZ TT ST | signal | Background Total | MC Total || DATA"`
- `bin_splits` (optional): emit multiple tables with duplicated dataset rows and different bin subsets
	- list of objects with:
		- `label` (optional): split/table label
		- `bins` (required): list of bin indices or bin-label strings to include in that split
		- `bins_group_label` (optional): LaTeX-only override for the multicolumn bins header text

Output format:

- Bins are rendered as columns
- Datasets are rendered as rows
- Total entries are rendered as additional rows
- Total rows are separated by dashed lines
- Optional percent contributions are rendered inline in each dataset cell as `yield (percent%)` using the configured total-row denominator
- In `efficiency` mode, totals are computed as grouped efficiencies with respect to the grouped base-cut denominator
- In `efficiency` mode, table cells (including totals rows) are rendered directly as percentages
- When `bin_splits` is provided, one table is produced per split with the same dataset column and only the selected bins

See [plotting/yield_config_example.json](plotting/yield_config_example.json) for a complete working config.

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
