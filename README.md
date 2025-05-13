# Summary

The postprocessing is split into two (or arguably three) steps:

1. skim the .root files into apache parquet datasets
2. fill histograms from the parquet datasets. Some of the plotting routines can run directly off of the parquet files, but others require this step
3. (?) make plots. This is actually handled by a different repo

# Skimming to parquet

the command looks like

```bash
python process.py <dataset name> <skimmer name> <config name> --samplelist <name>
```

The dataset name is a lookup into the samplelist from samples/. When you make a new samplelist make sure to import it into samples/_____init__.py 

The skimmer name is the name of the skimmer you want to use. For silly reasons these are in the folder binning/*

The config name is the name of the config you want to use. These live in config/* 

There are some more optional arguments, most of which I hope are self-explanatory?

I don't currently have a working one for projected EECs, but it should be easy to implement

# Filling histograms

This has so far only been implemented for the res4tee skimmer. This is called with fillEECRes4Hist.py, with the implementation in buildEECres4Hists.py. It should be reasoably easy to port this for other observables

# Plotting

This is handled in https://github.com/ssrothman/EECplotting
