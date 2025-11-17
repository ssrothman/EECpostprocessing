# Summary

The postprocessing is split into two (or arguably three) steps:

1. skim the .root files into apache parquet datasets
2. fill histograms from the parquet datasets. Some of the plotting routines can run directly off of the parquet files, but others require this step
3. (?) make plots. This is actually handled by a different repo

# Setting up with a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ever subsequent time that you run the code, you only need to run the activation step.


# Skimming to parquet

the command looks like

```bash
python process.py <dataset name> <skimmer name> <config name> --samplelist <name>
```

The dataset name is a lookup into the samplelist from samples/. When you make a new samplelist make sure to import it into samples/\_\_init\_\_.py. The identifier passed to the --samplelist option is the name assigned to that samplelist in the \_\_init\_\_.py file. 

The skimmer name is the name of the skimmer you want to use. For annoying historical reasons these are in the folder skimming/. If you make a new one be sure to import it and add it to the dictionary in processing/EECProcessor.py. This is where the skimmer name you pass will be looked up. Currently available and not-obselete skimmers are:
 - Kinematics: dumps various kinematic variables. Creates datasets for event-level qauntities (eg rho, Zmass, ...), jet-level quantities (eg pt, eta, ...), and particle-level quantities (eg pt, pdgid, ...).
 - EECres4\*: dumps the res4\* entries. 
 - EECproj: dumps the projected EEC entries.

The config name is the name of the config you want to use. Config information lives in config/. The config name that you pass is the name of a "config suite", which identifies a collection of .json files that actually contain the configuration information. These live in configs/suites/, and hopefully the expected format is obvious just by looking at one. 

There are various other options that can be passed to the process.py script. Thse include:
 - --force: run even if the target destintion already contains the results of a previous run
 - --recover: look at the list of errored files from a previous invocation of process.py and try to rerun only on them. --recover implies --force
 - --filesplit N: split each file into N chunks. This can be useful if worker nodes do not have enough memory to process an entire file in one go. If you run --recovery, you must pass the same filesplit value as was used in the previous run. This is not checked, but there is a chance of double-counting events otherwise. 
 - --filebatch N: batch N files into each skimming task. This can reduce the overhead on the dask scheduler, but it typically not needed
 - --extra-tags str: a string to append to the destination filename, can be used to distinguish subsequent runs tht would otherwise by default overwrite each other
 - --syst str: the name of one of the "object systematics" that actually change object properties (as opposed to event weights). Currently only the JETMET systematics are supported. 
 - --no\*: various options to disable event weights, corrections, or selections
 - --Zreweight: not up-to-date, used to reweight the Z kinematics
 - --local: used for testing. In --local mode, the <dataset_name> is instead interpreted as a file path. 
 - --nfiles N: used to run on only a subset of the files in the dataset.
 - --startfile N: used to identify which subset of the dataset to run on with --nfiles N
 - --use-slurm: run the skimming on slurm
 - --use-local: run the skimming in parallel on the local machine
 - --use-local-debug: run the skimming in one thread on the local machine
 - --JECera str: overwrite the default JECera (specified in the sample file) with the user-specified one. Useful for testing
 - --verbose N: 0 = no verbosity, 1 = verbose

# Filling histograms

see the README in binning/

# Plotting

This is handled in https://github.com/ssrothman/EECplotting
