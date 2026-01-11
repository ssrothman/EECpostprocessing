# Summary

This is the code for filling EEC histograms off the parquet files produced by the EEC skimmers. At the moment it is only implemented for the 4-th order resoolved EECs, but it should be straightforward to port the functionality to the other observables. 

# Core logic

The core logic is in the file buildEECres4Hists.py. This is implemented in three functions:

  #### 1. fill_hist_from_parquet(...). This just fills a regular EEC histogram from the parquet, and takes arguments:
 
  ##### Generic:

   - basepath: the path to the parquet dataset to run over
   - systwt: the name of the event weight column in the parquet to use
   - prebinned: boolean flag to control whether or not to use prebinned EEC functionality. 
   - kinreweight: a function with signature (Z pt, Z rapidity) -> scale factor, or None. Used for event reweighting wrt Z kinematics
   - fs: filesystem object on which the parquet dataset can be found

 ##### Dataset splitting:
   - statN: number of statistically independent subsets we will create
   - statK: which of the statistically independent subsets to create right now [the function should be called N times, passing statK=0, 1, 2, .. N-1]

 ##### Poisson bootstrap (depricated, should not be used):
   - Nboot: number of bootstrap replicas to create. This should probably just be set to zero, as I have discovered that the bootstrapping procedure has poor convergence, and have implemented a different approach for uncertainties (see below)
   - rng_offset: offset into the rng stream for poisson bootstrapping
   - r123type: rng generator type for poisson bootstrapping
   - skipNominal: only record poisson bootstraps, and not the nominal variation

 ##### Debug:
   - nbatch: pass a positive value to run over the first nbatch fragments of the dataset. should only be used for testing
   - collect_debug_info: also record and save a bunch of debug info alongside the histogram

#### 2. fill_direct_covariance_type1(...). This builds a covariance matrix (the type1 label is legacy and doesn't mean anything), and takes arguments:

  ##### Generic:
 - basepath: the path to the parquet dataset to run over
 - systwt: the name of the event weight column in the parquet to use
 - kinreweight: a function with signature (Z pt, Z rapidity) -> scale factor, or None. Used for event reweighting wrt Z kinematics
 - fs: filesystem object on which the parquet dataset can be found

 ##### Dataset splitting:
   - statN: number of statistically independent subsets we will create
   - statK: which of the statistically independent subsets to create right now [the function should be called N times, passing statK=0, 1, 2, .. N-1]

 ##### Poisson bootstrap (depricated, should not be used):
   - Nboot: doesn't do anything

#### 3. fill_transferhist_from_parquet(...) This fills the transfer matrix histogram, and takes arguments:

  ##### Generic:

   - basepath: the path to the parquet dataset to run over
   - systwt: the name of the event weight column in the parquet to use
   - kinreweight: a function with signature (Z pt, Z rapidity) -> scale factor, or None. Used for event reweighting wrt Z kinematics
   - fs: filesystem object on which the parquet dataset can be found

 ##### Dataset splitting:
   - statN: number of statistically independent subsets we will create
   - statK: which of the statistically independent subsets to create right now [the function should be called N times, passing statK=0, 1, 2, .. N-1]

 ##### Poisson bootstrap (depricated, should not be used):
   - Nboot: number of bootstrap replicas to create. This should probably just be set to zero, as I have discovered that the bootstrapping procedure has poor convergence, and have implemented a different approach for uncertainties (see below)
   - rng_offset: offset into the rng stream for poisson bootstrapping
   - r123type: rng generator type for poisson bootstrapping
   - skipNominal: only record poisson bootstraps, and not the nominal variation

 ##### Debug:
   - nbatch: pass a positive value to run over the first nbatch fragments of the dataset. should only be used for testing

# Scripts

Directly setting up all of the arguments for these functions is kinda annoying, so I've written some scripts to partially automate things. The main one is

### fill_res4_hist.py

Prepares a call to the backend functionality, respecting all of the naming conventions, etc, that I have defined. 

Arguments:

  ##### Generic:

 - Runtag, Sample, Binner, Hist, Objsyst: used to lookup the correct filepath for the parquet dataset
 - Wtsyst: name of the weight systematic to use
 - usexrootd: whether to access the parquet dataset over xrootd or over the local filesystem
 - prebinned:

  ##### Poisson bootstrap (depricated, should not be used):

 - nboot, rng, skipNominal, r123type

 ##### Dataset splitting:
   - statN: number of statistically independent subsets we will create
   - statK: which of the statistically independent subsets to create right now [the function should be called N times, passing statK=0, 1, 2, .. N-1]

 ##### Z kinematics reweighting:
  - kinreweight_path: path to a correctionlib json containing the Z kinematic SF function
  - kinreweight_key: key within the correctionlib json for the correct SF function

  NB if these are not passed,their default values of None are valid and will just result in no Z kinematic reweighting

 ##### job execution:

  - --slurm: run on slurm
  - --condor: run on HTcondor

  ##### Debug:
   - nbatch: pass a positive value to run over the first nbatch fragments of the dataset. should only be used for testing
   - collect_debug_info: also record and save a bunch of debug info alongside the histogram

### fill_all_res4_hists.py

This wraps multiple calls to fill_res4_hist.py. The argument signature is almost identical, except that most of the parameters accept a list rather than a single value. These are interpreted into a list of commands to run, which are then executed with subprocess through a multiprocessing Pool. 

A few comments about scaling out:

 - It can be convenient to dump a bunch of jobs onto a batch system through slurm or condor
 - Each job reads the whole dataset, so you can't go too crazy without risking some I/O bottlenecks
 - If you want to break the dataset into chunks that can be processed independently for better scaling performance, the way to do this is through the statN/statK parameters. For example, to split the dataset into 10 independent jobs, you can do something like `--statK 10 --statN 0 1 2 3 4 5 6 7 8 9` to run 10 jobs each processing 1/10th of the dataset

# Things that will need to be updated

1. Make copies of all of the res4 functionality for projected EECs 
2. Change the functionality to replace the (R, r, c) indexing with just (R) indexing
3. Setup the binning correctly 
4. Add and test the prebinned functionality. It's only been implemented for the base function, and I'm not sure that what is implemented even works correctly. But it should be pretty straightforward anyway... 
5. Change the filesystem lookup to agree with whatever you are using