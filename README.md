# EEC postprocessing <!-- omit from toc -->

This repo handles the python postprocessing for EEC analysis. This is split into four different pieces:

1. skimming the .root files into parquet datasets
2. filling histograms from the parquet datasets
3. making plots. This can be done either directly off the parquet datasets (eg for kinematics) or off the pre-binned histograms from step 2. 
4. unfolding. This requires pre-binned histograms


## Table of contents <!-- omit from toc -->

- [Dependencies](#dependencies)
- [Package structure](#package-structure)
- [General utilities](#general-utilities)
  - [Datasets lists](#datasets-lists)
  - [Filesystem lookups](#filesystem-lookups)
    - [Storage locations](#storage-locations)
    - [Paths](#paths)
- [Skimming](#skimming)
- [Binning](#binning)
- [Plotting](#plotting)
- [Unfolding](#unfolding)

## Dependencies

This package relies on two small c++ wrappers that I wrote:

 - `fasteigenpy` ([here](https://github.com/ssrothman/fasteigenpy)) - optimized c++ eigen bindings. Used for matrix inverses, eigendecompositions, etc
 - `directcov` ([here](https://github.com/ssrothman/directcov)) - computation of covariance matrices from correlated, binned data

This package also relies on a number of external python packages:

 - pyarrow 
 - awkward
 - numpy
 - hist
 - correctionlib
 - fsspec-xrootd (not available via conda)
 - matplotlib >= 3.10
 - mplhep
 - torch (for unfolding)
 - pytorch-minimize (for unfolding)

Additionally, two of my utility packages are distributed as submodules:

 - `simonplot` - plotting utilities
 - `simonpy` - general-purpose utilities

After cloning, you need to run `git submodule update --init --recursive` to properly set these up. 

## Package structure

The basic structure of the package looks like:
```
general/ - common utilities needed for all three steps (skimming, binning, and plotting)
skimming/ - routines for building parquet datasets from rootfiles 
binning/ - routines for building histograms off parquet datasets
plotting/ - routines for making plots
unfolding/ - routines for running the unfolding
```

In the top-level README here, I provide only documentation for the `general/` utilities. The subpackages for skimming, binning, plotting, and unfolding have their own READMEs.

## General utilities

The `general` subpackage provides common functionality needed by all four other submodules. This primarily relates to filesystem lookups and dataset lists, so that everything can be looked up in the filesystem autoamtically, and naming conventions can be standardized in one place. 

### Datasets lists

`general/datasets` contains dataset lists, centrally defining things like names, colors, cross-sections, etc. The logic in `general/datasets/datasets.py` reads ALL .json files in this directory into a big datasets configuration dictionary. Adding new datasets to the list is as easy as adding new .json files here. The expected json structure is:

```json
{
    "runtag1" : {
        "base" : "common basepath for all datasets with this runtag",
        "dataset1" : {
            "tag" : "str or List[str] with path(s) to folder(s) containing the .root files, relative to the basepath defined earlier",
            "location" : "location name where the rootfiles live (see following section)",
            "era" : "JEC era name",
            "flags" : {
                "flag1" : value,
                "flag2" : value,
                ...
            },
            "label" : "label to print on plots",
            "lumi" : "luminosity value [should only be present for DATA datasets]",
            "xsec" : "cross-section value [should only be present for MC datasets]",
            "color" : "color to use on plots"
        }
    },
    "runtag2" : {
        ...
    },
    ...
}
```
The `location` field should be a lookup into the locations lookup, as described below. The `flags` field is used to define any other needed information for skimming the datasets. This is currently only used to skim the `HT < 70 GeV` subset of the inclusive Pythia sample, with the flag `{"genHT" : 70}`.

### Filesystem lookups

`general/fslookup` contains utility logic for looking up filesystem paths. 

#### Storage locations

In order to provide a unified interface independent of where files are actually stored, the storage location is defined with a `location` string, which is used as a lookup in `general/fslookup/location_lookup.json`. The expected format is
```json
{
    "location name" : [
        "root redirector, or \"local\"",
        "base path"
    ]
}
```
For example, I have defined 
```json
{
    "simon-LPC" : [
        "cmseos.fnal.gov",
        "/store/group/lpcpfnano/srothman/"
    ],
    "local-submit" : [
        "local",
        "/ceph/submit/data/group/cms/store/user/srothman/EEC_v2"
    ],
    "xrootd-submit" : [
        "submit55.mit.edu",
        "/store/user/srothman/EEC_v2"
    ]
}
```
for my folder on LPC and access to my local files on the MIT cluster, either locally or through xrootd. If you have files stored on a different system, it should be as easy as adding the needed lines to the `location-lookups.json` for my code to find them. 

#### Paths

In addition to defining common logic for filesystem lookups, I also define common logic for paths in the filesystem. Skimmed datasets can be found with 
```python
from general.fslookup.skim_path import lookup_skim_path

fs_object, path_string = lookup_skim_path(
    location : str = "location string from location-lookups.json",
    configsuite : str = "name of the config used for the skimming (defined in the skimming/ submodule)",
    runtag : str = "runtag (from the datasets json)",
    dataset : str = "name of the dataset (from the datasets json)",
    objsyst : str = "name of the systematic variation on the objects (ie not just an event weight)",
    table : str = " name of the skimmed table"
)
```

## Skimming

Documentation on the skimming submodule can be found in its own dedicated README [here](https://github.com/ssrothman/EECpostprocessing/tree/master/skimming/README.md)

## Binning

Documentation on the skimming submodule can be found in its own dedicated README [here](https://github.com/ssrothman/EECpostprocessing/tree/master/binning/README.md)

## Plotting

Documentation on the skimming submodule can be found in its own dedicated README [here](https://github.com/ssrothman/EECpostprocessing/tree/master/plotting/README.md)

## Unfolding

Documentation on the skimming submodule can be found in its own dedicated README [here](https://github.com/ssrothman/EECpostprocessing/tree/master/unfolding/README.md)