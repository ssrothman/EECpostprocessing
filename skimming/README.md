# Skimming Subpackage

The `skimming` subpackage provides a framework for skimming parquet datasets from the .root NANOAOD files.

- [Skimming Subpackage](#skimming-subpackage)
  - [Overview](#overview)
  - [Structure](#structure)
    - [Core Components](#core-components)
    - [Supporting Directories](#supporting-directories)
  - [Main API](#main-api)
    - [`skim()` Function](#skim-function)
  - [Key Modules](#key-modules)
    - [Objects (`objects/`)](#objects-objects)
    - [Selections (`selections/`)](#selections-selections)
    - [Weights (`weights/`)](#weights-weights)
    - [Tables (`tables/`)](#tables-tables)
  - [Configuration System](#configuration-system)
    - [Configuring corrections](#configuring-corrections)
    - [Event selection](#event-selection)
    - [Jet selection](#jet-selection)
    - [Object reconstruction](#object-reconstruction)
    - [Event weights](#event-weights)
    - [Suites](#suites)
  - [Usage Example](#usage-example)
  - [Scaling and Distributed Processing](#scaling-and-distributed-processing)
    - [1. Creating a "workspace" with all the needed configuration files, etc.](#1-creating-a-workspace-with-all-the-needed-configuration-files-etc)
    - [2. Staging out to the distributed system](#2-staging-out-to-the-distributed-system)
  - [Extension Points](#extension-points)


## Overview

This package orchestrates the full event processing pipeline:

1. **Loading raw events** from ROOT files using Awkward arrays
2. **Reconstructing physics objects** (jets, muons, MET, etc.) with corrections and systematics
3. **Applying event and jet selections** to filter events based on physics criteria
4. **Computing event weights** (scale factors, theoretical weights, etc.)
5. **Producing analysis tables** containing kinematic and analysis-specific variables

## Structure

### Core Components

- **skim.py**: Main entry point containing the `skim()` function that orchestrates the full processing pipeline
- **config/**: Configuration management system 
- **objects/**: Reconstructs and corrects physics objects (jets, muons, MET, EEC)
- **selections/**: Event and jet selection 
- **weights/**: Event weight calculation 
- **tables/**: Definitions of what to write into particular parquet datasets
- **scaleout/**: Utilities for distributed processing and job submission

### Supporting Directories

- **typing/**: Protocol definitions for type checking
- **data/**: Data files and reference information
- **scripts/**: Utility scripts for analysis tasks
- **test.py**: Integration test demonstrating basic workflow

## Main API

### `skim()` Function

The primary function for processing events:

```python
def skim(events, config, output_path, fs, tables):
    """
    Process physics events through the full pipeline.
    
    Parameters
    ----------
    events : ak.Array
        Awkward array containing ROOT event data
    config : dict
        Configuration dictionary 
    output_path : str
        Base path for output files
    fs : file-like object
        Filesystem interface (supports local or remote FS)
    tables : list
        List of table names to produce (e.g., ["EECres4Obs:True,tee"])
    """
```

**Workflow:**
1. Reconstructs corrected physics objects with systematics
2. Applies event selection filters
3. Applies jet selection filters
4. Computes event weights
5. Produces requested analysis tables
6. Writes output to `output_path` organized by table type

**Special case:** For the `count` table, only the total event count is computed and writted to json. This is used for normalizing dataset weights by the total cross-section, which requires knowing the total number of events in the MC sample. The `count` table can not be run at the same time as any other tables. 

## Key Modules

### Objects (`objects/`)

Reconstructs physics objects with corrections. This is the logic for interpreting fields in the NANO as python objects. Several objects need some computation for this association, and so dedicated classes have been provided for this purpose. These include:
- **Jets**: AK8 jets with JEC/JER and b-tagging, and AK4 jets lookup
- **Muons**: With basic selection requirements and rochester corrections applied
- **MET**: Missing transverse energy, with systematic variations
- **EEC**: EECs, with everything unflattened appropriately

Many objects can also be read directly from the NANO with no intermediate calculations. These are referred to as "generics", and the `GenericObjectContainer` class just provides a lookup to map these from NANO names to python names as needed.

All objects are encapsulated in a top-level object of the type `AllObjects`. This is what is actually passed around to the skimming logic.

### Selections (`selections/`)

Filters events based on physics criteria:
- **Event Selection**: Z→μμ selection, trigger requirements, etc.
- **Jet Selection**: Kinematic cuts, etc.

The selection modules are set up in a configurable fashion in `selections/factories.py`. I have currently only implemented one event selection routine and one jet selection routine, but it should be extremely straightforward to define your own in addition to mine. 

### Weights (`weights/`)

Computes event weights. Like the selections, this is set up in a configurable fashion in `weights/factory.py`. I have implemented two different event weight routines (`StandardWeights` and `GenonlyWeights`), and it should be extremely straightforward to implement more as needed.

### Tables (`tables/`)

Produces output tables:
- **EEC tables**: Energy correlator observables
- **Kinematic tables**: Jet and event kinematics
- **Cutflow tables**: Selection efficiency tracking
- **TableDriver**: Orchestrates table production

There is also a special table called `count`, which short-circuits the skimming logic to just count the total number of events in the .root files (not running any selections, or even actually reading anything besides the number of events from the .root file). 

## Configuration System

Configuration is managed through JSON files. This is handled in a modular fashion, and multiple configuration files are combined together into a given "suite" defining the entire configuration. This is loaded in the analysis code as

```python
from skimming.config.load_config import load_config

config = load_config('basic')  # Loads suite and merges all component configs
```

### Configuring corrections

Configuration for corrections is in `config/corrections/`. I have included default configurations for btagging and JECs, which I hope are clear enough to be self-documenting.

### Event selection

Configuration for the event selection is in `config/eventselection/`. The expected format is:
```json
{
    "eventsel" : {
        "class" : "Name of event selection class to use",
        "params" : {
            "parameter 1" : value,
            "parameter 2" : value,
            ...
        }
    }
}
```

### Jet selection

Configuration for the jet selection is in `config/jetselection`. The expected format is:
```json
{
    "jetsel" : {
        "class" : "Name of jet selection class to use",
        "params" : {
            "parameter 1" : value,
            "parameter 2" : value,
            ...
        }
    }
}
```

### Object reconstruction

Which objects to read from the NANO files and how to interpret them as python objects is configured by configuration in `config/objects'. The expected format is
```json
{
    "objects" : {
        "JECTARGET" : "Name of object to apply JECs to",
        "Object 1" : {
            "class" : "Class name",
            "params" : {
                "param 1" : value,
                "param 2" : value,
                ...
            },
            "MConly" : bool
        },
        "Object 2" : {
            "class" : "",
            "params" : {
                ...
            },
            "MConly" : bool
        },
        ...
    }
}
```
### Event weights 

The event weights are configured with json in `config/weights`. The expected format is
```json
{
    "eventweight" : {
        "class" : "Event weight class name",
        "params" : {
            ...
        }
    }
}
```

### Suites

The configuration in the different json files is combined modularly to get the full configuration to be used in skimming. The combined configurations are referred to as "suites", and each suite gets its own json file in `config/suites`. The expected format is
```json
{
    "configs" : [
        "path1.json",
        "path2.json",
        ...
    ],
    "configsuite_name" : "Name of this suite. Used to determine the filesystem path at which to write the skimmed table",
    "configsuite_comments" : "Human-readable comments"
}
```


## Usage Example

```python
from skimming.config.load_config import load_config
from skimming.skim import skim
import awkward as ak

# Load configuration
config = load_config('basic')

# Load events from ROOT (simplified example)
events = ak.from_numpy(...)  # Load from ROOT source

# Process events
skim(
    events=events,
    config=config,
    output_path='output/',
    fs=filesystem_interface,
    tables=['EECres4Obs:True,tee', 'cutflow']
)
```

## Scaling and Distributed Processing

The `scaleout/` module has routines for processing large datasets either on slurm or HTcondor. These are driven by scripts in the `scripts/` directory. 

Scaleout is separated into two steps:

### 1. Creating a "workspace" with all the needed configuration files, etc.

This is done with the `setup_skimming_workspace` script. E.g.:
```bash
 > setup_skimming_workspace.py test_skimming Apr_23_2025 Pythia_inclusive nominal --tables CutflowTable SimonJetKinematicsTable --output-location xrootd-submit --config-suite basic 
```

This will create a folder `test_skimming` with the following contents:
 - **config.json** containg the full configuration dictionary
 - **target_files.txt** containing a list of all the root files that need to be skimmed
 - **skimscript.py** a python script that takes one argument: the index of the file to be skimmed, and runs the skimming.

Things can be tested locally & at small scale by just running
```bash
> python skimscript.py 0
```

### 2. Staging out to the distributed system 

This is done with either the `stage_to_condor.py` or `stage_to_slurm.py` scripts. e.g. 
```bash
> stage_to_condor.py test_skimming test --files-per-job 5
```
This will set up the needed submission scripts for running on either slurm or condor, and if the `--exec` flag is passed, execute the submission scripts and submit jobs to the cluster.

## Extension Points

The package is designed for extensibility:
- **Custom Selections**: Write alternative selections classes, and add them to the factory lookup
- **Custom Weights**: Write alternative weights classes, and add them to the factory lookup
- **Custom Tables**: Add new table implementations in `tables/` and add them to the driver lookup
- **Custom Objects**: Extend object reconstruction in `objects/`