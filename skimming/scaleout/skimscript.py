import argparse
import os

from skimming.fsutil.location_lookup import lookup_hostid

parser = argparse.ArgumentParser(description="Skim script for scaleout processing.")

parser.add_argument('i', type=int, help="Input file index")

args = parser.parse_args()

import json
from skimming.skim import skim
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from skimming.datasets.datasets import location_lookup
from fsspec_xrootd import XRootDFileSystem
NanoAODSchema.warn_missing_crossrefs = False

# Find the input file based on the provided index
input_file = None
with open('./target_files.txt') as f:
    for i, line in enumerate(f):
        if i == args.i:
            input_file = line.strip()
            break
if input_file is None:
    raise IndexError(f"Input file index {args.i} out of range")

# load config
with open("./config.json") as f:
    config = json.load(f)

# setup output location and filesystem
targetfs, rootpath = location_lookup(config['output_location'])

if config['input_location'] != 'local':
    hostid = lookup_hostid(config['input_location'])
    input_file = 'root://' + hostid + '//' + input_file
    
events = NanoEventsFactory.from_root(input_file+":Events", schemaclass=NanoAODSchema).events()
skim(
    events,  # pyright: ignore[reportArgumentType]
    config,
    os.path.join(rootpath, config['output_path']),
    targetfs,
    config['tables']
)