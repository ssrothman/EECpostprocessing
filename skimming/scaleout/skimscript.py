import argparse
from ast import pattern
import os
import re

from general.fslookup.location_lookup import lookup_hostid

parser = argparse.ArgumentParser(description="Skim script for scaleout processing.")

parser.add_argument('i', type=int, help="Input file index")
parser.add_argument('--override-location', type=str, default=None, help='Override the output location')
parser.add_argument('--split-by-rows', type=int, default=-1, help='Split the input file by the specified number of rows')
args = parser.parse_args()

import json
from skimming.skim import skim
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from general.datasets.datasets import location_lookup
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

if args.override_location is not None:
    print(f"[Info] Overriding output location to {args.override_location}")
    config['output_location'] = args.override_location

# setup output location and filesystem
targetfs, rootpath = location_lookup(config['output_location'])

if config['input_location'] != 'local':
    hostid = lookup_hostid(config['input_location'])
    input_file = 'root://' + hostid + '//' + input_file
    
events = NanoEventsFactory.from_root(input_file+":Events", schemaclass=NanoAODSchema).events()
Nevt = len(events)

if Nevt == 0:
    print(f"No events found in file {input_file}, skipping skim.")
    exit(0)

if args.split_by_rows < 0:
    args.split_by_rows = Nevt*2

for start in range(0, Nevt, args.split_by_rows):
    stop = min(start + args.split_by_rows, Nevt)

    print("doing block from {} to {}".format(start, stop))
    subevts = NanoEventsFactory.from_root(
        input_file+":Events", 
        schemaclass=NanoAODSchema,
        entry_start=start,
        entry_stop=stop
    ).events()

    skim(
        subevts,  # pyright: ignore[reportArgumentType]
        config,
        os.path.join(rootpath, config['output_path']),
        targetfs,
        config['tables']
    )