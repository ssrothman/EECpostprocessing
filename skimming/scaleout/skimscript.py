import argparse
from ast import pattern
import os
import re

from general.fslookup.location_lookup import lookup_hostid

parser = argparse.ArgumentParser(description="Skim script for scaleout processing.")

parser.add_argument('i', type=int, help="Input file index")

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

#check if we are running on a submit host
#if we are, the hostname will be of the form submit%d.mit.edu
#use `re` to check this
import re
pattern = r"submit\d+\.mit\.edu"
hostname = os.uname().nodename
if re.match(pattern, hostname) and config['output_location'] == 'xrootd-submit':
    # if we are running on a submit host and the input location is xrootd-submit,
    # we can actually get the input from the local filesystem
    print("[Info] Changing output location to local-submit because we are running on a submit node")
    config['output_location'] = 'local-submit'

# setup output location and filesystem
targetfs, rootpath = location_lookup(config['output_location'])

if config['input_location'] != 'local':
    hostid = lookup_hostid(config['input_location'])
    input_file = 'root://' + hostid + '//' + input_file
    
events = NanoEventsFactory.from_root(input_file+":Events", schemaclass=NanoAODSchema).events()

if len(events) == 0:
    print(f"No events found in file {input_file}, skipping skim.")
    exit(0)

skim(
    events,  # pyright: ignore[reportArgumentType]
    config,
    os.path.join(rootpath, config['output_path']),
    targetfs,
    config['tables']
)