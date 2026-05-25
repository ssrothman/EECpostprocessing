#!/usr/bin/env python

import argparse

from general.datasets.datasets import get_target_files
from skimming.util.parse_workspace import infer_workspace_metadata

parser = argparse.ArgumentParser(description='Rebuild target files for skimming')
parser.add_argument('--dir', help='Directory containing the skimming workspace', default='./', required=False)
args = parser.parse_args()

metadata, tables = infer_workspace_metadata(args.dir)
print(f"Rebuilding target files for workspace with metadata: {metadata}")
target_files, location = get_target_files(
    metadata['runtag'], metadata['dataset'], 
    exclude_dropped=tables != ['count']
)
print(f"Found {len(target_files)} target files at location {location}")
with open(f"{args.dir}/target_files.txt", "w") as f:
    for tf in target_files:
        f.write(f"{tf}\n")