#!/usr/bin/env python

import argparse

from simonpy.dictmerge import accumulate_dict

parser = argparse.ArgumentParser("accumulate json output from a skimmer")
parser.add_argument('path')

args = parser.parse_args()

import os

allfiles = os.listdir(args.path)

relevantfiles = []
for f in allfiles:
    if f.endswith('json') and f != 'merged.json':
        relevantfiles.append(f)

accu = None

import json
from tqdm import tqdm
for fname in tqdm(relevantfiles):
    with open(os.path.join(args.path, fname), 'r') as f:
        part = json.load(f)

    if accu is None:
        accu = part

    else:
        accu = accumulate_dict(accu, part)

with open(os.path.join(args.path, 'merged.json'), 'w') as f:
    json.dump(accu, f, indent=4)