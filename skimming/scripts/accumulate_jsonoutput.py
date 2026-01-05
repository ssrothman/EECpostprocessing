#!/usr/bin/env python

import argparse

from fslookup.skim_path import lookup_skim_path
from simonpy.dictmerge import accumulate_dict

parser = argparse.ArgumentParser("accumulate json output from a skimmer")
parser.add_argument('runtag', type=str, help='Runtag')
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('table', type=str, help='Table name')
parser.add_argument('--objsyst', type=str, help='Object systematic variation', default=None)
parser.add_argument('--location', type=str, help='Storage location', default='local-submit')
parser.add_argument('--configsuite', type=str, help='Configuration suite name', default='BasicConfig')

args = parser.parse_args()

if args.objsyst is None:
    if 'data' in args.dataset.lower():
        args.objsyst = 'DATA'
    else:
        args.objsyst = 'nominal'

fs, path = lookup_skim_path(
    location=args.location,
    configsuite=args.configsuite,
    runtag=args.runtag,
    dataset=args.dataset,
    objsyst=args.objsyst,
    table=args.table
)

import os

allfiles = fs.listdir(path)

relevantfiles = []
for f in allfiles:
    if f['name'].endswith('json') and f['name'] != 'merged.json':
        relevantfiles.append(f['name'])

accu = None

import json
from tqdm import tqdm
for fname in tqdm(relevantfiles):
    with fs.open(os.path.join(path, fname), 'r') as f:
        part = json.load(f)

    if accu is None:
        accu = part

    else:
        accu = accumulate_dict(accu, part)

with fs.open(os.path.join(path, 'merged.json'), 'w') as f:
    json.dump(accu, f, indent=4)