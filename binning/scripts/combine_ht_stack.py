#!/usr/bin/env python
"""
Combine HT-binned Pythia histograms into a single pre-weighted stack.

Each HT bin is weighted by 1000 * xsec_i / count_i so the combined
output is normalized to 1 pb^-1. The unfolding then multiplies by
target_lumi (fb^-1) via the isStack mechanism in read_hist().

Usage:
    python -m binning.scripts.combine_ht_stack \
        --config-suite EvtMCprojConfig \
        --runtag v8 \
        --location dylan-lxplus-eos \
        --objsyst NOM \
        --wtsyst nominal \
        --table proj_totalReco \
        --output-dataset DYJetsToLL_Pythia_HTsum
        [--cov]
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config-suite', default='EvtMCprojConfig')
parser.add_argument('--runtag',       default='v8')
parser.add_argument('--location',     default='dylan-lxplus-eos')
parser.add_argument('--objsyst',      default='NOM')
parser.add_argument('--wtsyst',       default='nominal')
parser.add_argument('--table',        required=True)
parser.add_argument('--output-dataset', default='DYJetsToLL_Pythia_HTsum')
parser.add_argument('--output-objsyst', default=None,
                    help='objsyst for output path (defaults to --objsyst)')
parser.add_argument('--cov',          action='store_true')
args = parser.parse_args()
if args.output_objsyst is None:
    args.output_objsyst = args.objsyst

from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
from general.datasets.datasets import lookup_count
import shutil

HT_BINS = [
    ('DYJetsToLL_Pythia_HT70to100',    159.1),
    ('DYJetsToLL_Pythia_HT100to200',   159.4),
    ('DYJetsToLL_Pythia_HT200to400',   43.60),
    ('DYJetsToLL_Pythia_HT400to600',   5.918),
    ('DYJetsToLL_Pythia_HT600to800',   1.439),
    ('DYJetsToLL_Pythia_HT800to1200',  0.6462),
    ('DYJetsToLL_Pythia_HT1200to2500', 0.1514),
    ('DYJetsToLL_Pythia_HT2500toInf',  0.003395),
]

combined = None
first_fs = None

for dataset, xsec in HT_BINS:
    count = lookup_count(args.runtag, dataset, args.config_suite, args.objsyst)
    w = 1000.0 * xsec / count  # normalize to 1 pb^-1

    fs, inpath = get_hist_path(
        args.location, args.config_suite, args.runtag,
        dataset, args.objsyst, args.wtsyst,
        args.table, args.cov, -1, -1
    )
    with fs.open(inpath, 'rb') as f:
        arr = np.nan_to_num(np.load(f), nan=0.0)

    print(f"  {dataset}: xsec={xsec} pb, count={count:.0f}, w={w:.4e}, sum={arr.sum():.4e}")
    combined = w * arr if combined is None else combined + w * arr
    if first_fs is None:
        first_fs = fs

fs, outpath = get_hist_path(
    args.location, args.config_suite, args.runtag,
    args.output_dataset, args.output_objsyst, args.wtsyst,
    args.table, args.cov, -1, -1
)

import os
os.makedirs(os.path.dirname(outpath), exist_ok=True)

print(f"Writing combined histogram to {outpath}")
with fs.open(outpath, 'wb') as f:
    np.save(f, combined)

# Copy bincfg from first HT bin (all share the same binning)
if not args.cov:
    _, src_bincfg = get_hist_bincfg_path(
        args.location, args.config_suite, args.runtag,
        HT_BINS[0][0], args.objsyst, args.table
    )
    _, dst_bincfg = get_hist_bincfg_path(
        args.location, args.config_suite, args.runtag,
        args.output_dataset, args.output_objsyst, args.table
    )
    with first_fs.open(src_bincfg, 'r') as f:
        bincfg_data = f.read()
    with fs.open(dst_bincfg, 'w') as f:
        f.write(bincfg_data)
    print(f"Copied bincfg to {dst_bincfg}")
