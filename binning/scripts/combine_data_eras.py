#!/usr/bin/env python
"""
Combine data era histograms into a single stack (simple sum, no xsec weighting).

Usage:
    python -m binning.scripts.combine_data_eras \
        --config-suite EvtDataprojConfig \
        --runtag data_v1 \
        --location dylan-lxplus-eos \
        --objsyst DATA \
        --wtsyst nominal \
        --table proj_totalReco \
        --output-dataset DATA_2018ABD
        [--cov]
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config-suite', default='EvtDataprojConfig')
parser.add_argument('--runtag',       default='data_v1')
parser.add_argument('--location',     default='dylan-lxplus-eos')
parser.add_argument('--objsyst',      default='DATA')
parser.add_argument('--wtsyst',       default='nominal')
parser.add_argument('--table',        required=True)
parser.add_argument('--output-dataset', default='DATA_2018ABD')
parser.add_argument('--cov',          action='store_true')
args = parser.parse_args()

from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
import os

ERAS = ['DATA_2018A', 'DATA_2018B', 'DATA_2018D']

combined = None
first_fs = None

for dataset in ERAS:
    fs, inpath = get_hist_path(
        args.location, args.config_suite, args.runtag,
        dataset, args.objsyst, args.wtsyst,
        args.table, args.cov, -1, -1
    )
    with fs.open(inpath, 'rb') as f:
        arr = np.nan_to_num(np.load(f), nan=0.0)

    print(f"  {dataset}: sum={arr.sum():.4e}")
    combined = arr.copy() if combined is None else combined + arr
    if first_fs is None:
        first_fs = fs

fs, outpath = get_hist_path(
    args.location, args.config_suite, args.runtag,
    args.output_dataset, args.objsyst, args.wtsyst,
    args.table, args.cov, -1, -1
)

os.makedirs(os.path.dirname(outpath), exist_ok=True)

print(f"Writing combined histogram to {outpath}")
with fs.open(outpath, 'wb') as f:
    np.save(f, combined)

if not args.cov:
    _, src_bincfg = get_hist_bincfg_path(
        args.location, args.config_suite, args.runtag,
        ERAS[0], args.objsyst, args.table
    )
    _, dst_bincfg = get_hist_bincfg_path(
        args.location, args.config_suite, args.runtag,
        args.output_dataset, args.objsyst, args.table
    )
    with first_fs.open(src_bincfg, 'r') as f:
        bincfg_data = f.read()
    with fs.open(dst_bincfg, 'w') as f:
        f.write(bincfg_data)
    print(f"Copied bincfg to {dst_bincfg}")
