#!/usr/bin/env -S python

import argparse

from arrow import get

parser = argparse.ArgumentParser(description='combine histograms from different data subsets')

parser.add_argument("runtag", type=str, help="Runtag to process")
parser.add_argument("dataset", type=str, help="Dataset to process")
parser.add_argument("objsyst", type=str, help="Object systematic variation to process")
parser.add_argument("wtsyst", type=str, help="Weight systematic variation to process")
parser.add_argument('table', type=str, help='Table name')
parser.add_argument('oldStatN', type=int, help='N for statsplit to combine')

parser.add_argument("--location", type=str, 
                    help="Location to write output to",
                    default='local-submit')
parser.add_argument("--config-suite", type=str,
                    help="Configuration suite to use for skimming",
                    default='BasicConfig')

parser.add_argument('--statN', type=int, help='N for statsplit (-1 for no statsplit)',
                    default=-1)
parser.add_argument('--statK', type=int, help='K for statsplit processing (ignored if N==-1)',
                    default=-1)

parser.add_argument('--cov', action='store_true', help='Covariance computation')

args = parser.parse_args()

# first, basic check on the inputs
if args.oldStatN <= 0:
    raise ValueError("oldStatN must be > 0")
if args.statN > 0 and args.oldStatN % args.statN != 0:
    raise ValueError("oldStatN must be divisible by statN")
if args.statN > 0 and args.oldStatN <= args.statN:
    raise ValueError("oldStatN must be greater than statN")

from general.fslookup.hist_lookup import get_hist_path
import numpy as np

accu = None
print(args.oldStatN)

for oldStatK in range(args.oldStatN):
    if oldStatK % args.statN != args.statK and args.statN > 0:
        continue

    print(f"Processing statsplit {oldStatK}...")
    # here we would load the histogram for this statsplit and add it to a cumulative histogram

    fs, inpath = get_hist_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        args.wtsyst,
        args.table,
        cov = args.cov,
        statN = args.oldStatN,
        statK = oldStatK
    )

    with fs.open(inpath, 'rb') as f:
        if accu is None:
            accu = np.load(f)
        else:
            accu += np.load(f)

if accu is None:
    raise ValueError("No histograms were loaded, please check the inputs")

# now accu should contain the combined histogram, we can save it to the new location
fs, outpath = get_hist_path(
    args.location,
    args.config_suite,
    args.runtag,
    args.dataset,
    args.objsyst,
    args.wtsyst,
    args.table,
    cov = args.cov,
    statN = args.statN,
    statK = args.statK
)

print("Writing combined histogram to", outpath)
with fs.open(outpath, 'wb') as f:
    np.save(f, accu)