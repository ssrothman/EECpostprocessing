#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='binscript for scaleout processing.')

parser.add_argument("runtag", type=str, help="Runtag to process")
parser.add_argument("dataset", type=str, help="Dataset to process")
parser.add_argument("objsyst", type=str, help="Object systematic variation to process")
parser.add_argument('table', type=str, help='Table name')

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

parser.add_argument('--bincfg', type=str, help='name of binning config to use',
                    default=None)

parser.add_argument('--evtwt', type=str,
                    default='nominal')

parser.add_argument('--cov', action='store_true', help='Covariance computation')

args = parser.parse_args()

# imports
import json
from binning.main import build_hist, build_transfer_config, fill_cov, fill_hist
from general.datasets.datasets import location_lookup
from general.fslookup.location_lookup import lookup_hostid
from general.fslookup.skim_path import lookup_skim_path
import os.path
import pyarrow.dataset as ds
import numpy as np
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning

if args.bincfg is None:
    args.bincfg = '_'.join(args.table.split('_')[:-1])

# load binning config
binpkgpath = os.path.dirname(os.path.dirname(__file__))
bincfgpath = os.path.join(
    binpkgpath,
    'config',
    args.bincfg
)
with open(bincfgpath + '.json') as f:
    bincfg = json.load(f)

if args.table.endswith('reco'):
    thebinning = bincfg['reco']
elif args.table.endswith('gen'):
    thebinning = bincfg['gen']
elif args.table.endswith('transfer'):
    thebinning = build_transfer_config(
        bincfg['gen'],
        bincfg['reco']
    )
else:
    raise NotImplementedError("Only gen, reco, transfer can be pre-binned!")

fs, skimpath = lookup_skim_path(
    args.location,
    args.config_suite,
    args.runtag,
    args.dataset,
    args.objsyst,
    args.table
)

H = build_hist(thebinning)
dataset = ds.dataset(skimpath, format='parquet', filesystem=fs)

if args.table.endswith('transfer'):
    itemwt = 'wt_reco'
    ab = ArbitraryGenRecoBinning()
    Hreco = build_hist(bincfg['reco'])
    Hgen = build_hist(bincfg['gen'])
    ab.setup_from_histograms(
        Hreco,
        Hgen
    )
else:
    itemwt = 'wt'
    ab = ArbitraryBinning()
    ab.setup_from_histogram(H)

if args.cov:
    thefun = fill_cov
else:
    thefun = fill_hist

result = thefun(
    H, dataset,
    'wt_%s' % args.evtwt,
    itemwt = itemwt,
    statN = args.statN,
    statK = args.statK,
    reweight = None #for now
)

outpath = os.path.join(
    os.path.dirname(skimpath),
    '%s_BINNED' % args.table
)

if args.cov:
    outpath += '_covmat'

outpath += '_%s' % args.evtwt

if args.statN > 0:
    outpath += '_%dstat%d' % (args.statN, args.statK)

if args.cov:
    halfshape = result.shape[:len(result.shape)//2]
    thelen = np.prod(halfshape)
    result = result.reshape((thelen, thelen)) # type: ignore
else:
    result = result.values(flow=True).ravel() # type: ignore
    
print("Writing result to", outpath)
with fs.open(outpath + '.npy', 'wb') as f:
    np.save(f, result)

bincfg_path = os.path.join(
    os.path.dirname(skimpath),
    '%s_bincfg.json' % args.table
)

if fs.exists(bincfg_path):
    oldbinning = type(ab)() # use same type as ab [either ArbitraryBinning or ArbitraryGenRecoBinning]

    with fs.open(bincfg_path, 'r') as f:
        oldbinning.from_dict(json.load(f))
    
    if oldbinning != ab:
        raise RuntimeError("Binning config mismatch for existing bincfg at %s" % bincfg_path)
else:
    print("Writing binning config to", bincfg_path)
    with fs.open(bincfg_path, 'w') as f:
        json.dump(ab.to_dict(), f, indent=4)