#!/usr/bin/env -S python

import argparse

parser = argparse.ArgumentParser(description='binscript for scaleout processing.')

parser.add_argument("runtag", type=str, help="Runtag to process")
parser.add_argument("dataset", type=str, help="Dataset to process")
parser.add_argument("objsyst", type=str, help="Object systematic variation to process")
parser.add_argument('table', type=str, nargs='?', default=None, help='Table name')
parser.add_argument('--tables', type=str, nargs='+', default=None,
                    help='Table names to process in sequence')

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

if args.table is not None and args.tables is not None:
    parser.error("Specify either positional table or --tables, not both")
if args.table is None and args.tables is None:
    parser.error("Must specify a table (positional) or --tables")

tables = args.tables if args.tables is not None else [args.table]

# imports
import json
from binning.main import build_hist, build_transfer_config, fill_cov, fill_hist
from general.datasets.datasets import location_lookup
from general.fslookup.location_lookup import lookup_hostid
from general.fslookup.skim_path import lookup_skim_path
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
import os.path
import pyarrow.dataset as ds
import numpy as np
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning

def run_table(table: str):
    bincfg_name = args.bincfg
    if bincfg_name is None:
        bincfg_name = '_'.join(table.split('_')[:-1])

    # load binning config
    binpkgpath = os.path.dirname(os.path.dirname(__file__))
    bincfgpath = os.path.join(
        binpkgpath,
        'config',
        bincfg_name
    )
    with open(bincfgpath + '.json') as f:
        bincfg = json.load(f)

    if table.endswith('Reco'):
        thebinning = bincfg['reco']
    elif table.endswith('Gen'):
        thebinning = bincfg['gen']
    elif table.endswith('transfer'):
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
        table
    )

    H, prebinned = build_hist(thebinning)
    dataset = ds.dataset(skimpath, format='parquet', filesystem=fs)

    if table.endswith('transfer'):
        itemwt = 'wt_reco'
        ab = ArbitraryGenRecoBinning()
        Hreco, prebinned_reco = build_hist(bincfg['reco'])
        Hgen, prebinned_gen = build_hist(bincfg['gen'])
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
        H, prebinned,
        dataset,
        'wt_%s' % args.evtwt,
        itemwt = itemwt,
        statN = args.statN,
        statK = args.statK,
        reweight = None #for now
    )

    _, outpath = get_hist_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        args.evtwt,
        table,
        args.cov,
        args.statN,
        args.statK
    )

    if args.cov:
        halfshape = result.shape[:len(result.shape)//2]
        thelen = np.prod(halfshape)
        output = result.reshape((thelen, thelen)) # type: ignore
    else:
        output = result.values(flow=True).ravel() # type: ignore
        
    print("Writing result to", outpath)
    with fs.open(outpath, 'wb') as f:
        np.save(f, output)

    _, bincfg_path = get_hist_bincfg_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        table
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


for table in tables:
    print("Processing table", table)
    run_table(table)