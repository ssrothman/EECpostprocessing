#!/usr/bin/env -S python
"""
Build pre-binned histograms for stack datasets by summing histograms from subdatasets.

This script reads the pre-binned histograms from all subdatasets and sub-stacks
in a stack configuration, sums them together, and writes out the combined histogram
and binning configuration.
"""

import argparse
import os
import json
import uuid
from typing import Literal
import numpy as np

from general.datasets.datasets import lookup_count, lookup_dataset
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.checksum import checksum_file
from skimming.tables.expand_tables import expand_tables, table_names

def read_hist(
    location: str,
    config_suite: str,
    runtag: str,
    dataset: str,
    objsyst: str,
    wtsyst: str,
    table: str,
    is_stack : bool,
    cov: bool,
    statN: int = -1,
    statK: int = -1,
) -> tuple[np.ndarray, ArbitraryBinning | ArbitraryGenRecoBinning]:
    
    if is_stack:
        stackcfg = lookup_dataset(
            'stacks',
            dataset
        )

        accu: np.ndarray | None = None
        binning: ArbitraryBinning | ArbitraryGenRecoBinning | None = None

        for subdset in stackcfg['dsets']:
            nextvals, nextbinning = read_hist(
                location,
                config_suite,
                runtag,
                subdset,
                objsyst,
                wtsyst,
                table,
                is_stack=False,
                cov=cov,
                statN=statN,
                statK=statK
            )

            if accu is None:
                accu = nextvals
                binning = nextbinning
            else:
                if binning != nextbinning:
                    raise ValueError(f"Binning mismatch between subdatasets in stack {dataset}")
                if accu.shape != nextvals.shape:
                    raise ValueError(f"Shape mismatch between subdataset histograms in stack {dataset}")
                
                accu += nextvals

        for substack in stackcfg['stacks']:
            nextvals, nextbinning = read_hist(
                location,
                config_suite,
                runtag,
                substack,
                objsyst,
                wtsyst,
                table,
                is_stack=True,
                cov=cov,
                statN=statN,
                statK=statK
            )

            if accu is None:
                accu = nextvals
                binning = nextbinning
            else:
                if binning != nextbinning:
                    raise ValueError(f"Binning mismatch between sub-stacks in stack {dataset}")
                if accu.shape != nextvals.shape:
                    raise ValueError(f"Shape mismatch between sub-stack histograms in stack {dataset}")
            
            accu += nextvals

        assert(accu is not None and binning is not None), "No subdatasets or sub-stacks found in stack configuration"

        return accu, binning
    else:
        fs, histpath = get_hist_path(
            location,
            config_suite,
            runtag,
            dataset,
            objsyst,
            wtsyst,
            table,
            cov,
            statN,
            statK
        )

        _, binningpath = get_hist_bincfg_path(
            location,
            config_suite,
            runtag,
            dataset,
            objsyst,
            table
        )

        print("Reading histogram values from", histpath)
        with fs.open(histpath, 'rb') as f:
            values : np.ndarray = np.load(f)

        binning = ArbitraryGenRecoBinning() if '_transfer' in table else ArbitraryBinning()
        print("Reading histogram binning config from", binningpath)
        with fs.open(binningpath, 'r') as f:
            binning.from_dict(json.load(f))
        
        dset_info = lookup_dataset(runtag, dataset)
        if 'xsec' in dset_info:
            xsec : float = dset_info['xsec']
            count : int | float = lookup_count(runtag, dataset)
            wt : float = 1000 * xsec * 1.0 / count
            
            if cov:
                values *= wt * wt
            else:
                values *= wt
        
        return values, binning

def handle_one_table(
        args : argparse.Namespace,
        table : str
):
    
    if not args.nocheck:
        fs, outpath = get_hist_path(
            args.location,
            args.config_suite,
            args.runtag,
            args.stack_name,
            args.objsyst,
            args.wtsyst,
            table,
            args.cov,
            args.statN,
            args.statK
        )

        if fs.exists(outpath):
            print(f"Output file {outpath} already exists. Use --nocheck to overwrite.")
            return 
    
    values, binning = read_hist(
        args.location,
        args.config_suite,
        args.runtag,
        args.stack_name,
        args.objsyst,
        args.wtsyst,
        table,
        is_stack=True,
        cov=args.cov,
        statN=args.statN,
        statK=args.statK
    )

    fs, outpath = get_hist_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.stack_name,
        args.objsyst,
        args.wtsyst,
        table,
        args.cov,
        args.statN,
        args.statK
    )

    fs.makedirs(os.path.dirname(outpath), exist_ok=True)

    print("Writing output histogram to", outpath)
    with fs.open(outpath, 'wb') as f:
        np.save(f, values)

    _, bincfg_path = get_hist_bincfg_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.stack_name,
        args.objsyst,
        table
    )
    print("Writing output binning config to", bincfg_path)
    with fs.open(bincfg_path, 'w') as f:
        json.dump(binning.to_dict(), f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build pre-binned histograms for stack datasets'
    )
    
    parser.add_argument('stack_name', type=str, help='Name of the stack dataset')
    parser.add_argument('objsyst', type=str, help='Object systematic variation')
    parser.add_argument('wtsyst', type=str, help='Weight systematic variation')
    
    parser.add_argument('--location', type=str, default='xrootd-submit',
                        help='Filesystem location for output')
    parser.add_argument('--config-suite', type=str, default='BasicConfig',
                        help='Configuration suite to use')
    parser.add_argument('--runtag', type=str, required=True,
                        help='Run tag/campaign name')
    
    parser.add_argument('--tables', type=str, nargs='+', required=True)

    parser.add_argument('--statN', type=int, default=-1,
                        help='N for statistical splits (-1 for no split)')
    parser.add_argument('--statK', type=int, default=-1,
                        help='K for statistical splits')
    
    parser.add_argument('--cov', action='store_true',
                        help='Covariance matrix')
    parser.add_argument('--nocheck', action='store_true',
                        help='Do not check if output file already exists')
    
    args = parser.parse_args()
    
    tables = table_names(expand_tables(args.tables))

    for table in tables:
        print(f"Processing table {table}...")
        try:
            handle_one_table(args, table)
        except Exception as e:
            print(f"Error processing table {table}: {e}")