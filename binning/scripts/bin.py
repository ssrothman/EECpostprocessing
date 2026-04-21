#!/usr/bin/env -S python

import argparse

parser = argparse.ArgumentParser(description='Build histogram outputs for one or more tables.')

parser.add_argument("runtag", type=str, help="Runtag to process")
parser.add_argument("dataset", type=str, help="Dataset to process")
parser.add_argument("objsyst", type=str, help="Object systematic variation to process")
parser.add_argument('wtsyst', type=str, help="Weight systematic variation to process")
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

parser.add_argument('--cov', action='store_true', help='Covariance computation')

parser.add_argument('--nocheck', action='store_true', help='Skip checks for existing output')

args = parser.parse_args()

if args.table is not None and args.tables is not None:
    parser.error("Specify either positional table or --tables, not both")
if args.table is None and args.tables is None:
    parser.error("Must specify a table (positional) or --tables")

tables = args.tables if args.tables is not None else [args.table]

# imports
from general.fslookup.hist_lookup import get_hist_path
from binning.run_table import run_table

tables_to_run = []

if not args.nocheck:
    for table in tables:
        fs, path = get_hist_path(
            args.location,
            args.config_suite,
            args.runtag,
            args.dataset,
            args.objsyst,
            args.wtsyst,
            table,
            args.cov,
            args.statN,
            args.statK
        )
        if fs.exists(path):
            print("Output already exists for table %s at %s, skipping (use --nocheck to override)" % (table, path))
            continue
        else:
            tables_to_run.append(table)
else:
    tables_to_run = tables

for table in tables_to_run:
    print("Processing table", table)
    run_table(args, table)
