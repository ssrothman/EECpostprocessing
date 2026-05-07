#!/usr/bin/env -S python

try:
    import directcov
except ImportError:
    print("Warning: the 'directcov' package (required for covariance computation) is not installed. Please install it to use the --cov option.")

import argparse
import numpy as np

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

parser.add_argument('--justcheck', action='store_true', help='Just check for existing output, do not run anything')
parser.add_argument('--validate-existing', action='store_true', help='Validate existing output (does nothing if --nocheck is specified)')
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
        print("looking for", path)
        if fs.exists(path):
            if args.validate_existing:
                with fs.open(path, 'rb') as f:
                    try:
                        _ = np.load(f)
                        print("Output already exists for table %s at %s, skipping (use --nocheck to override)" % (table, path))
                        continue
                    except Exception as e:
                        print("Error occurred while loading existing output for table %s at %s: %s" % (table, path, str(e)))
            
            else:
                print("Output already exists for table %s at %s, skipping (use --nocheck to override)" % (table, path))
                continue
        else:
            tables_to_run.append(table)
else:
    tables_to_run = tables

if args.justcheck:
    if len(tables_to_run) == 0:
        exit(0)
    else:
        print("Tables that need to be run:")
        for table in tables_to_run:
            print(table)
        exit(1)

for table in tables_to_run:
    print("Processing table", table)
    run_table(args, table)
