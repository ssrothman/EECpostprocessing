#!/usr/bin/env python

import argparse


from general.datasets.datasets import get_target_files, lookup_dataset
from general.fslookup.location_lookup import location_lookup, lookup_hostid
from skimming.objects.AllObjects import AllObjects
from skimming.util.check_missing import check_workspace


parser = argparse.ArgumentParser(description="Use skimming results to determine which input root files remain to be processed")
parser.add_argument('path', type=str, help="Path to the skimming workspace to check")
parser.add_argument('-j', type=int, default=1, help="Number of parallel workers for input-file checks (default: 1)")
parser.add_argument('--write-missing', type=str, default=None, help="Write missing input files to this file")
parser.add_argument('--write-glitched', type=str, default=None, help="Write glitched skim results (unmatched files) to this file")
parser.add_argument('--dont-short-circuit', action='store_true', help="Don't exit early when len(target_files) == len(skimresults). Removes the assumption that each skim result correctly corresponds to a unique target file and there are no glitched skimresults. This assumption should always be true, but you can enable this flag to check explicitely")
args = parser.parse_args()

missing_files, skimresults = check_workspace(
    workspace_path=args.path,
    dont_short_circuit=args.dont_short_circuit,
    j=args.j
)

print("Missing files:", len(missing_files))
for f in sorted(missing_files):
    print(f)

if args.write_missing and len(missing_files) > 0:
    with open(args.write_missing, 'w') as f:
        for target_file in sorted(missing_files):
            f.write(target_file + '\n')
    print("Missing files written to %s" % args.write_missing)

    if args.write_glitched:
        with open(args.write_glitched, 'w') as f:
            for skim_file in sorted(skimresults):
                f.write(skim_file + '\n')
        print("Glitched skim results written to %s" % args.write_glitched)