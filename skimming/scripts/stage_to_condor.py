#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Stage skimming workspace to HTCONDOR queue")
parser.add_argument("where", type=str, help="Directory of workspace to stage")
parser.add_argument("name", type=str, help='condor job name')
parser.add_argument('--files-per-job', type=int, default=1,
                    help='Number of input files to process per job (default: 1)')
args = parser.parse_args()

from skimming.scaleout.condor import stage_via_condor
import os
stage_via_condor(args.where, args.name, args.files_per_job)
print("Submit with: ")
print("  condor_submit %s"%os.path.join(args.where, "condor_submit.sh"))