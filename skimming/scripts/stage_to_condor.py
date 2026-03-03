#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Stage skimming workspace to HTCONDOR queue")
parser.add_argument("where", type=str, help="Directory of workspace to stage")
parser.add_argument("name", type=str, help='condor job name')
parser.add_argument('--files-per-job', type=int, default=1,
                    help='Number of input files to process per job (default: 1)')
parser.add_argument('--exec', action='store_true',
                    help='If set, actually execute the condor staging (otherwise just create scripts)')
args = parser.parse_args()

from skimming.scaleout.condor import stage_via_condor
import os
stage_via_condor(args.where, args.name, args.files_per_job)

if args.exec:
    cmd = 'condor_submit condor_submit.sh'
    import subprocess
    output = subprocess.run(cmd, shell=True, capture_output=True, 
                   cwd=args.where)
    print(output.stdout.decode())
    print(output.stderr.decode())
    if output.returncode != 0:
        raise RuntimeError("Condor submission failed")
else:
    print("Submit with: ")
    print("  condor_submit %s"%os.path.join(args.where, "condor_submit.sh"))