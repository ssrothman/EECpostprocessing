#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Stage skimming workspace to SLURM queue")
parser.add_argument("where", type=str, help="Directory of workspace to stage")
parser.add_argument("name", type=str, help='slurm job name')
parser.add_argument('--files-per-job', type=int, default=1,
                    help='Number of input files to process per job (default: 1)')
args = parser.parse_args()

from skimming.scaleout.slurm import stage_via_slurm
import os
stage_via_slurm(args.where, args.name, args.files_per_job)
print("Submit with: ")
print("  sbatch %s"%os.path.join(args.where, "submit_slurm.sh"))
