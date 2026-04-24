#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Stage skimming workspace to SLURM queue")
parser.add_argument("where", type=str, help="Directory of workspace to stage")
parser.add_argument("name", type=str, help='slurm job name')
parser.add_argument('--files-per-job', type=int, default=1,
                    help='Number of input files to process per job (default: 1)')
parser.add_argument('--mem', type=str, default='4gb',
                    help='Requested memory per job (default: 4gb)')
parser.add_argument('--exec', action='store_true', help='Whether to execute the staging command immediately (default: False)')
args = parser.parse_args()

from skimming.scaleout.slurm import stage_via_slurm
import os
if args.mem.strip() == '':
    raise RuntimeError("--mem must not be empty")

stage_via_slurm(args.where, args.name, args.files_per_job, mem=args.mem)

if not args.exec:
    print("Submit with: ")
    print("  sbatch %s"%os.path.join(args.where, "submit_slurm.sh"))
else:
    cmd = 'sbatch submit_slurm.sh'
    import subprocess
    output = subprocess.run(cmd, shell=True, capture_output=True, cwd=args.where)
    print(output.stdout.decode())
    print(output.stderr.decode()) 
    if output.returncode != 0:
        raise RuntimeError("Failed to submit SLURM jobs")
