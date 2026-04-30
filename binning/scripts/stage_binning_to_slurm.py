#!/usr/bin/env python3

import argparse
import os
import subprocess

from binning.scaleout.common import find_missing_file

parser = argparse.ArgumentParser(description="Stage binning workspace to SLURM")
parser.add_argument("where", type=str, help="Workspace directory")
parser.add_argument("name", type=str, help="Job name")
parser.add_argument("--commands-per-job", type=int, default=1, help="Commands to run in each array task")
parser.add_argument("--time", type=str, default="01:00:00", help="Time limit (HH:MM:SS)")
parser.add_argument("--mem", type=str, default="4G", help="Memory per task")
parser.add_argument("--cpus", type=int, default=1, help="CPUs per task")
parser.add_argument("--exec", action="store_true", help="Submit immediately")
parser.add_argument("--missing", action="store_true", help="Use highest commands_missing_N.txt instead of commands.txt")
args = parser.parse_args()

commands_file = find_missing_file(args.where, required=args.missing) if args.missing else 'commands.txt'

from binning.scaleout.slurm import stage_via_slurm

ncommands = stage_via_slurm(
    working_dir=args.where,
    name=args.name,
    commands_per_job=args.commands_per_job,
    time=args.time,
    mem=args.mem,
    cpus=args.cpus,
    commands_file=commands_file,
)

if ncommands == 0:
    print("No commands to submit.")

if args.exec:
    result = subprocess.run("sbatch submit_slurm.sh", shell=True, capture_output=True, cwd=args.where)
    print(result.stdout.decode())
    print(result.stderr.decode())
    if result.returncode != 0:
        raise RuntimeError("SLURM submission failed")
else:
    print("Submit with:")
    print("  sbatch %s" % os.path.join(args.where, "submit_slurm.sh"))
