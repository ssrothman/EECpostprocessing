#!/usr/bin/env python3

import argparse
import os
import subprocess

from binning.scaleout.common import find_missing_file

parser = argparse.ArgumentParser(description="Stage binning workspace to HTCondor")
parser.add_argument("where", type=str, help="Workspace directory")
parser.add_argument("--commands-per-job", type=int, default=1, help="Commands to run in each condor process")
parser.add_argument("--mem", type=str, default="2gb", help="Requested memory")
parser.add_argument("--cpus", type=int, default=1, help="Requested CPUs")
parser.add_argument("--exec", action="store_true", help="Submit immediately")
parser.add_argument("--missing", action="store_true", help="Use highest commands_missing_N.txt instead of commands.txt")
args = parser.parse_args()

commands_file = find_missing_file(args.where, required=True) if args.missing else 'commands.txt'

from binning.scaleout.condor import stage_via_condor

ncommands = stage_via_condor(
    working_dir=args.where,
    commands_per_job=args.commands_per_job,
    mem=args.mem,
    cpus=args.cpus,
    commands_file=commands_file,
)

if ncommands == 0:
    print("No commands to submit.")

if args.exec:
    result = subprocess.run("condor_submit condor_submit.sh", shell=True, capture_output=True, cwd=args.where)
    print(result.stdout.decode())
    print(result.stderr.decode())
    if result.returncode != 0:
        raise RuntimeError("Condor submission failed")
else:
    print("Submit with:")
    print("  condor_submit %s" % os.path.join(args.where, "condor_submit.sh"))
