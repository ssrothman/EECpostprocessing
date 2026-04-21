#!/usr/bin/env -S python

import argparse
import os
import subprocess


parser = argparse.ArgumentParser(description="Run one command from a binning scaleout workspace")
parser.add_argument("i", type=int, help="Command index to execute")
parser.add_argument(
    "--commands-file",
    type=str,
    default="commands.txt",
    help="Path to command manifest (default: commands.txt)",
)
args = parser.parse_args()

commands_file = args.commands_file
if not os.path.isabs(commands_file):
    commands_file = os.path.join(os.getcwd(), commands_file)

command = None
with open(commands_file) as f:
    for i, line in enumerate(f):
        if i == args.i:
            command = line.strip()
            break

if command is None:
    raise IndexError(f"Command index {args.i} out of range")

print(f"Running command index {args.i}: {command}")
subprocess.run(command, shell=True, check=True, cwd=os.getcwd())
