#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description="Resubmit all failed HTCONDOR jobs for skimming workspaces in a directory")
parser.add_argument("where", type=str, help="Directory containing workspaces to resubmit")
parser.add_argument('--resub-idle', action='store_true', help="Also resubmit idle jobs")
parser.add_argument('--no-resub-running', action='store_true', help="Do not resubmit running jobs")
parser.add_argument("--skip-still-running", action='store_true', help="Skip workspaces that still have running jobs")
parser.add_argument('--exec', action='store_true', help="Directly execute condor_submit command after preparing resubmission scripts")
parser.add_argument('--dont-check-singularity', action='store_true', help="Don't check the .err files for singularity errors")
parser.add_argument('-j', type=int, default=1, help="Number of parallel resubmissions to do (default: 1)")
args = parser.parse_args()

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_command(wd):
    cmd = 'resubmit_condor.py %s' % wd
    if args.resub_idle:
        cmd += ' --resub-idle'
    if args.no_resub_running:
        cmd += ' --no-resub-running'
    if args.skip_still_running:
        cmd += ' --skip-still-running'
    if args.exec:
        cmd += ' --exec'
    if args.dont_check_singularity: 
        cmd += ' --dont-check-singularity'
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return wd, output.returncode, output.stdout, output.stderr

subdirs = os.listdir(args.where)
workspaces = []
for sd in sorted(subdirs):
    fullpath = os.path.join(args.where, sd)
    if os.path.isdir(fullpath):
        workspaces.append(fullpath)

if len(workspaces) == 0:
    print("No workspaces found in %s" % args.where)
    exit(0)

for ws in workspaces:
    print("Queueing workspace %s" % ws)

max_workers = max(1, args.j)
failures = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_workspace = {executor.submit(run_command, ws): ws for ws in workspaces}
    for future in as_completed(future_to_workspace):
        workspace = future_to_workspace[future]
        wd, returncode, stdout, stderr = future.result()
        print("Resubmission output for %s:" % wd)
        if stdout.strip():
            print(stdout)
        if returncode != 0:
            if stderr.strip():
                print(stderr)
            failures.append(workspace)

if len(failures) > 0:
    raise RuntimeError("Resubmission failed for workspace(s): %s" % ", ".join(failures))