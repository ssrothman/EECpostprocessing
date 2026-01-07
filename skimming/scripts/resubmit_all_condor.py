#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description="Resubmit all failed HTCONDOR jobs for skimming workspaces in a directory")
parser.add_argument("where", type=str, help="Directory containing workspaces to resubmit")
parser.add_argument('--resub-idle', action='store_true', help="Also resubmit idle jobs")
parser.add_argument('--no-resub-running', action='store_true', help="Do not resubmit running jobs")
parser.add_argument("--skip-still-running", action='store_true', help="Skip workspaces that still have running jobs")
parser.add_argument('--exec', action='store_true', help="Directly execute condor_submit command after preparing resubmission scripts")
args = parser.parse_args()

import os
import subprocess

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
    output = subprocess.run(cmd, shell=True, capture_output=True)
    print(output.stdout.decode())
    if output.returncode != 0:
        print(output.stderr.decode())
        raise RuntimeError("Resubmission failed for workspace %s" % wd)

subdirs = os.listdir(args.where)
for sd in subdirs:
    fullpath = os.path.join(args.where, sd)
    if os.path.isdir(fullpath):
        print("Resubmitting workspace in %s" % fullpath)
        run_command(fullpath)