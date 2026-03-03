#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Resubmit failed HTCONDOR jobs for a skimming workspace")
parser.add_argument("where", type=str, help="Directory of workspace to resubmit")
parser.add_argument('--resub-idle', action='store_true', help="Also resubmit idle jobs")
parser.add_argument('--no-resub-running', action='store_true', help="Do not resubmit running jobs")
parser.add_argument('--exec', action='store_true', help="Directly execute condor_submit command after preparing resubmission scripts")
parser.add_argument('--skip-still-running', action='store_true', help="Skip workspaces that still have running jobs")
args = parser.parse_args()

#first, discover the condor cluster id 
import os
condorlogs = os.listdir(os.path.join(args.where, "condor"))
clusterids = set()
for fn in condorlogs:
    if fn.endswith(".log"):
        parts = fn.split("_")
        clusterid = int(parts[-2])
        clusterids.add(clusterid)

if len(clusterids) == 0:
    raise RuntimeError("No condor logs found in workspace")
elif len(clusterids) > 1:
    import warnings
    print("WARNING: Multiple cluster IDs found in condor logs. Using most recent (ie largest job number).")
    clusterid = max(clusterids)
else:
    clusterid = clusterids.pop()

# get exit codes for completed jobs
cmd = "condor_history %d -format '%%v\t' Args -format '%%v\n' ExitCode" % clusterid
import subprocess
result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=args.where)
lines = result.stdout.split("\n")
failed_jobs = set()
for line in lines:
    if line.strip() == "":
        continue
    parts = line.split("\t")
    procid = int(parts[0])
    exitcode = int(parts[1])
    if exitcode != 0:
        failed_jobs.add(procid)
print("Failed jobs", failed_jobs)

# get info on running or held jobs
#condor_q 
cmd = "condor_q %d -nobatch -format '%%v\t' Args -format '%%v\n' JobStatus" % clusterid
result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=args.where)
lines = result.stdout.split("\n")
idle_jobs = set()
running_jobs = set()
removed_jobs = set()
completed_jobs = set()
held_jobs = set()
transfering_jobs = set()
suspended_jobs = set()
for line in lines:
    if line.strip() == "":
        continue
    parts = line.split("\t")
    procid = int(parts[0])
    jobstatus = int(parts[1])
    if jobstatus == 1 or jobstatus == 0:
        idle_jobs.add(procid)
    elif jobstatus == 2:
        running_jobs.add(procid)
    elif jobstatus == 3:
        removed_jobs.add(procid)    
    elif jobstatus == 4:
        completed_jobs.add(procid)
    elif jobstatus == 5:
        held_jobs.add(procid)
    elif jobstatus == 6:
        transfering_jobs.add(procid)
    elif jobstatus == 7:
        suspended_jobs.add(procid)
    else:
        print("Unknown job status %d for job %d"%(jobstatus, procid))

if len(idle_jobs) + len(running_jobs) + len(transfering_jobs) + len(held_jobs) + len(suspended_jobs) > 0:
    print("There are still jobs in the condor queue:")
    print("  Idle jobs: ", idle_jobs)
    print("  Running jobs: ", running_jobs)
    print("  Transfering jobs: ", transfering_jobs)
    print("  Held jobs: ", held_jobs)
    print("  Suspended jobs: ", suspended_jobs)
    if args.skip_still_running:
        print("Skipping resubmission since some jobs are still running.")
        exit(0)

#only resubmit idle jobs if requested
if args.resub_idle and len(idle_jobs) > 0:
    failed_jobs.update(idle_jobs)
    print("Also resubmitting idle jobs: ", idle_jobs)

# by default DO resubmit running jobs, unless --no-resub-running is given
if not args.no_resub_running and len(running_jobs) > 0:
    failed_jobs.update(running_jobs)
    print("Also resubmitting running jobs: ", running_jobs)

#assume removed and completed jobs were handled by condor_history
if len(held_jobs) > 0:
    failed_jobs.update(held_jobs)
    print("Also resubmitting held jobs: ", held_jobs)


#transfering jobs are probably ok, do NOT resubmit

if len(suspended_jobs) > 0:
    failed_jobs.update(suspended_jobs)
    print("Also resubmitting suspended jobs: ", suspended_jobs)

#now, sort the failed jobs into ranges for resubmission
if len(failed_jobs) == 0:
    print("All jobs succeeded! Nothing to resubmit.")
    exit(0)

print("Final list of jobs to resubmit: ", failed_jobs)

#next, find the most recent condor submission details and make a modified copy for resubmission
import glob
submit_files = glob.glob('condor_submit*.sh', root_dir=args.where)
if len(submit_files) == 0:
    raise RuntimeError("No condor template files found in workspace")

#make keys for sorting the template files
#if the file is named submit_condor.sh, assign it number 0
#otherwise, it should be submit_condor_X.sh where X is an int
#extract that int for the key!

file_numbers = []
for fname in submit_files   :
    parts = fname.split(".")[0].split("_")
    if len(parts) == 2:
        file_numbers.append(0)
    else:
        file_numbers.append(int(parts[2]))
sorted_templates = [x for _,x in sorted(zip(file_numbers, submit_files))]  

subfile = os.path.join(args.where, sorted_templates[-1])  #most recent template file
next_number = max(file_numbers)+1  

import shutil
new_template = os.path.join(args.where, "condor_submit_%d.sh"%next_number)
shutil.copyfile(subfile, new_template)

with open(os.path.join(args.where, 'job_indices_%d.txt'%next_number), 'w') as f:
    for job in sorted(failed_jobs):
        f.write("%d\n"%job)

#finally, modify the new scripts to point to the correct places
sed_command = "sed -i 's/Process/index/g' %s" % (
    new_template
)
subprocess.run(sed_command, shell=True, check=True)
sed_command = "sed -i 's/queue .*/queue index from %s /g' %s" % (
    'job_indices_%d.txt'%next_number,
    new_template
)
subprocess.run(sed_command, shell=True, check=True)

if args.exec:
    cmd = "condor_submit %s" % os.path.basename(new_template)
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=args.where)
    print(output.stdout)
    if output.returncode != 0:
        print(output.stderr)
        raise RuntimeError("Condor submission failed")
else:
    print("Resubmission script created.")
    print("Submit with: ")
    print("  condor_submit %s"%new_template)