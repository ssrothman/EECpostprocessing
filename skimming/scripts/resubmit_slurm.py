#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description="Resubmit failed SLURM jobs for a skimming workspace")
parser.add_argument("where", type=str, help="Directory of workspace to resubmit")
args = parser.parse_args()

#first, discover the slurm array id 
import os
slurmlogs = os.listdir(os.path.join(args.where, "slurm"))
arrayids = set()
for fn in slurmlogs:
    if fn.endswith(".err"):
        parts = fn.split("_")
        arrayid = int(parts[-2])
        arrayids.add(arrayid)

if len(arrayids) == 0:
    raise RuntimeError("No slurm logs found in workspace")
elif len(arrayids) > 1:
    import warnings
    print("WARNING: Multiple array IDs found in slurm logs. Using most recent (ie largest job number).")
    arrayid = max(arrayids)
else:
    arrayid = arrayids.pop()

#then, use some bash commands to find which jobs failed
import subprocess
cmd = "sacct -j SLURM_JOB_ID --format=JobId%50,State --noheader | grep -v COMPLETED | grep -v batch | grep -v extern | awk '{print $1}' | awk -F_ '{print $2}'".replace("SLURM_JOB_ID", str(arrayid))
result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=args.where)
failed_jobs = [int(x) for x in result.stdout.split("\n") if x.strip() != ""]

#then, sort the failed jobs into ranges for resubmission
failed_jobs = sorted(failed_jobs)
if len(failed_jobs) == 0:
    print("All jobs succeeded! Nothing to resubmit.")
    exit(0)

resubmit_ranges = []
start=0
for i in range(1,len(failed_jobs)):
    if failed_jobs[i] != failed_jobs[i-1]+1:
        resubmit_ranges.append( (failed_jobs[start], failed_jobs[i-1]) )
        start = i

resubmit_ranges.append( (failed_jobs[start], failed_jobs[-1]) )

ranges_str=''
for r in resubmit_ranges:
    if r[0] == r[1]:
        ranges_str += "%d,"%(r[0])
    else:
        ranges_str += "%d-%d,"%(r[0], r[1])
ranges_str = ranges_str[:-1]  #remove trailing comma
print("Resubmitting the following job ranges: %s"%ranges_str)

#next, find the most recent slurm template and make a modified copy for resubmission
import glob
template_files = glob.glob('submit_slurm*.sh', root_dir=args.where)
if len(template_files) == 0:
    raise RuntimeError("No slurm template files found in workspace")

#make keys for sorting the template files
#if the file is named submit_slurm.sh, assign it number 0
#otherwise, it should be submit_slurm_X.sh where X is an int
#extract that int for the key!

file_numbers = []
for fname in template_files:
    parts = fname.split(".")[0].split("_")
    if len(parts) == 2:
        file_numbers.append(0)
    else:
        file_numbers.append(int(parts[2]))
sorted_templates = [x for _,x in sorted(zip(file_numbers, template_files))]  

tfile = sorted_templates[-1]  #most recent template file
next_number = max(file_numbers)+1  

import shutil
new_template = os.path.join(args.where, "submit_slurm_%d.sh"%next_number)
shutil.copyfile(os.path.join(args.where, tfile), new_template)

#finally, modify the new template to have the correct array ranges
#use sed for this

sed_command = "sed -i 's/.*--array.*/%s/g' %s" % (
    "#SBATCH --array=%s"%ranges_str,
    new_template
)
subprocess.run(sed_command, shell=True, check=True)

print("Resubmission script created.")
print("Submit with: ")
print("  sbatch %s"%new_template)