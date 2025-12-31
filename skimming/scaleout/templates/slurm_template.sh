#!/bin/bash -l

#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --partition=submit
#SBATCH --job-name=NAME
#SBATCH --array=0-NJOBS
#SBATCH --output=slurm/NAME_%A_%a.out
#SBATCH --error=slurm/NAME_%A_%a.err

cd WORKINGDIR

export PYTHONUNBUFFERED=1

echo "Starting job $SLURM_ARRAY_TASK_ID"
echo "Running on host $(hostname)"

for i in $(seq 0 $((FILES_PER_JOB - 1))); do
    index=$((FILES_PER_JOB * SLURM_ARRAY_TASK_ID + i))
    echo "Processing file with index $index"
    if [ $index -ge NFILES ]; then
        echo "Index $index exceeds total files NFILES, skipping."
        continue
    fi
    python skimscript.py $index
done