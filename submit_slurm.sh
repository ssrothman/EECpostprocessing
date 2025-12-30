#!/bin/bash -l

#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=CPU
#SBATCH --mem=MEM
#SBATCH --partition=submit
#SBATCH --job-name=NAME
#SBATCH --array=0-NJOBS
#SBATCH --output=slurm/NAME_%a.out
#SBATCH --error=slurm/NAME_%a.err

cd WORKINGDIR

export PYTHONUNBUFFERED=1

python skimscript.py i=$SLURM_ARRAY_TASK_ID
