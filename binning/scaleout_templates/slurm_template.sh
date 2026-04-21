#!/bin/bash -l

#SBATCH --time=TIME
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=CPUS
#SBATCH --mem=MEM
#SBATCH --partition=submit
#SBATCH --job-name=NAME
#SBATCH --array=0-NJOBS_MINUS_ONE
#SBATCH --output=slurm/NAME_%A_%a.out
#SBATCH --error=slurm/NAME_%A_%a.err

set -e

cd WORKINGDIR

export PYTHONUNBUFFERED=1

JOB_INDEX=$SLURM_ARRAY_TASK_ID
echo "Starting job $JOB_INDEX on host $(hostname)"

for i in $(seq 0 $((COMMANDS_PER_JOB - 1))); do
	index=$((COMMANDS_PER_JOB * JOB_INDEX + i))
	if [ $index -ge NCOMMANDS ]; then
		continue
	fi

	command=$(sed -n "$((index + 1))p" commands.txt)
	echo "Running command index $index: $command"
	/bin/bash -lc "$command"
done