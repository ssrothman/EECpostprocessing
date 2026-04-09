#!/bin/bash

set -e

JOB_INDEX=$1
echo "Starting job $JOB_INDEX"
echo "Running on host $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
export PYTHONHOME=/cvmfs/sft.cern.ch/lcg/releases/Python/3.11.9-2924c/x86_64-el9-gcc13-opt
unset PYTHONPATH
source /afs/cern.ch/user/d/dponman/EECpostproc/venv/bin/activate
export PYTHONPATH=/afs/cern.ch/user/d/dponman/EECpostproc:$PYTHONPATH

cd WORKINGDIR

for i in $(seq 0 $((FILES_PER_JOB - 1))); do
    index=$((FILES_PER_JOB * JOB_INDEX + i))
    echo "Processing file with index $index"
    if [ $index -ge NFILES ]; then
        echo "Index $index exceeds total files NFILES, skipping."
        continue
    fi
    python skimscript.py $index
done
