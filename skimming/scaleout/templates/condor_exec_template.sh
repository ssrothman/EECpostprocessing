#!/bin/bash

set -e

JOB_INDEX=$1
echo "Starting job $JOB_INDEX"
echo "Running on host $(hostname)"

source /afs/cern.ch/user/d/dponman/EECpostproc/venv/bin/activate
source /afs/cern.ch/user/d/dponman/EECpostproc/env.sh

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
