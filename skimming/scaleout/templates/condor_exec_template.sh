#!/bin/bash

set -euxo pipefail # Exit with nonzero exit code if anything fails 

JOB_INDEX=$1
echo "Starting job $JOB_INDEX"
echo "Running on host $(hostname)"

git clone git@github.com:ssrothman/EECpostprocessing.git -b refactor
cd postprocessing/
source env.sh
git submodule update --init --recursive

cd ../

for i in $(seq 0 $((FILES_PER_JOB - 1))); do
    index=$((FILES_PER_JOB * JOB_INDEX + i))
    echo "Processing file with index $index"
    if [ $index -ge NFILES ]; then
        echo "Index $index exceeds total files NFILES, skipping."
        continue
    fi
    python skimscript.py $index
done