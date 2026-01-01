#!/bin/bash

set -e # Exit with nonzero exit code if anything fails 

JOB_INDEX=$1
echo "Starting job $JOB_INDEX"
echo "Running on host $(hostname)"

git clone https://github.com/ssrothman/EECpostprocessing.git -b refactor --depth 1 --no-tags --recurse-submodules --shallow-submodules --single-branch
cd EECpostprocessing/
source env.sh

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