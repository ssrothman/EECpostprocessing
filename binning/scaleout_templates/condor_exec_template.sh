#!/bin/bash

set -e # Exit with nonzero exit code if anything fails 

JOB_INDEX=$1
echo "Starting job $JOB_INDEX"
echo "Running on host $(hostname)"

git clone https://github.com/ssrothman/EECpostprocessing.git -b master --depth 1 --no-tags --recurse-submodules --shallow-submodules --single-branch
cd EECpostprocessing/
source env.sh

cd ../

for i in $(seq 0 $((COMMANDS_PER_JOB - 1))); do
    index=$((COMMANDS_PER_JOB * JOB_INDEX + i))
    if [ $index -ge NCOMMANDS ]; then
        echo "Index $index exceeds total commands NCOMMANDS, skipping."
        continue
    fi
    python binscript.py --commands-file COMMANDS_FILE $index
done
