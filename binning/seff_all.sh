#!/bin/bash

for i in $(cat slurm_job_ids.txt); do
    echo "Job ID: $i"
    seff $i
    echo "----------------------------------------"
done
