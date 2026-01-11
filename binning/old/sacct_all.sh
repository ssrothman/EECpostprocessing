#!/bin/bash

./get_slurm_job_ids.sh
sacct -o "JobID%16, MaxRSS, TotalCPU, Elapsed, State, ExitCode" --units G -j $(cat slurm_job_ids.txt | awk '{print $1".batch"}' | paste -s -d,)
