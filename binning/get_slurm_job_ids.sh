#!/bin/bash

grep "SLURM JOB ID" slurm/*.out | awk '{print $5}' > slurm_job_ids.txt
