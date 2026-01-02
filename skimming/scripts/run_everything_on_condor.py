#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Run EVERYTHING.")
parser.add_argument('--signal-mc', type=str, nargs='+', help="List of signal MC datasets to process",
                    default=[
                        'Pythia_inclusive', 
                        'Herwig_inclusive',
                        'Pythia_HT-0to70',
                        'Pythia_HT-70to100',
                        'Pythia_HT-100to200',
                        'Pythia_HT-200to400',
                        'Pythia_HT-400to600',
                        'Pythia_HT-600to800',
                        'Pythia_HT-800to1200',
                        'Pythia_HT-1200to2500',
                        'Pythia_HT-2500toInf',
                    ])
parser.add_argument('--background-mc', type=str, nargs='+', help="List of background MC datasets to process",
                    default=[
                        'WW',
                         'WZ',
                         'ZZ',
                         'TT',
                         'ST_t',
                         'ST_t_anti',
                         'ST_tW',
                         'ST_tW_anti'
                    ])
parser.add_argument('--data', type=str, nargs='+', help="List of data datasets to process",
                    default=[
                        'DATA_2018A', 
                        'DATA_2018B', 
                        'DATA_2018C', 
                        'DATA_2018D'
                    ])
parser.add_argument('--config-suite', type=str, help="name of config suite to use",
                     default="basic")
parser.add_argument('--objsysts', type=str, nargs='+', help="List of object syst variations to process",
                    default=[
                        'nominal',
                        'JES_UP',
                        'JES_DN',
                        'JER_UP',
                        'JER_DN',
                        'UNCLUSTERED_UP', 
                        'UNCLUSTERED_DN', 
                        'CH_UP', 
                        'CH_DN',
                        'TRK_EFF'
                    ])
parser.add_argument('--tables', type=str, nargs='+', help="List of table variations to process",
                    default=['all', 'count'])
parser.add_argument('--files_per_job', type=int, help="Number of input files per job",
                     default=10)
parser.add_argument('--runtag', type=str, help="Datasets runtag",
                    default='Apr_23_2025')
args = parser.parse_args()

import os
import subprocess

def setup_and_stage(dset, objsyst, table):
    cmd = 'setup_skimming_workspace.py skim_%s_%s_%s %s %s %s --tables %s --output-location xrootd-submit --config-suite %s' % (
        dset,
        objsyst,
        table,
        args.runtag,
        dset,
        objsyst,
        table,
        args.config_suite
    )
    output = subprocess.run(cmd, shell=True, capture_output=True)
    print(output.stdout.decode())
    if output.returncode != 0:
        print(output.stderr.decode())
        raise RuntimeError("Workspace setup failed")

    if not os.path.exists('skim_%s_%s_%s'%(dset, objsyst, table)):
        return # if workspace setup didn't do anything,
                    # it's because all the desired outputs already exist! 
                    # so we can skip staging too.
                    
    cmd = 'stage_to_condor.py skim_%s_%s_%s/ %s_%s_%s --files-per-job %d --exec' % (
        dset,
        objsyst,
        table,
        dset,
        objsyst,
        table,
        args.files_per_job
    )
    output = subprocess.run(cmd, shell=True, capture_output=True)
    print(output.stdout.decode())
    if output.returncode != 0:
        print(output.stderr.decode())
        raise RuntimeError("Staging to condor failed")

for smc in args.signal_mc:
    for table in args.tables:
        for objsyst in args.objsysts:
            if table == 'count' and objsyst != 'nominal':
                continue
            setup_and_stage(smc, objsyst, table)

for bmc in args.background_mc:
    objsyst = 'nominal'
    for table in args.tables:
        setup_and_stage(bmc, objsyst, table)

for data in args.data:
    objsyst = 'DATA'
    for table in args.tables:
        setup_and_stage(data, objsyst, table)