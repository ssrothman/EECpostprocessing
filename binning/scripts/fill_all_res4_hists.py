import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("Runtag", type=str)
parser.add_argument("Binner", type=str)
#which samples to run over
parser.add_argument("--samples", type=str, nargs='+',
                    default=[#"Pythia_inclusive",
                             "Pythia_HT-0to70",
                             "Pythia_HT-70to100",
                             "Pythia_HT-100to200",
                             "Pythia_HT-200to400",
                             "Pythia_HT-400to600",
                             "Pythia_HT-600to800",
                             "Pythia_HT-800to1200",
                             "Pythia_HT-1200to2500",
                             "Pythia_HT-2500toInf"])
                             #"Herwig_inclusive"])
#which hists to make
parser.add_argument("--hists", type=str, nargs='+',
                    default=['reco', 'unmatchedReco', 'untransferedReco', 
                             'gen', 'unmatchedGen', 'untransferedGen',
                             'transfer'])
#systematics
parser.add_argument('--skipNominal', action='store_true')
parser.add_argument('--objsysts', type=str, nargs='*', 
                    default=['JES_UP', 'JES_DN', 
                             'JER_UP', 'JER_DN', 
                             'UNCLUSTERED_UP', 'UNCLUSTERED_DN',
                             'TRK_EFF',
                             'CH_UP', 'CH_DN'])
parser.add_argument('--wtsysts', type=str, nargs='*', 
                    default=['idsfUp', 'idsfDown', 
                             'aSUp', 'aSDown',
                             'isosfUp', 'isosfDown', 
                             'triggersfUp', 'triggersfDown',
                             'prefireUp', 'prefireDown',
                             'PDFaSUp', 'PDFaSDown',
                             'scaleUp', 'scaleDown',
                             'PUUp', 'PUDown', 
                             'PDFUp', 'PDFDown',
                             'ISRUp', 'ISRDown',
                             'FSRUp', 'FSRDown'])

#bootstrapping
parser.add_argument('--boot_per_file', type=int, default=25)
parser.add_argument('--total_boot', type=int, default=4500)
parser.add_argument('--start_boot', type=int, default=0)

#statsplit
parser.add_argument('--statN', type=int, nargs='+', default=[2, 2, -1])
parser.add_argument('--statK', type=int, nargs='+', default=[0, 1, -1])

#kinreweight
parser.add_argument('--kinreweight_path', type=str, default='kinSF/Zkin.json')
parser.add_argument('--kinreweight_key', type=str, default='auto')

#process mode
process_group = parser.add_mutually_exclusive_group(required=False)
process_group.add_argument("--slurm", action='store_true')
process_group.add_argument("--condor", action='store_true')

#genuine options
parser.add_argument('--prebinned', action='store_true')
parser.add_argument('--r123type', type=str, default='philox', 
                    choices=['philox', 'threefry'],
                    help="Type of random number generator to use")
parser.add_argument('--collect_debug_info', action='store_true')
parser.add_argument('--force', action='store_true')

parser.add_argument('-j', '--jobs', type=int, default=1)

args = parser.parse_args()

from tqdm import tqdm
import subprocess
import os
from multiprocessing import Pool

def make_command(sample, hist, objsyst, wtsyst, statN, statK, boot, rng):
    if args.kinreweight_key == 'auto':
        if 'Pythia' in sample:
            kinkey = 'Pythia_Zkinweight'
        elif 'Herwig' in sample:
            kinkey = 'Herwig_Zkinweight'
        else:
            raise ValueError(f"Unknown sample type for kinreweight key: {sample}")
    else:
        kinkey = args.kinreweight_key

    command = [
        'python', 'scripts/fill_res4_hist.py',
        args.Runtag, sample, args.Binner, hist, objsyst, wtsyst,
        '--nboot', str(boot),
        '--rng', str(rng),
        '--r123type', args.r123type,
        '--statN', str(statN),
        '--statK', str(statK),
        '--kinreweight_path', args.kinreweight_path,
        '--kinreweight_key', kinkey,
    ]
    if boot > 0:
        command.append('--skipNominal')
    if args.prebinned:
        command.append('--prebinned')
    if args.collect_debug_info:
        command.append('--collect_debug_info')
    if args.force:
        command.append('--force')
    if args.slurm:
        command.append('--slurm')
    if args.condor:
        command.append('--condor')
    if not (args.condor or args.slurm):
        command.append('--mute')

    return command

def run_command(command):
    subprocess.run(command, check=True)

if __name__ == "__main__":
    commands = []
    for sample in args.samples:
        for hist in args.hists:
            for statN, statK in zip(args.statN, args.statK):
                if not args.skipNominal:
                    commands.append(make_command(sample, hist,
                                                 'nominal', 'nominal', 
                                                 statN, statK, 
                                                 0, 0))
                    if hist != 'transfer':
                        for rng in range(args.total_boot // args.boot_per_file):
                            commands.append(make_command(sample, hist,
                                                         'nominal', 'nominal',
                                                         statN, statK,
                                                         args.boot_per_file, rng+args.start_boot))
                for objsyst in args.objsysts: 
                    commands.append(make_command(sample, hist,
                                                 objsyst, 'nominal', 
                                                 statN, statK, 
                                                 0, 0))
                for wtsyst in args.wtsysts:
                    commands.append(make_command(sample, hist,
                                                 'nominal', wtsyst, 
                                                 statN, statK, 
                                                 0, 0))

    np.random.shuffle(commands)
    with Pool(args.jobs) as pool:
        results = list(tqdm(pool.imap(run_command, commands), 
                                  total=len(commands), 
                                  desc="Filling histograms"))
