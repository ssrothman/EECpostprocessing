from processing.MatchingProcessor import MatchingProcessor
from processing.TransferProcessor import TransferProcessor
from processing.EECProcessor import EECProcessor
from processing.scaleout import setup_cluster_on_submit, custom_scale, setup_htcondor

from reading.files import get_rootfiles

from RecursiveNamespace import RecursiveNamespace

from coffea.nanoevents import NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor

import os
import argparse
import json

################### ARGUMENT PARSING ###################

parser = argparse.ArgumentParser(description='Produce histograms off of NanoAOD files')

processor_parsers = parser.add_subparsers(help='coffea processor to use', dest='processor', required=True)
eec_parser = processor_parsers.add_parser('EEC', help='EEC processor')
eec_parser.add_argument('--binwt', action='store_true')
eec_parser.add_argument('--noeff', action='store_false', dest='ineff')
eec_parser.add_argument('--statsplit', action='store_true')
matching_parser = processor_parsers.add_parser('matching', help='matching processor')
matching_parser.add_argument('matchings', nargs='+', default=['DefaultMatchParticles', 'NaiveMatchParticles'])
transfer_parser = processor_parsers.add_parser('transfer', help='transfer processor')

input_group = parser.add_mutually_exclusive_group(required=False)
input_group.add_argument('--LPC', dest='input', action='store_const', const='LPC')
input_group.add_argument('--MIT', dest='input', action='store_const', const='MIT')
input_group.add_argument('--local', dest='input', action='store_const', const='local')
parser.set_defaults(input='LPC')

parser.add_argument('--tag', dest='tag', type=str, help='production tag', required=True)
parser.add_argument('--nfiles', dest='nfiles', type=int, help='number of files to process', default=None, required=False)

scale_group = parser.add_mutually_exclusive_group(required=False)
scale_group.add_argument('--force-local', dest='force_local', action='store_true', help='force local execution')
scale_group.add_argument('--local-futures', dest='local_futures', action='store_true', help='force local execution with futures')
scale_group.add_argument('--custom-scale', dest='custom_scale', action='store_true')
scale_group.add_argument('--force-slurm', dest='force_slurm', action='store_true', help='force execution via slurm')

args = parser.parse_args()

######################################################################


################### INPUT ###################

if args.input == 'local':
    files = [args.tag]
else:
    if args.input == 'LPC':
        hostid = "cmseos.fnal.gov"
        rootpath = '/store/group/lpcpfnano/srothman/%s'%args.tag
    elif args.input == 'MIT':
        hostid = 'submit50.mit.edu'
        rootpath = '/store/user/srothman/%s'%args.tag

    files = get_rootfiles(hostid, rootpath)
    if args.nfiles is not None:
        files = files[:args.nfiles]
print("Processing %d files"%len(files))
print(files[0])

##############################################

################### PROCESSOR ###################
if args.processor == 'matching':
    processor_instance = MatchingProcessor(args.matchings)
elif args.processor == 'transfer':
    processor_instance = TransferProcessor()
elif args.processor == 'EEC':
    with open("config.json", 'r') as f:
        config = RecursiveNamespace(**json.load(f))
    processor_instance = EECProcessor(config, args.statsplit)
else:
    raise ValueError("Unknown processor %s"%args.processor)

##################################################

################### OUTPUT ###################
if args.input == 'local':
    destination = 'testlocal'
else:
    destination = "/data/submit/srothman/EEC/%s/%s"%(args.tag, args.processor)
    destination = "./%s/%s"%(args.tag, args.processor)
    if os.path.exists(destination):
        raise ValueError("Destination %s already exists"%destination)

os.makedirs(destination, exist_ok=True)
##################################################

################### EXECUTION ###################

use_slurm = len(files) > 10
if args.force_local or args.local_futures or args.custom_scale:
    use_slurm = False
if args.force_slurm:
    use_slurm = True

if use_slurm:
    print("using slurm")
    cluster, client = setup_cluster_on_submit(1, 100, destination)
    #cluster, client = setup_htcondor(1, 10, destination)

    runner = Runner(
        executor=DaskExecutor(client=client, status=True),
        #chunksize=100000,
        schema=NanoAODSchema
    )
elif not args.custom_scale:
    print("running locally")
    runner = Runner(
        executor=FuturesExecutor(workers=4) if args.local_futures else IterativeExecutor(),
        #executor=FuturesExecutor(workers=10, status=True),
        chunksize=1000,
        schema=NanoAODSchema
    )
else:
    print("doing custom scale")
    custom_scale(files, processor_instance, destination)
    runner = None
##################################################


################### RUNNING ###################

if runner is not None:
    out = runner(
        {"DYJetsToLL" : files},
        treename='Events',
        processor_instance=processor_instance
    )

    with open(os.path.join(destination,"hists.pkl"), 'wb') as fout:
        import pickle
        pickle.dump(out, fout)

    if use_slurm:
        client.close()
        cluster.close()

##################################################
