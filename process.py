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

parser.add_argument('--statsplit', action='store_true')

parser.add_argument("what", type=str)
parser.add_argument('jettype', type=str)
parser.add_argument('EECtype', type=str)
parser.add_argument("era", type=str)

syst_group = parser.add_mutually_exclusive_group(required=False)
syst_group.add_argument('--nom', dest='syst', action='store_const',
                        const='nom')

syst_group.add_argument('--JER', dest='syst', action='store_const', 
                        const='JER')
syst_group.add_argument('--JES', dest='syst', action='store_const',
                        const='JES')

syst_group.add_argument("--wt_prefire", dest='syst', action='store_const',
                        const='wt_prefire')
syst_group.add_argument("--wt_idsf", dest='syst', action='store_const',
                        const='wt_idsf')
syst_group.add_argument("--wt_isosf", dest='syst', action='store_const',
                        const='wt_isosf')
syst_group.add_argument("--wt_triggersf", dest='syst', action='store_const',
                        const='wt_triggersf')
syst_group.add_argument("--wt_scale", dest='syst', action='store_const',
                        const='wt_scale')
syst_group.add_argument("--wt_ISR", dest='syst', action='store_const',
                        const='wt_ISR')
syst_group.add_argument("--wt_FSR", dest='syst', action='store_const',
                        const='wt_FSR')
syst_group.add_argument("--wt_PDF", dest='syst', action='store_const',
                        const='wt_PDF')
syst_group.add_argument("--wt_aS", dest='syst', action='store_const',
                        const='wt_aS')
syst_group.add_argument("--wt_PDFaS", dest='syst', action='store_const',
                        const='wt_PDFaS')
parser.set_defaults(syst='nom')

syst_updn_group = parser.add_mutually_exclusive_group(required=False)
syst_updn_group.add_argument('--DN', dest='syst_updn', 
                             action='store_const', const='DN')
syst_updn_group.add_argument('--UP', dest='syst_updn',
                             action='store_const', const='UP')
parser.set_defaults(syst_updn=None)

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--LPC', dest='input', action='store_const', const='LPC')
input_group.add_argument('--MIT', dest='input', action='store_const', const='MIT')
input_group.add_argument('--local', dest='input', action='store_const', const='local')
parser.set_defaults(input='LPC')

parser.add_argument('--tag', dest='tag', type=str, help='production tag', required=True)
parser.add_argument('--nfiles', dest='nfiles', type=int, help='number of files to process', default=None, required=False)
parser.add_argument('--startfile', type=int, default=0, required=False)

scale_group = parser.add_mutually_exclusive_group(required=False)
scale_group.add_argument('--force-local', dest='force_local', action='store_true', help='force local execution')
scale_group.add_argument('--local-futures', dest='local_futures', action='store_true', help='force local execution with futures')
scale_group.add_argument('--custom-scale', dest='custom_scale', action='store_true')
scale_group.add_argument('--force-slurm', dest='force_slurm', action='store_true', help='force execution via slurm')

args = parser.parse_args()

if args.syst != 'nom' and args.syst_updn is None:
    raise ValueError("Must specify UP or DN for systematic")

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
        files = files[args.startfile:args.nfiles+args.startfile]
print("Processing %d files"%len(files))
print(files[0])

##############################################

################### PROCESSOR ###################
with open("configs/base.json", 'r') as f:
    config = RecursiveNamespace(**json.load(f))

with open("configs/%s.json"%args.jettype, 'r') as f:
    config.update(json.load(f))

with open("configs/%sEEC.json"%args.EECtype, 'r') as f:
    config.update(json.load(f))

processor_instance = EECProcessor(config, statsplit=args.statsplit,
                                  what=args.what, syst=args.syst,
                                  syst_updn=args.syst_updn,
                                  era = args.era)

##################################################

################### OUTPUT ###################
out_fname = 'hists'
if args.syst != 'nom':
    out_fname += '_%s_%s'%(args.syst, args.syst_updn)
out_fname += '_file%dto%d'%(args.startfile, args.startfile+len(files))
if args.statsplit:
    out_fname += '_statsplit'
out_fname += '.pkl'

if args.input == 'local':
    destination = 'testlocal'
else:
    destination = "/data/submit/srothman/EEC/%s/%s"%(args.tag, args.what)
    #destination = "./%s/%s"%(args.tag, args.what)
    if os.path.exists(os.path.join(destination, out_fname)):
        raise ValueError("Destination %s already exists"%os.path.join(destination, out_fname))

print("Outputting to %s"%os.path.join(destination, out_fname))

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
        #chunksize=1000,
        schema=NanoAODSchema,
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

    with open(os.path.join(destination,out_fname), 'wb') as fout:
        import pickle
        pickle.dump(out, fout)

    if use_slurm:
        client.close()
        cluster.close()

##################################################
