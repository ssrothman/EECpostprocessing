from processing.EECProcessor import EECProcessor
from processing.scaleout import setup_cluster_on_submit, custom_scale, setup_htcondor

from reading.files import get_rootfiles

from RecursiveNamespace import RecursiveNamespace

from coffea.nanoevents import NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor

import os
import argparse
import json

from samples.latest import SAMPLE_LIST

################### ARGUMENT PARSING ###################

parser = argparse.ArgumentParser(description='Produce histograms off of NanoAOD files')

parser.add_argument('--statsplit', action='store_true')

parser.add_argument("sample", type=str)
parser.add_argument("what", type=str)
parser.add_argument('jettype', type=str)
parser.add_argument('EECtype', type=str)

parser.add_argument('--treatAsData', action='store_true')

parser.add_argument('--extra-tags', type=str, default=None, required=False, nargs='*')

parser.add_argument('--bTag', type=str, default='tight', required=False, choices=['tight', 'medium', 'loose'])

parser.add_argument('--noRoccoR', action='store_true')
parser.add_argument('--noJER', action='store_true')
parser.add_argument('--noJEC', action='store_true')
parser.add_argument('--noPUweight', action='store_true')
parser.add_argument('--noPrefireSF', action='store_true')
parser.add_argument('--noIDsfs', action='store_true')
parser.add_argument('--noIsosfs', action='store_true')
parser.add_argument('--noTriggersfs', action='store_true')
parser.add_argument('--noBtagSF', action='store_true')

parser.add_argument('--Zreweight', action='store_true')

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
syst_group.add_argument('--wt_PU', dest='syst', action='store_const',
                        const='wt_PU')
syst_group.add_argument('--wt_btagSF', dest='syst', action='store_const',
                        const='wt_btagSF')
parser.set_defaults(syst='nom')

syst_updn_group = parser.add_mutually_exclusive_group(required=False)
syst_updn_group.add_argument('--DN', dest='syst_updn', 
                             action='store_const', const='DN')
syst_updn_group.add_argument('--UP', dest='syst_updn',
                             action='store_const', const='UP')
parser.set_defaults(syst_updn=None)

parser.add_argument("--local", action='store_true')

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


if args.local:
    files = [args.sample]
else:
    sample = SAMPLE_LIST.lookup(args.sample)
    files = sample.get_files()
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

config.tagging.wp = args.bTag

processor_instance = EECProcessor(
        config, statsplit=args.statsplit,
        what=args.what, syst=args.syst,
        syst_updn=args.syst_updn,
        era = '2018A' if args.local else sample.JEC,
        flags = None if args.local else sample.flags,
        noRoccoR = args.noRoccoR,
        noJER = args.noJER,
        noJEC = args.noJEC,
        noPUweight = args.noPUweight,
        noPrefireSF = args.noPrefireSF,
        noIDsfs = args.noIDsfs,
        noIsosfs = args.noIsosfs,
        noTriggersfs = args.noTriggersfs,
        noBtagSF = args.noBtagSF,
        Zreweight = args.Zreweight,
        treatAsData = args.treatAsData)

##################################################

################### OUTPUT ###################
out_fname = 'hists'
if args.syst != 'nom':
    if args.syst.startswith('wt_'):
        out_fname += '_%s%s'%(args.syst[3:], args.syst_updn)
    else:
        out_fname += '_%s%s'%(args.syst, args.syst_updn)

out_fname += '_file%dto%d'%(args.startfile, args.startfile+len(files))

if args.bTag == 'tight':
    out_fname += '_tight'
elif args.bTag == 'medium':
    out_fname += '_medium'
elif args.bTag == 'loose':
    out_fname += '_loose'

if args.statsplit:
    out_fname += '_statsplit'
if args.noRoccoR:
    out_fname += '_noRoccoR'
if args.noJER:
    out_fname += '_noJER'
if args.noJEC:
    out_fname += '_noJEC'
if args.noPUweight:
    out_fname += '_noPUweight'
if args.noPrefireSF:
    out_fname += '_noPrefireSF'
if args.noIDsfs:
    out_fname += '_noIDsfs'
if args.noIsosfs:
    out_fname += '_noIsosfs'
if args.noTriggersfs:
    out_fname += '_noTriggersfs'
if args.noBtagSF:
    out_fname += '_noBtagSF'
if args.Zreweight:
    out_fname += '_Zreweight'

if args.extra_tags is not None:
    for tag in args.extra_tags:
        out_fname += '_%s'%tag

if args.treatAsData:
    out_fname += '_asData'

out_fname += '.pkl'

if args.local:
    destination = 'testlocal'
else:
    destination = "/data/submit/srothman/EEC/%s/%s/%s"%(SAMPLE_LIST.tag, sample.name, args.what)
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
    cluster, client = setup_cluster_on_submit(1, 200, destination)
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
