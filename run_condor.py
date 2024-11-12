import argparse

parser = argparse.ArgumentParser(description='Produce histograms off of NanoAOD files')

parser.add_argument("sample", type=str)
parser.add_argument("what", type=str)
parser.add_argument('jettype', type=str)
parser.add_argument('EECtype', type=str)
parser.add_argument('workspace', type=str)


parser.add_argument('--samplelist', type=str, default='latest', required=False)

parser.add_argument('--treatAsData', action='store_true')
parser.add_argument('--manualcov', action='store_true')
parser.add_argument('--poissonbootstrap', type=int, default=0, required=False)
parser.add_argument('--statsplit', type=int, default=1, required=False)

parser.add_argument('--filebatch', type=int, default=1, required=False)

syst_group = parser.add_mutually_exclusive_group(required=False)
syst_group.add_argument('--scanSyst', action='store_const', 
                        const='scanAll', dest='syst')
syst_group.add_argument('--noSyst', action='store_const',
                        const='none', dest='syst')
syst_group.add_argument('--scanJetMETSyst', action='store_const',
                       const='scanJetMET', dest='syst')
syst_group.add_argument('--scanMuonSyst', action='store_const',
                        const='scanMuon', dest='syst')
syst_group.add_argument('--scanTriggerSyst', action='store_const',
                        const='scanTrigger', dest='syst')
syst_group.add_argument('--scanTheorySyst', action='store_const',
                        const='scanTheory', dest='syst')
syst_group.add_argument('--scanPSSyst', action='store_const',
                        const='scanPS', dest='syst')
syst_group.add_argument('--scanBtagEffSyst', action='store_const',
                        const='scanBtagEff', dest='syst')
syst_group.add_argument('--scanBtagSFSyst', action='store_const',
                        const='scanBtagSF', dest='syst')
syst_group.add_argument('--scanPileupSyst', action='store_const',
                        const='scanPileup', dest='syst')
syst_group.add_argument('--scanCBxsec', action='store_const',
                        const='scanCBxsec', dest='syst')
syst_group.add_argument('--scanLxsec', action='store_const',
                        const='scanLxsec', dest='syst')
parser.set_defaults(syst='noSyst')
parser.add_argument('--skipNominal', action='store_true')

parser.add_argument('--bTag', type=str, default='tight', required=False, choices=['tight', 'medium', 'loose'])

parser.add_argument('--noBkgVeto', action='store_true')
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

parser.add_argument('--nfiles', dest='nfiles', type=int, 
                    default=None, required=False)
parser.add_argument('--startfile', type=int, default=0, required=False)

args = parser.parse_args()

import os.path
outpath = 'root://submit50.mit.edu//store/user/srothman/condor/%s/%s'%("fullrun6",os.path.split(args.workspace)[-1])

import samples
SAMPLE_LIST = samples.samplelists[args.samplelist].SAMPLE_LIST

################### INPUT ###################
sample = SAMPLE_LIST.lookup(args.sample)
files = sample.get_files()
if args.nfiles is not None:
    files = files[args.startfile:args.nfiles+args.startfile]

import random
random.shuffle(files)

print("Processing %d files"%len(files))
print("files[0] :", files[0])
##############################################

################### CONFIG ###################
from RecursiveNamespace import RecursiveNamespace
import json

with open("configs/base.json", 'r') as f:
    config = RecursiveNamespace(**json.load(f))

with open("configs/%s.json"%args.jettype, 'r') as f:
    config.update(json.load(f))

with open("configs/%sEEC.json"%args.EECtype, 'r') as f:
    config.update(json.load(f))

config.tagging.wp = args.bTag

argsdict = {
    'config' : config.to_dict(),
    'statsplit' : args.statsplit,
    'what' : args.what,
    'scanSyst' : args.syst,
    'era' : sample.JEC,
    'flags' : sample.flags,
    'noRoccoR' : args.noRoccoR,
    'noJER' : args.noJER,
    'noJEC' : args.noJEC,
    'noPUweight' : args.noPUweight,
    'noPrefireSF' : args.noPrefireSF,
    'noIDsfs' : args.noIDsfs,
    'noIsosfs' : args.noIsosfs,
    'noTriggersfs' : args.noTriggersfs,
    'noBtagSF' : args.noBtagSF,
    'Zreweight' : args.Zreweight,
    'treatAsData' : args.treatAsData,
    'manualcov' : args.manualcov,
    'poissonbootstrap' : args.poissonbootstrap,
    'noBkgVeto' : args.noBkgVeto,
    'skipNominal' : args.skipNominal,
}
##############################################

################### SETUP ####################
import os
os.makedirs(args.workspace, exist_ok=False)
os.makedirs(os.path.join(args.workspace, 'output'))

configpath = os.path.join(args.workspace, 'config.json')
with open(configpath, 'w') as f:
    json.dump(config.to_dict(), f, indent=4)

argspath = os.path.join(args.workspace, 'args.json')
with open(argspath, 'w') as f:
    json.dump(argsdict, f, indent=4)

filelist = {}
for i, file in enumerate(files):
    ibatch = i//args.filebatch
    if ibatch in filelist:
        filelist[ibatch].append(file)
    else:
        filelist[ibatch] = [file]

filespath = os.path.join(args.workspace, 'filelist.json')
with open(filespath, 'w') as f:
    json.dump(filelist, f, indent=4)

import shutil
execpypath = os.path.join(args.workspace, 'CONDOR_EXEC.py')
shutil.copyfile('condor/CONDOR_EXEC.py', execpypath)

execshpath = os.path.join(args.workspace, 'CONDOR_EXEC.sh')
shutil.copyfile('condor/CONDOR_EXEC.sh', execshpath)

#subpath = os.path.join(args.workspace, 'condor.sub')
#shutil.copyfile('condor/condor.sub', subpath)

tgzpypath = os.path.join(args.workspace, 'postprocessing.tgz')
shutil.copyfile('condor/postprocessing.tgz', tgzpypath)
##############################################

with open(os.path.join(args.workspace, 'condor.sub'), 'w') as f:
    f.write("""
executable                = CONDOR_EXEC.sh
arguments                 = $(ProcId) %s/hists_$(ProcId).pkl
request_memory            = 6GB
should_transfer_files     = YES
output                    = output/$(Cluster).$(ProcId).out
error                     = output/$(Cluster).$(ProcId).err
log                       = output/$(Cluster).$(ProcId).log
on_exit_remove            = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold              = (ExitBySignal == True) || (ExitCode != 0)
max_retries               = 3
requirements              = Machine =!= LastRemoteHost
use_x509userproxy         = True
max_idle                  = 100
x509userproxy             = /home/submit/srothman/myticket
+AccountingGroup          = "analysis.srothman"
+JobFlavour               = "espresso"
+SingularityImage         = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:0.7.22-py3.10-gf48fa"
Transfer_Input_Files      = CONDOR_EXEC.sh, CONDOR_EXEC.py, filelist.json, args.json, postprocessing.tgz
Transfer_Output_Files     = ""
Requirements              = ( BOSCOCluster =!= "t3serv008.mit.edu" && BOSCOCluster =!= "ce03.cmsaf.mit.edu" && BOSCOCluster =!= "eofe8.mit.edu")
+DESIRED_Sites            = "T2_AT_Vienna,T2_BE_IIHE,T2_BE_UCL,T2_BR_SPRACE,T2_BR_UERJ,T2_CH_CERN,T2_CH_CERN_AI,T2_CH_CERN_HLT,T2_CH_CERN_Wigner,T2_CH_CSCS,T2_CH_CSCS_HPC,T2_CN_Beijing,T2_DE_DESY,T2_DE_RWTH,T2_EE_Estonia,T2_ES_CIEMAT,T2_ES_IFCA,T2_FI_HIP,T2_FR_CCIN2P3,T2_FR_GRIF_IRFU,T2_FR_GRIF_LLR,T2_FR_IPHC,T2_GR_Ioannina,T2_HU_Budapest,T2_IN_TIFR,T2_IT_Bari,T2_IT_Legnaro,T2_IT_Pisa,T2_IT_Rome,T2_KR_KISTI,T2_MY_SIFIR,T2_MY_UPM_BIRUNI,T2_PK_NCP,T2_PL_Swierk,T2_PL_Warsaw,T2_PT_NCG_Lisbon,T2_RU_IHEP,T2_RU_INR,T2_RU_ITEP,T2_RU_JINR,T2_RU_PNPI,T2_RU_SINP,T2_TH_CUNSTDA,T2_TR_METU,T2_TW_NCHC,T2_UA_KIPT,T2_UK_London_IC,T2_UK_SGrid_Bristol,T2_UK_SGrid_RALPP,T2_US_Caltech,T2_US_Florida,T2_US_Nebraska,T2_US_Purdue,T2_US_UCSD,T2_US_Vanderbilt,T2_US_Wisconsin,T3_CH_CERN_CAF,T3_CH_CERN_DOMA,T3_CH_CERN_HelixNebula,T3_CH_CERN_HelixNebula_REHA,T3_CH_CMSAtHome,T3_CH_Volunteer,T3_US_HEPCloud,T3_US_NERSC,T3_US_OSG,T3_US_PSC,T3_US_SDSC,T3_US_MIT"

queue %d
"""%(outpath, len(filelist.keys())))
