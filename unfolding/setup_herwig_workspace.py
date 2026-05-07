import fasteigenpy  # must import before torch

import numpy as np
import os
import json
import shutil

from general.fslookup.skim_path import lookup_skim_path
from unfolding.histogram import Histogram
from simonpy.stats_v2 import smart_inverse
from simonpy.AbitraryBinning import ArbitraryBinning

PYTHIA_WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace'
WORKSPACE        = '/eos/user/d/dponman/proj_unfold_workspace_herwig'
OBJSYST          = 'NOM'
WTSYST           = 'nominal'

herwig = {
    'location'     : 'dylan-lxplus-eos',
    'config_suite' : 'EvtMCprojConfig',
    'runtag'       : 'herwig_v3',
    'dataset'      : 'DYJetsToLL_Herwig',
    'objsyst'      : OBJSYST,
}

valid = np.load(os.path.join(PYTHIA_WORKSPACE, 'valid_bins.npy'))

print("Loading Herwig reco...")
fs, skimpath = lookup_skim_path(
    herwig['location'], herwig['config_suite'], herwig['runtag'],
    herwig['dataset'], herwig['objsyst'], 'proj_Reco'
)
with fs.open(skimpath + '_BINNED_%s.npy' % WTSYST, 'rb') as f:
    values = np.load(f)
with fs.open(skimpath + '_BINNED_covmat_%s.npy' % WTSYST, 'rb') as f:
    covmat = np.load(f)

length = len(covmat)
covmat = np.delete(covmat, range(length - 50, length), 0)
covmat = np.delete(covmat, range(length - 50, length), 1)
covmat = np.delete(covmat, range(0, 50), 0)
covmat = np.delete(covmat, range(0, 50), 1)
values = values[50:-50]

values = values[valid]
covmat = covmat[np.ix_(valid, valid)]

print("Inverting covariance matrix...")
invcov = smart_inverse(covmat, False)

binning = ArbitraryBinning()
with fs.open(skimpath + '_bincfg.json', 'r') as f:
    binning.from_dict(json.load(f))

reco = Histogram(values, covmat, invcov, binning)

os.makedirs(WORKSPACE, exist_ok=True)
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))

gen = Histogram.from_disk(os.path.join(PYTHIA_WORKSPACE, 'gen'))
gen.dump_to_disk(os.path.join(WORKSPACE, 'gen'))

shutil.copytree(
    os.path.join(PYTHIA_WORKSPACE, 'detectormodel'),
    os.path.join(WORKSPACE, 'detectormodel'),
    dirs_exist_ok=True
)
np.save(os.path.join(WORKSPACE, 'valid_bins.npy'), valid)

print("Workspace written to:", WORKSPACE)
