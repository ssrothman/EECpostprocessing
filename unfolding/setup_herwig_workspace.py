import fasteigenpy  # must import before torch

import numpy as np
import os
import json
import shutil

from unfolding.histogram import Histogram
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
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

def clean_covmat(cov):
    bad = np.isnan(np.diag(cov)) | (np.diag(cov) == 0)
    cov = np.where(np.isnan(cov), 0.0, cov)
    np.fill_diagonal(cov, np.where(bad, 1e30, np.diag(cov)))
    return cov

print("Loading Herwig reco...")
fs, valpath = get_hist_path(herwig['location'], herwig['config_suite'], herwig['runtag'],
                            herwig['dataset'], herwig['objsyst'], WTSYST,
                            'proj_totalReco', False, -1, -1)
fs, covpath = get_hist_path(herwig['location'], herwig['config_suite'], herwig['runtag'],
                            herwig['dataset'], herwig['objsyst'], WTSYST,
                            'proj_totalReco', True, -1, -1)
with fs.open(valpath, 'rb') as f:
    values = np.load(f)
with fs.open(covpath, 'rb') as f:
    covmat = np.load(f)

values = np.nan_to_num(values, nan=0.0)
covmat = clean_covmat(covmat)
print("Valid bins:", np.sum(np.diag(covmat) < 1e29), "of", len(values))

_, bincfgpath = get_hist_bincfg_path(herwig['location'], herwig['config_suite'],
                                      herwig['runtag'], herwig['dataset'],
                                      herwig['objsyst'], 'proj_totalReco')
binning = ArbitraryBinning()
with fs.open(bincfgpath, 'r') as f:
    binning.from_dict(json.load(f))

print("Inverting covariance matrix...")
reco = Histogram(values, covmat, binning)
reco.compute_invcov()

os.makedirs(WORKSPACE, exist_ok=True)
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))

mcgen = Histogram.from_disk(os.path.join(PYTHIA_WORKSPACE, 'mcgen'))
mcgen.dump_to_disk(os.path.join(WORKSPACE, 'mcgen'))

shutil.copytree(os.path.join(PYTHIA_WORKSPACE, 'model'),
                os.path.join(WORKSPACE, 'model'),
                dirs_exist_ok=True)

print("Workspace written to:", WORKSPACE)
