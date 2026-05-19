import fasteigenpy  # must import before torch

import numpy as np
import os
import json
import shutil

from unfolding.histogram import Histogram
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
from simonpy.AbitraryBinning import ArbitraryBinning

PYTHIA_WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace'
WORKSPACE        = '/eos/user/d/dponman/proj_unfold_workspace_data'
WTSYST           = 'nominal'

DATA_ERAS = [
    {'location': 'dylan-lxplus-eos', 'config_suite': 'EvtDataprojConfig',
     'runtag': 'data_v1', 'dataset': 'DATA_2018A', 'objsyst': 'DATA'},
    {'location': 'dylan-lxplus-eos', 'config_suite': 'EvtDataprojConfig',
     'runtag': 'data_v1', 'dataset': 'DATA_2018B', 'objsyst': 'DATA'},
    {'location': 'dylan-lxplus-eos', 'config_suite': 'EvtDataprojConfig',
     'runtag': 'data_v1', 'dataset': 'DATA_2018D', 'objsyst': 'DATA'},
]

def load_raw_reco(cfg):
    fs, valpath = get_hist_path(cfg['location'], cfg['config_suite'], cfg['runtag'],
                                cfg['dataset'], cfg['objsyst'], WTSYST,
                                'proj_totalReco', False, -1, -1)
    fs, covpath = get_hist_path(cfg['location'], cfg['config_suite'], cfg['runtag'],
                                cfg['dataset'], cfg['objsyst'], WTSYST,
                                'proj_totalReco', True, -1, -1)
    with fs.open(valpath, 'rb') as f:
        values = np.load(f)
    with fs.open(covpath, 'rb') as f:
        covmat = np.load(f)
    n = len(covmat)
    covmat = covmat[50:n-50, 50:n-50]
    values = values[50:-50]
    return values, covmat, fs, cfg

valid_pythia = np.load(os.path.join(PYTHIA_WORKSPACE, 'valid_bins.npy'))

print("Loading data reco histograms...")
combined_values = None
combined_cov    = None
binning_cfg     = None
for era in DATA_ERAS:
    print("  Loading", era['dataset'], "...")
    vals, cov, fs, cfg = load_raw_reco(era)
    if combined_values is None:
        combined_values = vals.copy()
        combined_cov    = cov.copy()
        binning_cfg     = cfg
    else:
        combined_values += vals
        combined_cov    += cov

valid_data = ~np.isnan(np.diag(combined_cov))
valid      = valid_pythia & valid_data
print("Valid bins: Pythia", valid_pythia.sum(), "Data", valid_data.sum(), "Intersection", valid.sum())

combined_values = combined_values[valid]
combined_cov    = combined_cov[np.ix_(valid, valid)]

_, bincfgpath = get_hist_bincfg_path(binning_cfg['location'], binning_cfg['config_suite'],
                                      binning_cfg['runtag'], binning_cfg['dataset'],
                                      binning_cfg['objsyst'], 'proj_totalReco')
binning = ArbitraryBinning()
with fs.open(bincfgpath, 'r') as f:
    binning.from_dict(json.load(f))

print("Inverting covariance matrix...")
reco = Histogram(combined_values, combined_cov, binning)
reco.compute_invcov()

os.makedirs(WORKSPACE, exist_ok=True)
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))

mcgen = Histogram.from_disk(os.path.join(PYTHIA_WORKSPACE, 'mcgen'))
mcgen.dump_to_disk(os.path.join(WORKSPACE, 'mcgen'))

shutil.copytree(os.path.join(PYTHIA_WORKSPACE, 'model'),
                os.path.join(WORKSPACE, 'model'),
                dirs_exist_ok=True)
np.save(os.path.join(WORKSPACE, 'valid_bins.npy'), valid)

print("Workspace written to:", WORKSPACE)
