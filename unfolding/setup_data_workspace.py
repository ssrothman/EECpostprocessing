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

def load_raw_reco(cfg, wtsyst):
    fs, skimpath = lookup_skim_path(
        cfg['location'], cfg['config_suite'], cfg['runtag'],
        cfg['dataset'], cfg['objsyst'], 'proj_Reco'
    )
    with fs.open(skimpath + '_BINNED_%s.npy' % wtsyst, 'rb') as f:
        values = np.load(f)
    with fs.open(skimpath + '_BINNED_covmat_%s.npy' % wtsyst, 'rb') as f:
        covmat = np.load(f)
    length = len(covmat)
    covmat = np.delete(covmat, range(length - 50, length), 0)
    covmat = np.delete(covmat, range(length - 50, length), 1)
    covmat = np.delete(covmat, range(0, 50), 0)
    covmat = np.delete(covmat, range(0, 50), 1)
    values = values[50:-50]
    return values, covmat, fs, skimpath

valid = np.load(os.path.join(PYTHIA_WORKSPACE, 'valid_bins.npy'))

print("Loading data reco histograms...")
combined_values = None
combined_cov    = None
binning_fs      = None
binning_path    = None
for era in DATA_ERAS:
    print("  Loading", era['dataset'], "...")
    vals, cov, fs, skimpath = load_raw_reco(era, WTSYST)
    if combined_values is None:
        combined_values = vals.copy()
        combined_cov    = cov.copy()
        binning_fs      = fs
        binning_path    = skimpath
    else:
        combined_values += vals
        combined_cov    += cov

combined_values = combined_values[valid]
combined_cov    = combined_cov[np.ix_(valid, valid)]

print("Inverting covariance matrix...")
invcov = smart_inverse(combined_cov, False)

binning = ArbitraryBinning()
with binning_fs.open(binning_path + '_bincfg.json', 'r') as f:
    binning.from_dict(json.load(f))

reco = Histogram(combined_values, combined_cov, invcov, binning)

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
