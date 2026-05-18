import fasteigenpy  # must import before torch

import numpy as np
import os
import json

from unfolding.specs import dsspec
from unfolding.histogram import Histogram
from unfolding.detectormodel import DetectorModel, load_hist_from_dataset
from general.fslookup.skim_path import lookup_skim_path
from simonpy.AbitraryBinning import ArbitraryBinning
from simonpy.stats_v2 import smart_inverse

HT_BINS = [
    ('DYJetsToLL_Pythia_HT70to100',    159.1),
    ('DYJetsToLL_Pythia_HT100to200',   159.4),
    ('DYJetsToLL_Pythia_HT200to400',   43.60),
    ('DYJetsToLL_Pythia_HT400to600',   5.918),
    ('DYJetsToLL_Pythia_HT600to800',   1.439),
    ('DYJetsToLL_Pythia_HT800to1200',  0.6462),
    ('DYJetsToLL_Pythia_HT1200to2500', 0.1514),
    ('DYJetsToLL_Pythia_HT2500toInf',  0.003395),
]

WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace'
OBJSYST   = 'NOM'
WTSYST    = 'nominal'
LUMI      = (14.02 + 7.06 + 31.83) * 1e3  # pb^-1, 2018 A+B+D

def make_dset(dataset):
    return {
        'location'     : 'dylan-lxplus-eos',
        'config_suite' : 'EvtMCprojConfig',
        'runtag'       : 'v8',
        'dataset'      : dataset,
        'isMC'         : True,
    }

def load_n_events(dataset):
    dset = make_dset(dataset)
    fs, path = lookup_skim_path(
        dset['location'], dset['config_suite'],
        dset['runtag'], dataset, OBJSYST, 'count'
    )
    with fs.open(os.path.join(path, 'merged.json'), 'r') as f:
        return json.load(f)['n_events']

def load_ht_array(what, strip_flow=True):
    """Sum a binned array over all HT bins, weighted by xsec/n_events * LUMI."""
    result = None
    for dataset, xsec in HT_BINS:
        dset = make_dset(dataset)
        arr = load_hist_from_dataset(dset, OBJSYST, '%s_BINNED_%s.npy' % (what, WTSYST))
        if strip_flow:
            arr = arr[50:-50]
        weight = xsec / load_n_events(dataset) * LUMI
        result = weight * arr if result is None else result + weight * arr
    return result

def load_ht_covmat(what):
    """Sum a binned covmat over all HT bins, weighted by (xsec/n_events * LUMI)^2."""
    result = None
    for dataset, xsec in HT_BINS:
        dset = make_dset(dataset)
        arr = load_hist_from_dataset(dset, OBJSYST, '%s_BINNED_covmat_%s.npy' % (what, WTSYST))
        length = len(arr)
        arr = np.delete(arr, range(length - 50, length), 0)
        arr = np.delete(arr, range(length - 50, length), 1)
        arr = np.delete(arr, range(0, 50), 0)
        arr = np.delete(arr, range(0, 50), 1)
        weight = xsec / load_n_events(dataset) * LUMI
        result = weight**2 * arr if result is None else result + weight**2 * arr
    return result

# --- valid mask from combined HT reco covariance + transfer ---
print("Computing valid mask from HT-stitched reco...")
raw_cov = load_ht_covmat('proj_Reco')
valid   = ~np.isnan(np.diag(raw_cov))
print("Valid bins (reco cov):", valid.sum(), "of", len(valid))

# extend valid mask: exclude bins where transfer has zero row sums
transfer_raw_full = None
for dataset, xsec in HT_BINS:
    dset = make_dset(dataset)
    arr  = load_hist_from_dataset(dset, OBJSYST, 'proj_transfer_BINNED_%s.npy' % WTSYST)
    w    = xsec / load_n_events(dataset) * LUMI
    transfer_raw_full = w * arr if transfer_raw_full is None else transfer_raw_full + w * arr

n_with_flow_check = int(np.sqrt(len(transfer_raw_full)))
t0_full = transfer_raw_full.reshape(n_with_flow_check, n_with_flow_check)
t0_full = t0_full[50:-50, 50:-50]          # 450x450

print("Building HT-stitched Pythia reco...")
reco_vals = load_ht_array('proj_Reco')[valid]
reco_cov  = raw_cov[np.ix_(valid, valid)]
reco_invcov = np.linalg.inv(reco_cov + 1e-10 * np.eye(len(reco_cov)))

fs0, skimpath0 = lookup_skim_path('dylan-lxplus-eos', 'EvtMCprojConfig', 'v8',
                                   HT_BINS[0][0], OBJSYST, 'proj_Reco')
binning_reco = ArbitraryBinning()
with fs0.open(skimpath0 + '_bincfg.json', 'r') as f:
    binning_reco.from_dict(json.load(f))

reco = Histogram(reco_vals, reco_cov, reco_invcov, binning_reco)
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))
print("HT-stitched Pythia reco sum:", reco_vals.sum())

# --- HT-stitched gen baseline ---
print("Building HT-stitched gen baseline...")
ht_gen_vals = load_ht_array('proj_Gen')[valid]

gen_cov    = np.diag(np.where(ht_gen_vals > 0, ht_gen_vals, 1.0))
gen_invcov = smart_inverse(gen_cov, False)

fs1, skimpath1 = lookup_skim_path('dylan-lxplus-eos', 'EvtMCprojConfig', 'v8',
                                   HT_BINS[0][0], OBJSYST, 'proj_Gen')
binning_gen = ArbitraryBinning()
with fs1.open(skimpath1 + '_bincfg.json', 'r') as f:
    binning_gen.from_dict(json.load(f))

gen = Histogram(ht_gen_vals, gen_cov, gen_invcov, binning_gen)
gen.dump_to_disk(os.path.join(WORKSPACE, 'gen'))
print("HT-stitched gen sum:", ht_gen_vals.sum())

print("Building HT-stitched detector model...")
umG      = load_ht_array('proj_unmatchedGen')[valid]
umR      = load_ht_array('proj_unmatchedReco')[valid]
totG     = load_ht_array('proj_Gen')[valid]
totR     = load_ht_array('proj_Reco')[valid]

nGen  = len(totG)
nReco = len(totR)

Gdenom = np.where(totG == 0, 1.0, totG)
gamma0 = umG / Gdenom

bkgR   = umR
Rdenom = totR - bkgR
Rdenom = np.where(Rdenom == 0, 1.0, Rdenom)
rho0   = bkgR / Rdenom

# reuse already-loaded transfer; apply valid mask
t0 = t0_full[np.ix_(valid, valid)]
matchedG = totG - umG
tdenom   = np.where(matchedG == 0, 1.0, matchedG)
t0      /= tdenom[np.newaxis, :]

model = DetectorModel(
    transfer0          = t0,
    gamma0             = gamma0,
    rho0               = rho0,
    transferVariations = np.zeros((0, nReco, nGen)),
    transferVarIndices = np.array([], dtype=int),
    gammaVariations    = np.zeros((0, nGen)),
    rhoVariations      = np.zeros((0, nReco)),
)
print(model)
model.dump_to_disk(os.path.join(WORKSPACE, 'detectormodel'))

np.save(os.path.join(WORKSPACE, 'valid_bins.npy'), valid)
print("Workspace written to:", WORKSPACE)
