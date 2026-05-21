import fasteigenpy  # must import before torch

import numpy as np
import os
import json

from unfolding.histogram import Histogram
from unfolding.detectormodel import DetectorModel
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
from general.datasets.datasets import lookup_count
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning

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

WORKSPACE    = '/eos/user/d/dponman/proj_unfold_workspace'
RUNTAG       = 'v8'
CONFIG_SUITE = 'EvtMCprojConfig'
LOCATION     = 'dylan-lxplus-eos'
OBJSYST      = 'NOM'
WTSYST       = 'nominal'
LUMI         = (14.02 + 7.06 + 31.83) * 1e3  # pb^-1, 2018 A+B+D

def load_binned(dataset, table, is_cov=False):
    fs, path = get_hist_path(LOCATION, CONFIG_SUITE, RUNTAG, dataset,
                             OBJSYST, WTSYST, table, is_cov, -1, -1)
    with fs.open(path, 'rb') as f:
        return np.load(f)

def ht_weight(dataset, xsec):
    return xsec / lookup_count(RUNTAG, dataset) * LUMI

def load_ht_array(table):
    result = None
    for dataset, xsec in HT_BINS:
        arr = load_binned(dataset, table)
        w = ht_weight(dataset, xsec)
        result = w * arr if result is None else result + w * arr
    return result

def load_ht_covmat(table):
    result = None
    for dataset, xsec in HT_BINS:
        arr = load_binned(dataset, table, is_cov=True)
        w = ht_weight(dataset, xsec)
        result = w**2 * arr if result is None else result + w**2 * arr
    return result

def clean_covmat(cov):
    """Replace NaN/zero diagonal entries with 1e30 (zero weight in chi-squared)."""
    bad = np.isnan(np.diag(cov)) | (np.diag(cov) == 0)
    cov = np.where(np.isnan(cov), 0.0, cov)
    np.fill_diagonal(cov, np.where(bad, 1e30, np.diag(cov)))
    return cov

# --- reco ---
print("Building HT-stitched Pythia reco...")
reco_vals = np.nan_to_num(load_ht_array('proj_totalReco'), nan=0.0)
reco_cov  = clean_covmat(load_ht_covmat('proj_totalReco'))
print("Valid reco bins:", np.sum(np.diag(reco_cov) < 1e29), "of", len(reco_vals))

fs0, bincfgpath0 = get_hist_bincfg_path(LOCATION, CONFIG_SUITE, RUNTAG,
                                         HT_BINS[0][0], OBJSYST, 'proj_totalReco')
binning_reco = ArbitraryBinning()
with fs0.open(bincfgpath0, 'r') as f:
    binning_reco.from_dict(json.load(f))

reco = Histogram(reco_vals, reco_cov, binning_reco)
reco.compute_invcov()
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))
print("Pythia reco sum:", reco_vals.sum())

# --- gen baseline ---
print("Building HT-stitched gen baseline...")
gen_vals = np.nan_to_num(load_ht_array('proj_totalGen'), nan=0.0)
gen_cov  = np.diag(np.where(gen_vals > 0, gen_vals, 1.0))

fs1, bincfgpath1 = get_hist_bincfg_path(LOCATION, CONFIG_SUITE, RUNTAG,
                                         HT_BINS[0][0], OBJSYST, 'proj_totalGen')
binning_gen = ArbitraryBinning()
with fs1.open(bincfgpath1, 'r') as f:
    binning_gen.from_dict(json.load(f))

genreco_binning = ArbitraryGenRecoBinning()
genreco_binning.from_dict({'gen': binning_gen.to_dict(), 'reco': binning_reco.to_dict()})

mcgen = Histogram(gen_vals, gen_cov, binning_gen)
mcgen.compute_invcov()
mcgen.dump_to_disk(os.path.join(WORKSPACE, 'mcgen'))
print("Pythia gen sum:", gen_vals.sum())

# --- detector model ---
print("Building HT-stitched detector model...")
umG  = np.nan_to_num(load_ht_array('proj_unmatchedGen'),  nan=0.0)
umR  = np.nan_to_num(load_ht_array('proj_unmatchedReco'), nan=0.0)
totG = np.nan_to_num(load_ht_array('proj_totalGen'),      nan=0.0)
totR = np.nan_to_num(load_ht_array('proj_totalReco'),     nan=0.0)
nGen = nReco = len(totG)

gamma0 = umG / np.where(totG == 0, 1.0, totG)
gamma0[gamma0 == 1] = 0                        # prevent gen bins from decoupling
Rdenom = totR - umR
rho0   = umR / np.where(Rdenom <= 0, 1.0, Rdenom)

transfer_raw = np.nan_to_num(load_ht_array('proj_transfer'), nan=0.0)
n = int(np.sqrt(len(transfer_raw)))
t0 = transfer_raw.reshape(n, n)
matchedG = np.where(totG - umG == 0, 1.0, totG - umG)
t0 /= matchedG[np.newaxis, :]

model = DetectorModel(
    transfer0          = t0,
    gamma0             = gamma0,
    rho0               = rho0,
    transferVariations = np.zeros((0, nReco, nGen)),
    transferVarIndices = np.array([], dtype=int),
    gammaVariations    = np.zeros((0, nGen)),
    rhoVariations      = np.zeros((0, nReco)),
    binning        = genreco_binning,
    nuisance_names = [],
)
print(model)
model.dump_to_disk(os.path.join(WORKSPACE, 'model'))
print("Workspace written to:", WORKSPACE)
