import fasteigenpy  # must import before torch

import numpy as np
import os

from unfolding.specs import dsspec, detectormodelspec
from unfolding.histogram import Histogram
from unfolding.detectormodel import DetectorModel, load_hist_from_dataset

pythia_v6 : dsspec = {
    'location'     : 'dylan-lxplus-eos',
    'config_suite' : 'EvtMCprojConfig',
    'runtag'       : 'v6',
    'dataset'      : 'DYJetsToLL_Pythia',
    'isMC'         : True,
}

WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace'
OBJSYST   = 'NOM'
WTSYST    = 'nominal'

# --- reco histogram (data to unfold) ---
print("Loading reco histogram...")
reco = Histogram.from_dataset(pythia_v6, 'proj_Reco', WTSYST, OBJSYST)
reco.dump_to_disk(os.path.join(WORKSPACE, 'reco'))

# --- gen histogram (MC truth, used as baseline) ---
print("Loading gen histogram...")
gen = Histogram.from_dataset(pythia_v6, 'proj_Gen', WTSYST, OBJSYST)
gen.dump_to_disk(os.path.join(WORKSPACE, 'gen'))

# --- detector model (built manually due to proj naming convention) ---
print("Building detector model...")

transfer = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_transfer_BINNED_%s.npy' % WTSYST)
umG      = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_unmatchedGen_BINNED_%s.npy' % WTSYST)
umR      = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_unmatchedReco_BINNED_%s.npy' % WTSYST)
totG     = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_Gen_BINNED_%s.npy' % WTSYST)
totR     = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_Reco_BINNED_%s.npy' % WTSYST)

nGen  = len(totG)
nReco = len(totR)

# gamma: fraction of gen that is background (unmatched)
Gdenom = np.where(totG == 0, 1.0, totG)
gamma0 = umG / Gdenom

# rho: fake rate (unmatched reco relative to matched reco)
bkgR   = umR
Rdenom = totR - bkgR
Rdenom = np.where(Rdenom == 0, 1.0, Rdenom)
rho0   = bkgR / Rdenom

# transfer matrix: normalise by matched gen
t0       = transfer.reshape(nReco, nGen)
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

valid_mask_path = os.path.join(WORKSPACE, 'valid_bins.npy')                                                                           
raw_cov = load_hist_from_dataset(pythia_v6, OBJSYST, 'proj_Reco_BINNED_covmat_nominal.npy')                                           
raw_cov = raw_cov[50:-50, 50:-50]                                                                                                     
valid = ~np.isnan(np.diag(raw_cov))                                                                                                   
np.save(valid_mask_path, valid)                                                                                                       
print("Valid bins:", valid.sum(), "of", len(valid))


print("Workspace written to:", WORKSPACE)
