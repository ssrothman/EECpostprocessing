import os
from general.fslookup.skim_path import lookup_skim_path
from unfolding.specs import dsspec, detectormodelspec
import numpy as np

def load_hist_from_dataset(dset:dsspec, objsyst, what:str) -> np.ndarray:
    fs, path = lookup_skim_path(
        dset['location'],
        dset['config_suite'],
        dset['runtag'],
        dset['dataset'], 
        objsyst,
        what
    )
    with fs.open(path, 'rb') as f:
        arr =  np.load(f)
    return arr
    
class DetectorModel:
    def __init__(self, transfer0 : np.ndarray, gamma0 : np.ndarray, rho0 : np.ndarray):
        self._transfer0 = transfer0
        self._gamma0 = gamma0
        self._rho0 = rho0

    def forward(self, gen : np.ndarray) -> np.ndarray:
        genPure = gen * (1.0 - self._gamma0)
        recoPure = self._transfer0 @ genPure
        reco = recoPure * (1.0 + self._rho0)
        return reco

    @classmethod
    def from_dataset(cls, dset : dsspec, cfg : detectormodelspec) -> "DetectorModel":
        t0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_transfer_BINNED_nominal.npy')

        umG0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_unmatchedGen_BINNED_nominal.npy')
        utG0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_untransferedGen_BINNED_nominal.npy')
        bkgG0 = umG0 + utG0
        totG0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_totalGen_BINNED_nominal.npy')
        Gdenom = np.where(totG0 == 0, 1.0, totG0)
        gamma0 = bkgG0 / Gdenom
        print()
        print("gen bkg: ", bkgG0.sum())
        print("gen total: ", totG0.sum())
        print("gen * gamma0: ", (gamma0 * totG0).sum())
        print()

        umR0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_unmatchedReco_BINNED_nominal.npy')
        utR0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_untransferedReco_BINNED_nominal.npy')
        bkgR0 = umR0 + utR0
        totR0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_totalReco_BINNED_nominal.npy')
        Rdenom = totR0 - bkgR0
        Rdenom = np.where(Rdenom == 0, 1.0, Rdenom)
        rho0 = bkgR0 / Rdenom
        print()
        print("reco bkg: ", bkgR0.sum())
        print("reco total: ", totR0.sum())
        print("reco * rho0: ", (rho0 * (totR0 - bkgR0)).sum())
        print() 

        t0 = t0.reshape(len(rho0), len(gamma0))
        print("t0 @ ones: ", (t0 @ np.ones_like(gamma0)).sum())
        tdenom = totG0 - bkgG0
        tdenom = np.where(tdenom == 0, 1.0, tdenom)
        t0 /= tdenom[None, :]
        print("t0 normalized @ genPure: ", (t0 @ (totG0 - bkgG0)).sum())

        return cls(t0, gamma0, rho0)
    
    @classmethod
    def from_disk(cls, path : str) -> "DetectorModel":
        with open(os.path.join(path, 'transfer0.npy'), 'rb') as f:
            t0 = np.load(f)
        with open(os.path.join(path, 'gamma0.npy'), 'rb') as f:
            gamma0 = np.load(f)
        with open(os.path.join(path, 'rho0.npy'), 'rb') as f:
            rho0 = np.load(f)
        return cls(t0, gamma0, rho0)
    
    def dump_to_disk(self, where : str):
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, 'transfer0.npy'), 'wb') as f:
            np.save(f, self._transfer0)
        with open(os.path.join(where, 'gamma0.npy'), 'wb') as f:
            np.save(f, self._gamma0)
        with open(os.path.join(where, 'rho0.npy'), 'wb') as f:
            np.save(f, self._rho0)