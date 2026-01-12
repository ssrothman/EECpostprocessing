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
    def __init__(self, transfer0 : np.ndarray):
        self._transfer0 = transfer0

    @property
    def transfer0(self) -> np.ndarray:
        return self._transfer0

    @classmethod
    def from_dataset(cls, dset : dsspec, cfg : detectormodelspec) -> "DetectorModel":
        t0 = load_hist_from_dataset(dset, 'nominal', 'res4tee_transfer_BINNED_nominal.npy')
        return cls(t0)
    
    @classmethod
    def from_disk(cls, path : str) -> "DetectorModel":
        with open(os.path.join(path, 'transfer0.npy'), 'rb') as f:
            t0 = np.load(f)
        return cls(t0)
    
    def dump_to_disk(self, where : str):
        with open(os.path.join(where, 'transfer0.npy'), 'wb') as f:
            np.save(f, self._transfer0)
