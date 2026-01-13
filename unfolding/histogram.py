import torch
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.stats_v2 import smart_inverse
from typing import TypedDict
import numpy as np
import os
import json
from general.fslookup.skim_path import lookup_skim_path
from unfolding.specs import dsspec

class Histogram:
    def __init__(self, values : np.ndarray, covmat : np.ndarray, invcov : np.ndarray, binning : ArbitraryBinning):
        self._values = values
        self._covmat = covmat
        self._invcov = invcov
        self._binning = binning

        self._device = 'numpy'
        
    @property
    def values(self) -> np.ndarray | torch.Tensor:
        return self._values 
    
    @property
    def covmat(self) -> np.ndarray | torch.Tensor:
        return self._covmat
    
    @property
    def invcov(self) -> np.ndarray | torch.Tensor:
        return self._invcov
    
    @property
    def binning(self) -> ArbitraryBinning:
        return self._binning

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def from_dataset(cls, cfg: dsspec, what: str, wtsyst: str, objsyst: str) -> "Histogram":
        fs, skimpath = lookup_skim_path(
            cfg['location'],
            cfg['config_suite'],
            cfg['runtag'],
            cfg['dataset'], 
            objsyst,
            what
        )
        
        valspath = skimpath + '_BINNED_%s.npy' % wtsyst
        covpath = skimpath + '_BINNED_covmat_%s.npy' % wtsyst
        binningpath = skimpath + '_bincfg.json'

        print("Reading", valspath, "...")

        with fs.open(valspath, 'rb') as f:
            values = np.load(f)
        with fs.open(covpath, 'rb') as f:
            covmat = np.load(f)

        binning = ArbitraryBinning()
        with fs.open(binningpath, 'r') as f:
            binning.from_dict(json.load(f))

        print("Inverting covariance matrix...")
        invcov = smart_inverse(covmat, False)

        return cls(values, covmat, invcov, binning)

    @classmethod
    def from_disk(cls, where: str) -> "Histogram":
        with open(os.path.join(where, 'values.npy'), 'rb') as f:
            values = np.load(f)
        with open(os.path.join(where, 'cov.npy'), 'rb') as f:
            covmat = np.load(f)
        with open(os.path.join(where, 'invcov.npy'), 'rb') as f:
            invcov = np.load(f)
        with open(os.path.join(where, 'bincfg.json'), 'r') as f:
            binning = ArbitraryBinning()
            binning.from_dict(json.load(f))
        
        return cls(values, covmat, invcov, binning)

    def dump_to_disk(self, where: str) -> None:
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, 'values.npy'), 'wb') as f:
            np.save(f, self._values)
        with open(os.path.join(where, 'cov.npy'), 'wb') as f:
            np.save(f, self._covmat)
        with open(os.path.join(where, 'invcov.npy'), 'wb') as f:
            np.save(f, self._invcov)
        with open(os.path.join(where, 'bincfg.json'), 'w') as f:
            json.dump(self._binning.to_dict(), f, indent=4)


    def to_torch(self):
        if self._device != 'numpy':
            return self
        
        self._device = 'cpu'

        assert isinstance(self._values, np.ndarray)
        assert isinstance(self._covmat, np.ndarray)
        assert isinstance(self._invcov, np.ndarray)

        self._values =  torch.from_numpy(self._values)
        self._covmat = torch.from_numpy(self._covmat)
        self._invcov = torch.from_numpy(self._invcov)

        return self

    def to_numpy(self, *args, **kwargs):
        if self._device == 'numpy':
            return self
        else:
            self._device = 'numpy'

            assert isinstance(self._values, torch.Tensor)
            assert isinstance(self._covmat, torch.Tensor)
            assert isinstance(self._invcov, torch.Tensor)

            self._values = self._values.numpy(*args, **kwargs)
            self._covmat = self._covmat.numpy(*args, **kwargs)
            self._invcov = self._invcov.numpy(*args, **kwargs)

            return self

    def to(self, device, *args, **kwargs):
        if device == 'numpy':
            return self.to_numpy(*args, **kwargs)
        else:
            self.to_torch()

            self._device = device

            assert isinstance(self._values, torch.Tensor)
            assert isinstance(self._covmat, torch.Tensor)
            assert isinstance(self._invcov, torch.Tensor)

            self._values = self._values.to(device, *args, **kwargs)
            self._covmat = self._covmat.to(device, *args, **kwargs)
            self._invcov = self._invcov.to(device, *args, **kwargs)

            return self
        
    def detach(self):
        if self._device == 'numpy':
            return self
        
        assert isinstance(self._values, torch.Tensor)
        assert isinstance(self._covmat, torch.Tensor)
        assert isinstance(self._invcov, torch.Tensor)

        self._values = self._values.detach()
        self._covmat = self._covmat.detach()
        self._invcov = self._invcov.detach()

        return self