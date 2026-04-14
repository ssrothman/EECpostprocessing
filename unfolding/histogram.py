from codecs import lookup

import torch
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.stats_v2 import smart_inverse
from typing import TypedDict
import numpy as np
import os
import json
from general.fslookup.skim_path import lookup_skim_path
from general.datasets.datasets import lookup_count, lookup_dataset
from unfolding.specs import dsspec, whichhistspec, histspec

class Histogram:
    def __init__(self, values : np.ndarray, covmat : np.ndarray, invcov : np.ndarray, binning : ArbitraryBinning):
        self._values = values
        self._covmat = covmat
        self._invcov = invcov
        self._binning = binning

        self._device = 'numpy'
        
    def imult(self, w : float) -> "Histogram":
        self._values *= w
        self._covmat *= w * w
        self._invcov /= w * w

        return self

    def iadd(self, other : "Histogram", w : float = 1.0, skip_invcov : bool = False) -> "Histogram":
        assert self._binning == other._binning
        assert isinstance(self._values, np.ndarray) and isinstance(other._values, np.ndarray)
        assert isinstance(self._covmat, np.ndarray) and isinstance(other._covmat, np.ndarray)

        self._values += w * other._values
        self._covmat += w * w * other._covmat
        if not skip_invcov:
            self._invcov = smart_inverse(self._covmat, False)
        else:
            self._invcov = np.zeros_like(self._covmat)

        return self

    def compute_invcov(self):
        print("Inverting covariance matrix...")
        assert isinstance(self._covmat, np.ndarray)
        self._invcov = smart_inverse(self._covmat, False)

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
    def from_dataset(cls, cfg: dsspec, histcfg: whichhistspec, whichhist : str, skip_invcov : bool = False) -> "Histogram":
        if cfg['isStack']:
            stackcfg = lookup_dataset(
                'stacks',
                cfg['dataset']
            )

            subHs = []

            for subdset in stackcfg['dsets']:
                subcfg = cfg.copy()
                subcfg.update({
                    'dataset' : subdset,
                    'isStack' : False
                })
                subHs.append(
                    cls.from_dataset(
                        subcfg,
                        histcfg,
                        whichhist=whichhist,
                        skip_invcov=True
                    )
                )

            for substack in stackcfg['stacks']:
                subcfg = cfg.copy()
                subcfg.update({
                    'dataset' : substack,
                    'isStack' : True
                })
                subHs.append(
                    cls.from_dataset(
                        subcfg,
                        histcfg,
                        whichhist=whichhist,
                        skip_invcov=True
                    )
                )
            
            H = subHs[0]
            for subH in subHs[1:]:
                H.iadd(subH, skip_invcov=True)

            if not skip_invcov:
                H.compute_invcov()

            return H
        else:
            fs, skimpath = lookup_skim_path(
                cfg['location'],
                cfg['config_suite'],
                cfg['runtag'],
                cfg['dataset'], 
                histcfg['objsyst'],
                cfg['what'] + '_' + whichhist
            )
            

            valspath = skimpath + '_BINNED_%s' % histcfg['wtsyst']
            covpath = skimpath + '_BINNED_covmat_%s' % histcfg['wtsyst']
            
            if cfg['statN'] > 0:
                valspath += '_%dstat%d' % (cfg['statN'], cfg['statK'])
                covpath += '_%dstat%d' % (cfg['statN'], cfg['statK'])

            valspath += '.npy'
            covpath += '.npy'

            binningpath = skimpath + '_bincfg.json'

            print("Reading", valspath, "...")

            with fs.open(valspath, 'rb') as f:
                values = np.load(f)
            with fs.open(covpath, 'rb') as f:
                covmat = np.load(f)

            binning = ArbitraryBinning()
            with fs.open(binningpath, 'r') as f:
                binning.from_dict(json.load(f))

            if cfg['isMC']:
                xsec = lookup_dataset(
                    cfg['runtag'],
                    cfg['dataset']
                )['xsec']
                count = lookup_count(
                    cfg['location'],
                    cfg['config_suite'],
                    cfg['runtag'],
                    cfg['dataset'],
                    histcfg['objsyst']
                )

                wt = 1000 * xsec * cfg['target_lumi'] / count 
                print("\txsec weight:", wt)
                values *= wt
                covmat *= wt * wt

            if not skip_invcov:
                print("Inverting covariance matrix...")
                invcov = smart_inverse(covmat, False)
            else:
                invcov = np.zeros_like(covmat)

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