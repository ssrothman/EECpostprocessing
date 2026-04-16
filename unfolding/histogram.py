from codecs import lookup

import torch
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.stats_v2 import smart_inverse, smart_sqrt
from typing import Literal, TypedDict
import numpy as np
import os
import json
from simonplot import plot_histogram, draw_matrix
from simonplot.binning import PrebinnedBinning
from simonplot.cut import NoopOperation
from simonplot.plottables.PrebinnedDatasets import ValCovPairDataset
from simonplot.variable import BasicPrebinnedVariable, ConstantVariable, CorrelationFromCovariance
from general.fslookup.skim_path import lookup_skim_path
from general.datasets.datasets import lookup_count, lookup_dataset
from unfolding.io import read_hist
from unfolding.specs import dsspec, whichsystspec, histspec

class Histogram:
    def __init__(self, values : np.ndarray, covmat : np.ndarray, binning : ArbitraryBinning,
                 invcov : np.ndarray | None = None, 
                 L : np.ndarray | None = None, Linv : np.ndarray | None = None,
                 eigvals : np.ndarray | None = None, eigvecs : np.ndarray | None = None):
        
        self._values = values
        self._covmat = covmat
        self._binning = binning

        self._device = 'numpy'

        self._invcov = invcov
        self._L = L
        self._Linv = Linv
        self._eigvals = eigvals
        self._eigvecs = eigvecs

        assert self._values.ndim == 1
        assert self._covmat.ndim == 2
        assert self._covmat.shape[0] == self._covmat.shape[1]
        assert self._values.shape[0] == self._covmat.shape[0]
        if self._invcov is not None:
            assert self._invcov.ndim == 2
            assert self._invcov.shape[0] == self._invcov.shape[1]
            assert self._invcov.shape == self._covmat.shape
        if self._L is not None:
            assert self._L.ndim == 2
            assert self._L.shape[0] == self._L.shape[1]
            assert self._L.shape == self._covmat.shape
        if self._Linv is not None:
            assert self._Linv.ndim == 2
            assert self._Linv.shape[0] == self._Linv.shape[1]
            assert self._Linv.shape == self._covmat.shape
        if self._eigvals is not None:
            assert self._eigvals.ndim == 1
            assert self._eigvals.shape[0] == self._covmat.shape[0]
        if self._eigvecs is not None:
            assert self._eigvecs.ndim == 2
            assert self._eigvecs.shape[0] == self._covmat.shape[0]
            assert self._eigvecs.shape[1] == self._covmat.shape[1]

    def _as_numpy(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values

        assert isinstance(values, torch.Tensor)
        return values.detach().cpu().numpy()

    def imult(self, w : float) -> "Histogram":
        self._values *= w
        self._covmat *= w * w

        if self._invcov is not None:
            self._invcov /= w * w
        
        if self._L is not None:
            self._L *= w
        if self._Linv is not None:
            self._Linv /= w
        
        if self._eigvals is not None:
            self._eigvals *= w * w

        # no transformation needed for eigvecs

        return self

    def iadd(self, other : "Histogram", w : float = 1.0) -> "Histogram":
        assert self._binning == other._binning
        assert isinstance(self._values, np.ndarray) and isinstance(other._values, np.ndarray)
        assert isinstance(self._covmat, np.ndarray) and isinstance(other._covmat, np.ndarray)

        self._values += w * other._values
        self._covmat += w * w * other._covmat

        # invalidate the inverse covariance and eigendecomposition since they are no longer correct
        self._invcov = None
        self._L = None 
        self._Linv = None
        self._eigvals = None
        self._eigvecs = None

        return self

    # in principle compute_invcov() and compute_sqrt() could be set up to share computation
    # but I think it doesn't matter that much...

    def compute_invcov(self) -> None:
        print("Inverting covariance matrix...")
        assert isinstance(self._covmat, np.ndarray)
        self._invcov, self._eigvals, self._eigvecs = smart_inverse(self._covmat, True)

    def compute_sqrt(self) -> None:
        print("Computing sqrt of covariance matrix...")
        assert isinstance(self._covmat, np.ndarray)
        self._L, self._Linv = smart_sqrt(self._covmat)

    @property
    def values(self) -> np.ndarray | torch.Tensor:
        return self._values 
    
    @property
    def covmat(self) -> np.ndarray | torch.Tensor:
        return self._covmat
    
    @property
    def invcov(self) -> np.ndarray | torch.Tensor:
        if self._invcov is None:
            self.compute_invcov()
        assert self._invcov is not None, "This should be impossible!"
        return self._invcov
    
    @property
    def L(self) -> np.ndarray | torch.Tensor:
        if self._L is None:
            self.compute_sqrt()
        assert self._L is not None, "This should be impossible!"
        return self._L
    
    @property
    def Linv(self) -> np.ndarray | torch.Tensor:
        if self._Linv is None:
            self.compute_sqrt()
        assert self._Linv is not None, "This should be impossible!"
        return self._Linv

    @property
    def binning(self) -> ArbitraryBinning:
        return self._binning

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def from_dataset(cls, cfg: dsspec, histcfg: whichsystspec, whichhist : Literal['totalReco', 'totalGen', 'unmatchedReco', 'unmatchedGen', 'untransferedReco', 'untransferedGen']) -> "Histogram":
        values, covmat, binning = read_hist(
            cfg,
            histcfg,
            whichhist,
            read_cov=True
        )

        return cls(values, covmat, binning)

    @classmethod
    def from_disk(cls, where: str) -> "Histogram":
        with open(os.path.join(where, 'values.npy'), 'rb') as f:
            values = np.load(f)
        with open(os.path.join(where, 'cov.npy'), 'rb') as f:
            covmat = np.load(f)
        with open(os.path.join(where, 'bincfg.json'), 'r') as f:
            binning = ArbitraryBinning()
            binning.from_dict(json.load(f))

        extras = {}
        if os.path.exists(os.path.join(where, 'invcov.npy')):
            with open(os.path.join(where, 'invcov.npy'), 'rb') as f:
                extras['invcov'] = np.load(f)
        if os.path.exists(os.path.join(where, 'L.npy')):
            with open(os.path.join(where, 'L.npy'), 'rb') as f:
                extras['L'] = np.load(f)
        if os.path.exists(os.path.join(where, 'Linv.npy')):
            with open(os.path.join(where, 'Linv.npy'), 'rb') as f:
                extras['Linv'] = np.load(f)
        if os.path.exists(os.path.join(where, 'eigvals.npy')):
            with open(os.path.join(where, 'eigvals.npy'), 'rb') as f:
                extras['eigvals'] = np.load(f)
        if os.path.exists(os.path.join(where, 'eigvecs.npy')):
            with open(os.path.join(where, 'eigvecs.npy'), 'rb') as f:
                extras['eigvecs'] = np.load(f)
        
        return cls(values, covmat, binning, **extras)

    def dump_to_disk(self, where: str) -> None:
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, 'values.npy'), 'wb') as f:
            np.save(f, self._values)
        with open(os.path.join(where, 'cov.npy'), 'wb') as f:
            np.save(f, self._covmat)
        with open(os.path.join(where, 'bincfg.json'), 'w') as f:
            json.dump(self._binning.to_dict(), f, indent=4)

        if self._invcov is not None:
            with open(os.path.join(where, 'invcov.npy'), 'wb') as f:
                np.save(f, self._invcov)
        if self._L is not None:
            with open(os.path.join(where, 'L.npy'), 'wb') as f:
                np.save(f, self._L)
        if self._Linv is not None:
            with open(os.path.join(where, 'Linv.npy'), 'wb') as f:
                np.save(f, self._Linv)
        if self._eigvals is not None:
            with open(os.path.join(where, 'eigvals.npy'), 'wb') as f:
                np.save(f, self._eigvals)
        if self._eigvecs is not None:
            with open(os.path.join(where, 'eigvecs.npy'), 'wb') as f:
                np.save(f, self._eigvecs)

    def plot(
        self,
        output_folder: str | None = None,
        extratext: str | None = None,
    ) -> None:
        
        if self.invcov is None:
            self.compute_invcov()
        
        values = self._as_numpy(self.values)
        covmat = self._as_numpy(self.covmat)
        invcov = self._as_numpy(self.invcov)
        invcov_cov = invcov @ covmat

        variable = BasicPrebinnedVariable()
        cut = NoopOperation()
        weight = ConstantVariable(1.0)
        binning = PrebinnedBinning()

        base_prefix = 'histogram'

        values_dataset = ValCovPairDataset(
            key=f'{base_prefix}_values',
            color=None,
            label=None,
            data=(values, covmat),
            binning=self._binning,
            isMC=True,
        )
        cov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_cov',
            color=None,
            label=None,
            data=(values, covmat),
            binning=self._binning,
            isMC=True,
        )
        invcov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_invcov',
            color=None,
            label=None,
            data=(values, invcov),
            binning=self._binning,
            isMC=True,
        )
        invcov_cov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_invcov_times_cov',
            color=None,
            label=None,
            data=(values, invcov_cov),
            binning=self._binning,
            isMC=True,
        )

        plot_histogram(
            variable,
            cut,
            weight,
            values_dataset,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_values_histogram',
            no_lumi_normalization=True,
        )

        draw_matrix(
            variable,
            cut,
            cov_dataset,
            binning,
            extratext=extratext,
            sym=True,
            logc=True,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_covariance_matrix',
        )

        draw_matrix(
            variable,
            cut,
            invcov_dataset,
            binning,
            extratext=extratext,
            sym=True,
            logc=True,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_inverse_covariance_matrix',
        )

        draw_matrix(
            variable,
            cut,
            invcov_cov_dataset,
            binning,
            extratext=extratext,
            sym=True,
            logc=False,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_inverse_covariance_times_covariance_matrix',
        )

        corr_variable = CorrelationFromCovariance(variable)

        draw_matrix(
            corr_variable,
            cut,
            cov_dataset,
            binning,
            extratext=extratext,
            sym=True,
            logc=False,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_covariance_correlation_matrix',
        )

        draw_matrix(
            corr_variable,
            cut,
            invcov_dataset,
            binning,
            extratext=extratext,
            sym=True,
            logc=False,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_inverse_covariance_correlation_matrix',
        )


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

    