from codecs import lookup

import torch
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.stats_v2 import smart_inverse
from typing import TypedDict
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
    def __init__(self, values : np.ndarray, covmat : np.ndarray, invcov : np.ndarray, binning : ArbitraryBinning):
        self._values = values
        self._covmat = covmat
        self._invcov = invcov
        self._binning = binning

        self._device = 'numpy'
        
    def _as_numpy(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values

        assert isinstance(values, torch.Tensor)
        return values.detach().cpu().numpy()

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
    def from_dataset(cls, cfg: dsspec, histcfg: whichsystspec, whichhist : str) -> "Histogram":
        values, covmat, binning = read_hist(
            cfg,
            histcfg,
            whichhist,
            read_cov=True
        )

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

    def plot(
        self,
        output_folder: str | None = None,
        extratext: str | None = None,
    ) -> None:
        
        values = self._as_numpy(self._values)
        covmat = self._as_numpy(self._covmat)
        invcov = self._as_numpy(self._invcov)

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

    