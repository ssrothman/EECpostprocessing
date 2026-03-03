import os
from typing import overload
from general.fslookup.skim_path import lookup_skim_path
from unfolding.specs import dsspec, detectormodelspec
import numpy as np
import torch

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

def get_model_matrices(dset : dsspec, syst : str, isobjsyst : bool, what : str):
        if isobjsyst:
            wtsyst = 'nominal'
            objsyst = syst
        else:
            objsyst = 'nominal'
            wtsyst = syst
        
        t = load_hist_from_dataset(
            dset, objsyst, 
            '%s_transfer_BINNED_%s.npy' % (what, wtsyst)
        )

        umG = load_hist_from_dataset(
            dset, objsyst, 
            '%s_unmatchedGen_BINNED_%s.npy' % (what, wtsyst)
        )
        utG = load_hist_from_dataset(
            dset, objsyst, 
            '%s_untransferedGen_BINNED_%s.npy' % (what, wtsyst)
        )
        bkgG = umG + utG

        totG = load_hist_from_dataset(
            dset, objsyst, 
            '%s_totalGen_BINNED_%s.npy' % (what, wtsyst)
        )

        Gdenom = np.where(totG == 0, 1.0, totG)
        gamma = bkgG / Gdenom

        umR = load_hist_from_dataset(
            dset, objsyst, 
            '%s_unmatchedReco_BINNED_%s.npy' % (what, wtsyst)
        )
        utR = load_hist_from_dataset(
            dset, objsyst, 
            '%s_untransferedReco_BINNED_%s.npy' % (what, wtsyst)
        )
        bkgR = umR + utR

        totR = load_hist_from_dataset(
            dset, objsyst, 
            '%s_totalReco_BINNED_%s.npy' % (what, wtsyst)
        )

        Rdenom = totR - bkgR
        Rdenom = np.where(Rdenom == 0, 1.0, Rdenom)
        rho = bkgR / Rdenom

        t = t.reshape(len(rho), len(gamma))
        tdenom = totG - bkgG
        tdenom = np.where(tdenom == 0, 1.0, tdenom)
        t /= tdenom[None, :]

        return t, gamma, rho
    
class DetectorModel:
    #list of arrays - used by save/load routines
    #class attribute
    _arrays = [
        'transfer0',                    # nominal transfer matrix
        'gamma0',                       # nominal gen background template
        'rho0',                         # nominal reco background template
        'transferVariations',           # systematic variations of transfer matrix
        'transferVarIndices',           # indices of which systematics have transfer matrix variations
        'gammaVariations',              # systematic variations of gen background template
        'rhoVariations',                # systematic variations of reco background template
    ]

    def __init__(self, 
                 transfer0 : np.ndarray, 
                 gamma0 : np.ndarray, 
                 rho0 : np.ndarray,
                 transferVariations : np.ndarray,
                 transferVarIndices : np.ndarray,
                 gammaVariations : np.ndarray,
                 rhoVariations : np.ndarray):
        
        self._transfer0 = transfer0
        self._gamma0 = gamma0
        self._rho0 = rho0

        self._transferVariations = transferVariations
        self._transferVarIndices = transferVarIndices
        self._gammaVariations = gammaVariations
        self._rhoVariations = rhoVariations 

        self._device = 'numpy'

        self._nSyst = self._gammaVariations.shape[0]
        self._nGen = self._gamma0.shape[0]
        self._nReco = self._rho0.shape[0]
        self._nTransferSyst = self._transferVariations.shape[0]

        # some basic checks that inputs are consistent
        assert self._transfer0.shape == (self._nReco, self._nGen)
        assert self._rho0.shape == (self._nReco,)
        assert self._gamma0.shape == (self._nGen,)

        assert self._transferVarIndices.shape == (self._nTransferSyst,)
        if len(self._transferVarIndices) > 0:
            assert len(self._transferVarIndices) == self._nTransferSyst
            assert np.min(self._transferVarIndices) >= 0
            assert np.max(self._transferVarIndices) < self._nSyst

        assert self._transferVariations.shape == (self._nTransferSyst, self._nReco, self._nGen)
        assert self._gammaVariations.shape == (self._nSyst, self._nGen)
        assert self._rhoVariations.shape == (self._nSyst, self._nReco)

    def __str__(self) -> str:
        result = 'DetectorModel:\n'
        result += f'  nGen: {self._nGen}\n'
        result += f'  nReco: {self._nReco}\n'
        result += f'  nSyst: {self._nSyst}\n'
        result += f'  nTransferSyst: {self._nTransferSyst}\n'
        result += f'  transfer0 {self._transfer0.shape}, {self._transfer0.dtype}\n'
        result += f'  gamma0 {self._gamma0.shape}, {self._gamma0.dtype}\n'
        result += f'  rho0 {self._rho0.shape}, {self._rho0.dtype}\n'
        return result
    
    @property
    def device(self) -> str:
        return self._device

    @property
    def nSyst(self) -> int:
        return self._nSyst
    
    @property
    def nGen(self) -> int:
        return self._nGen
    
    @property
    def nTransferSyst(self) -> int:
        return self._nTransferSyst

    @property
    def nReco(self) -> int:
        return self._nReco

    @overload
    def _gamma(self, theta: torch.Tensor) -> torch.Tensor:
        ...
    @overload
    def _gamma(self, theta: np.ndarray) -> np.ndarray:
        ...

    #implementation
    def _gamma(self, theta):
        if isinstance(theta, torch.Tensor):
            return self._gamma0 + torch.tensordot(theta, self._gammaVariations, 1)
        else:
            return self._gamma0 + np.tensordot(theta, self._gammaVariations, 1)
    
    @overload
    def _rho(self, theta: torch.Tensor) -> torch.Tensor:
        ...
    @overload
    def _rho(self, theta: np.ndarray) -> np.ndarray:
        ...

    #implementation
    def _rho(self, theta):
        if isinstance(theta, torch.Tensor):
            return self._rho0 + torch.tensordot(theta, self._rhoVariations, 1)
        else:
            return self._rho0 + np.tensordot(theta, self._rhoVariations, 1)
    
    @overload
    def _T(self, theta: torch.Tensor) -> torch.Tensor:
        ...
    @overload   
    def _T(self, theta: np.ndarray) -> np.ndarray:
        ...
    #implementation
    def _T(self, theta):
        if len(self._transferVarIndices) > 0:
            if isinstance(theta, torch.Tensor):
                return self._transfer0 + torch.tensordot(theta[self._transferVarIndices], self._transferVariations, 1) 
            else:
                return self._transfer0 + np.tensordot(theta[self._transferVarIndices], self._transferVariations, 1) 
        else:
            return self._transfer0


    @overload
    def forward(self, beta : torch.Tensor, theta : torch.Tensor) -> torch.Tensor:
        ...
    @overload
    def forward(self, beta : np.ndarray, theta : np.ndarray) -> np.ndarray:
        ...
    #implementation
    def forward(self, beta, theta):
        gamma = self._gamma(theta)
        rho = self._rho(theta)
        T = self._T(theta)

        genPure = beta * (1.0 - gamma)
        recoPure = T @ genPure
        reco = recoPure * (1.0 + rho)

        return reco

    @classmethod
    def from_dataset(cls, dset : dsspec, cfg : detectormodelspec, what : str) -> "DetectorModel":
        t0, gamma0, rho0 = get_model_matrices(
            dset,
            'nominal',
            False,
            what
        )

        gammaVariations = []
        rhoVariations = []
        transferVariations = []
        transferVarIndices = []

        for syst in cfg['systematics']:
            if syst['onesided']:
                t_up, gamma_up, rho_up = get_model_matrices(
                    dset,
                    syst['name'],
                    syst['isobjsyst'],
                    what
                )
                dT = t_up - t0
                dGamma = gamma_up - gamma0
                dRho = rho_up - rho0
            else:
                t_up, gamma_up, rho_up = get_model_matrices(
                    dset,
                    syst['name'] + "Up",
                    syst['isobjsyst'],
                    what
                )
                t_dn, gamma_dn, rho_dn = get_model_matrices(
                    dset,
                    syst['name'] + "Down",
                    syst['isobjsyst'],
                    what
                )
                dT = 0.5 * (t_up - t_dn)
                dGamma = 0.5 * (gamma_up - gamma_dn)
                dRho = 0.5 * (rho_up - rho_dn)

            gammaVariations.append(dGamma)
            rhoVariations.append(dRho)
            if syst['varytransfer']:
                transferVariations.append(dT)
                transferVarIndices.append(len(gammaVariations) - 1)

        if len(gammaVariations) == 0:
            gammaVariations = np.zeros((0, len(gamma0)))
            rhoVariations = np.zeros((0, len(rho0)))
        else:
            gammaVariations = np.stack(gammaVariations, axis=0)
            rhoVariations = np.stack(rhoVariations, axis=0)

        if len(transferVariations) == 0:
            transferVariations =  np.zeros((0, t0.shape[0], t0.shape[1]))
            transferVarIndices = np.array([], dtype=int)
        else:
            transferVariations = np.stack(transferVariations, axis=0)
            transferVarIndices = np.array(transferVarIndices, dtype=int)

        return cls(t0, gamma0, rho0, 
                   transferVariations, transferVarIndices,
                   gammaVariations, rhoVariations)
    
    @classmethod
    def from_disk(cls, path : str) -> "DetectorModel":
        arrays = {}
        for arrname in cls._arrays:
            with open(os.path.join(path, f'{arrname}.npy'), 'rb') as f:
                arrays[arrname] = np.load(f)
        return cls(**arrays)
    
    def dump_to_disk(self, where : str):
        os.makedirs(where, exist_ok=True)
        for arrname in self._arrays:
            arr = getattr(self, f'_{arrname}')
            with open(os.path.join(where, f'{arrname}.npy'), 'wb') as f:
                np.save(f, arr)

    def to_torch(self):
        if self._device != 'numpy':
            return self
        
        self._device = 'cpu'

        for arrname in self._arrays:
            arr = getattr(self, f'_{arrname}')
            arr_torch = torch.from_numpy(arr)
            setattr(self, f'_{arrname}', arr_torch)

        return self
    
    def to_numpy(self, *args, **kwargs):
        if self._device == 'numpy':
            return self
        
        self._device = 'numpy'

        for arrname in self._arrays:
            arr = getattr(self, f'_{arrname}')
            arr_numpy = arr.numpy(*args, **kwargs)
            setattr(self, f'_{arrname}', arr_numpy)

        return self
    
    def to(self, device, *args, **kwargs):
        if device == 'numpy':
            return self.to_numpy(*args, **kwargs)
        else:
            self.to_torch()

            self._device = device
            for arrname in self._arrays:
                arr = getattr(self, f'_{arrname}')
                arr_to = arr.to(device, *args, **kwargs)
                setattr(self, f'_{arrname}', arr_to)
            return self
        
    def detach(self):
        if self._device == 'numpy':
            return self
        
        for arrname in self._arrays:
            arr = getattr(self, f'_{arrname}')
            arr_detached = arr.detach()
            setattr(self, f'_{arrname}', arr_detached)

        return self