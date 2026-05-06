import os
from typing import overload
from general.fslookup.skim_path import lookup_skim_path
from unfolding.specs import dsspec, detectormodelspec, whichsystspec, systspec
import numpy as np
import torch
import hist
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonplot import plot_histogram, draw_matrix
from simonplot.binning import PrebinnedBinning
from simonplot.cut import NoopOperation, SliceOperation
from simonplot.plottables.PrebinnedDatasets import TransferMatrixDataset, ValCovPairDataset, ValNoCovDataset
from simonplot.variable import BasicPrebinnedVariable, ConstantVariable
from unfolding.io import read_hist


def hist_from_syst(spec : systspec, updn : str | None) -> whichsystspec:
    thename = spec['name']
    if updn is not None:
        if spec['isobjsyst']:
            if 'DOWN' in updn.upper():
                updn = 'DN'
            thename += '_' + updn.upper()
        else:
            thename += updn
    
    if spec['isobjsyst']:
        return {
            'objsyst' : thename,
            'wtsyst' : 'nominal'
        }
    else:
        return {
            'objsyst' : 'nominal',
            'wtsyst' : thename
        }

def get_transfer_binning(dset : dsspec) -> ArbitraryGenRecoBinning:
    binning = read_hist(
        dset, 
        {'objsyst' : 'nominal', 'wtsyst' : 'nominal'},
        'transfer',
        False
    )[-1]
    return binning


def get_model_matrices(dset : dsspec, hist : whichsystspec):
        t = read_hist(
            dset, hist, 
            'transfer',
            False
        )[0]

        umG = read_hist(
            dset, hist, 
            'unmatchedGen',
            False
        )[0]
        utG = read_hist(
            dset, hist, 
            'untransferedGen',
            False
        )[0]
        bkgG = umG + utG

        totG = read_hist(
            dset, hist, 
            'totalGen',
            False
        )[0]

        Gdenom = np.where(totG == 0, 1.0, totG)
        gamma = bkgG / Gdenom

        umR = read_hist(
            dset, hist, 
            'unmatchedReco',
            False
        )[0]
        utR = read_hist(
            dset, hist, 
            'untransferedReco',
            False
        )[0]
        bkgR = umR + utR

        totR = read_hist(
            dset, hist, 
            'totalReco',
            False
        )[0]

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
                 rhoVariations : np.ndarray,
                 binning : ArbitraryGenRecoBinning,
                 nuisance_names : list[str]):
        
        self._transfer0 = transfer0
        self._gamma0 = gamma0
        self._rho0 = rho0

        self._transferVariations = transferVariations
        self._transferVarIndices = transferVarIndices
        self._gammaVariations = gammaVariations
        self._rhoVariations = rhoVariations 

        self._binning = binning
        self._nuisance_names = nuisance_names

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

        assert len(nuisance_names) == self._nSyst

    def __str__(self) -> str:
        result = 'DetectorModel:\n'
        result += f'  nGen: {self._nGen}\n'
        result += f'  nReco: {self._nReco}\n'
        result += f'  nSyst: {self._nSyst}\n'
        result += f'  nTransferSyst: {self._nTransferSyst}\n'
        result += f'  transfer0 {self._transfer0.shape}, {self._transfer0.dtype}\n'
        result += f'  gamma0 {self._gamma0.shape}, {self._gamma0.dtype}\n'
        result += f'  rho0 {self._rho0.shape}, {self._rho0.dtype}\n'
        result += f'  nuisance names: {self._nuisance_names}\n'
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

    @property
    def binning(self) -> ArbitraryGenRecoBinning:
        return self._binning
    
    @property
    def nuisance_names(self) -> list[str]:
        return self._nuisance_names

    def _as_numpy(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values

        assert isinstance(values, torch.Tensor)
        return values.detach().cpu().numpy()

    def plot(self, output_folder: str | None = None, extratext: str | None = None, 
             detailed=False, nuisances=False) -> None:
        base_prefix = 'detectormodel'

        transfer0 = self._as_numpy(self._transfer0)
        gamma0 = self._as_numpy(self._gamma0)
        rho0 = self._as_numpy(self._rho0)
        gamma_var = self._as_numpy(self._gammaVariations)
        rho_var = self._as_numpy(self._rhoVariations)
        transfer_var = self._as_numpy(self._transferVariations)

        variable = BasicPrebinnedVariable()
        cut = NoopOperation()
        weight = ConstantVariable(1.0)
        plot_binning = PrebinnedBinning()

        gamma_dataset = ValNoCovDataset(
            key=f'{base_prefix}_gamma0',
            color=None,
            label=None,
            data=gamma0,
            binning=self.binning.genbinning,
            isMC=True,
        )
        rho_dataset = ValNoCovDataset(
            key=f'{base_prefix}_rho0',
            color=None,
            label=None,
            data=rho0,
            binning=self.binning.recobinning,
            isMC=True,
        )

        plot_histogram(
            variable,
            cut,
            weight,
            gamma_dataset,
            plot_binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_gamma_nominal',
            no_lumi_normalization=True,
        )

        plot_histogram(
            variable,
            cut,
            weight,
            rho_dataset,
            plot_binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_rho_nominal',
            no_lumi_normalization=True,
        )

        if nuisances:
            delta_gammma_datasets = []
            delta_rho_datasets = []
            for i in range(self._nSyst):
                delta_gamma = gamma_var[i]
                delta_rho = rho_var[i]

                delta_gamma_dataset = ValNoCovDataset(
                    key=f'{base_prefix}_gamma_variation_{i}',
                    color=None,
                    label=self.nuisance_names[i],
                    data=np.abs(delta_gamma),
                    binning=self.binning.genbinning,
                    isMC=True,
                )
                delta_rho_dataset = ValNoCovDataset(
                    key=f'{base_prefix}_rho_variation_{i}',
                    color=None,
                    label=self.nuisance_names[i],
                    data=np.abs(delta_rho),
                    binning=self.binning.recobinning,
                    isMC=True,
                )

                delta_gammma_datasets.append(delta_gamma_dataset)
                delta_rho_datasets.append(delta_rho_dataset)
            
            plot_histogram(
                variable,
                cut,
                weight,
                delta_gammma_datasets,
                plot_binning,
                extratext=extratext,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_gamma_variations',
                no_lumi_normalization=True,
                logy = True,
                no_ratiopad = True,
                override_ylabel = "$|\\delta G_{\\theta}|$"
            )
            plot_histogram(
                variable,
                cut,
                weight,
                delta_rho_datasets,
                plot_binning,
                extratext=extratext,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_rho_variations',
                no_lumi_normalization=True,
                logy = True,
                no_ratiopad = True,
                override_ylabel = "$|\\delta R_{\\theta}|$"
            )

        purity = np.diagonal(transfer0) / np.sum(transfer0, axis=0)
        stability = np.diagonal(transfer0) / np.sum(transfer0, axis=1)
        
        purity_dataset = ValNoCovDataset(
            key=f'{base_prefix}_purity',
            color=None,
            label='Purity',
            data=purity,
            binning=self.binning.genbinning,
            isMC=True,
        )
        stability_dataset = ValNoCovDataset(
            key=f'{base_prefix}_stability',
            color=None,
            label='Stability',
            data=stability,
            binning=self.binning.recobinning,
            isMC=True,
        )

        plot_histogram(
            variable,
            cut,
            weight,
            purity_dataset,
            plot_binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_purity',
            no_lumi_normalization=True,
            no_ratiopad = True,
        )
        plot_histogram(
            variable,
            cut,
            weight,
            stability_dataset,
            plot_binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_stability',
            no_lumi_normalization=True,
            no_ratiopad = True,
        )
        
        if detailed:
            gen_edges = self.binning.genbinning.edges
            reco_edges = self.binning.recobinning.edges

            ptgen_edges = gen_edges['Jpt_gen']
            ptreco_edges = reco_edges['Jpt_reco']

            Rgen_edges = gen_edges['R_gen'] 
            Rreco_edges = reco_edges['R_reco']

            for ipt in range(len(ptgen_edges) - 1):
                for iR in range(len(Rgen_edges) - 1):
                    gencut_i = SliceOperation(
                        {
                            'Jpt_gen' : (ptgen_edges[ipt], ptgen_edges[ipt+1]),
                            'R_gen' : (Rgen_edges[iR], Rgen_edges[iR+1])
                        },
                        []
                    )

                    plot_histogram(
                        variable,
                        gencut_i,
                        weight,
                        purity_dataset,
                        plot_binning,
                        extratext=extratext,
                        output_folder=output_folder,
                        override_filename=f'{base_prefix}_purity_ptgen_{ipt}_Rgen_{iR}',
                        no_lumi_normalization=True,
                        no_ratiopad = True,
                    )
            
            for ipt in range(len(ptreco_edges) - 1):
                for iR in range(len(Rreco_edges) - 1):
                    recocut_i = SliceOperation(
                        {
                            'Jpt_reco' : (ptreco_edges[ipt], ptreco_edges[ipt+1]),
                            'R_reco' : (Rreco_edges[iR], Rreco_edges[iR+1])
                        },
                        []
                    )

                    plot_histogram(
                        variable,
                        recocut_i,
                        weight,
                        stability_dataset,
                        plot_binning,
                        extratext=extratext,
                        output_folder=output_folder,
                        override_filename=f'{base_prefix}_stability_ptreco_{ipt}_Rreco_{iR}',
                        no_lumi_normalization=True,
                        no_ratiopad = True,
                    )

        t0dset = TransferMatrixDataset(
            key = f'{base_prefix}_transfer0',
            color = None,
            label = 'Transfer Matrix',
            data = transfer0,
            binning = self.binning,
            isMC = True,
        )

        draw_matrix(
             variable,
             cut,
             t0dset, # type: ignore
             plot_binning,
             extratext=extratext,
             sym=False,
             logc=True, 
             output_folder=output_folder,
             override_filename=f'{base_prefix}_transfer_matrix_nominal',
             override_cbarlabel = "Transfer matrix"
        )

        if nuisances:
            for i in range(self._nTransferSyst):
                dT = transfer_var[i]
                dT_dset = TransferMatrixDataset(
                    key = f'{base_prefix}_transfer_variation_{i}',
                    color = None,
                    label = self.nuisance_names[self._transferVarIndices[i]],
                    data = np.abs(dT),
                    binning = self.binning,
                    isMC = True,
                )
                if extratext is not None:
                    extratext_i = self.nuisance_names[self._transferVarIndices[i]] + '\n' + extratext
                else:
                    extratext_i = self.nuisance_names[self._transferVarIndices[i]]
                draw_matrix(
                    variable,
                    cut,
                    t0dset, # type: ignore
                    plot_binning,
                    extratext=extratext_i,
                    sym=True,
                    logc=True, 
                    output_folder=output_folder,
                    override_filename=f'{base_prefix}_transfer_variation_%s' % self.nuisance_names[self._transferVarIndices[i]],
                    override_cbarlabel = "$\\delta T_{\\theta}$ [%s]" % dT_dset.label
                )

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
    def from_dataset(cls, cfg : detectormodelspec) -> "DetectorModel":
        t0, gamma0, rho0 = get_model_matrices(
            cfg['dset'],
            {
                'objsyst' : 'nominal',
                'wtsyst' : 'nominal'
            }
        )

        gammaVariations = []
        rhoVariations = []
        transferVariations = []
        transferVarIndices = []
        nuisance_names = []

        for syst in cfg['systematics']:
            if 'label' in syst and syst['label'] is not None:
                nuisance_names.append(syst['label'])
            else:
                nuisance_names.append(syst['name'])

            if syst['onesided']:
                t_up, gamma_up, rho_up = get_model_matrices(
                    cfg['dset'],
                    hist_from_syst(syst, None),
                )
                dT = t_up - t0
                dGamma = gamma_up - gamma0
                dRho = rho_up - rho0
            else:
                t_up, gamma_up, rho_up = get_model_matrices(
                    cfg['dset'] ,
                    hist_from_syst(syst, "Up")
                )
                t_dn, gamma_dn, rho_dn = get_model_matrices(
                    cfg['dset'],
                    hist_from_syst(syst, "Down")
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

        binning = get_transfer_binning(cfg['dset'])

        # transform the binning s.t. all reco axes are named "*_reco"
        # and all gen axes are named "*_gen"
        for recoaxis in binning.recobinning.axis_names:
            if not recoaxis.endswith('_reco'):
                binning.recobinning.rename_axis(recoaxis, recoaxis + '_reco')
        for genaxis in binning.genbinning.axis_names:
            if not genaxis.endswith('_gen'):
                binning.genbinning.rename_axis(genaxis, genaxis + '_gen')

        return cls(t0, gamma0, rho0, 
                   transferVariations, transferVarIndices,
                   gammaVariations, rhoVariations,
                   binning, nuisance_names)
    
    @classmethod
    def from_disk(cls, path : str) -> "DetectorModel":
        arrays = {}
        for arrname in cls._arrays:
            with open(os.path.join(path, f'{arrname}.npy'), 'rb') as f:
                arrays[arrname] = np.load(f)
        arrays['binning'] = ArbitraryGenRecoBinning()
        arrays['binning'].load_from_file(os.path.join(path, 'binning.json'))
        arrays['nuisance_names'] = []
        with open(os.path.join(path, 'nuisance_names.txt'), 'r') as f:
            for line in f:
                arrays['nuisance_names'].append(line.strip())
        return cls(**arrays)
    
    def dump_to_disk(self, where : str):
        os.makedirs(where, exist_ok=True)
        for arrname in self._arrays:
            arr = getattr(self, f'_{arrname}')
            with open(os.path.join(where, f'{arrname}.npy'), 'wb') as f:
                np.save(f, arr)
        self._binning.dump_to_file(os.path.join(where, 'binning.json'))
        with open(os.path.join(where, 'nuisance_names.txt'), 'w') as f:
            for name in self._nuisance_names:
                f.write(name + '\n')

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