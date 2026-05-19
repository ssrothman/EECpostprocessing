from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
from simonpy.stats_v2 import smart_inverse, smart_sqrt
from typing import Any, Literal, Sequence, TypedDict
import numpy as np
import os
import json
from general.fslookup.skim_path import lookup_skim_path
from general.datasets.datasets import lookup_count, lookup_dataset
from unfolding.io import read_hist
from unfolding.specs import NuisanceTreatment, dsspec, whichsystspec, histspec

try:
    import hist
    from networkx import find_minimal_d_separator
    import simonplot as splt
    from simonplot import plot_histogram, draw_matrix
    from simonplot.binning import PrebinnedBinning
    from simonplot.cut import NoopOperation
    from simonplot.cut.PrebinnedCut import SliceOperation
    from simonplot.drivers import draw_radial_histogram
    from simonplot.plottables.PlotStuff import BoxSpec
    from simonplot.plottables.PrebinnedDatasets import TransferMatrixDataset, ValCovPairDataset, CovNoValDataset, ValNoCovDataset
    from simonplot.variable import BasicPrebinnedVariable, ConstantVariable, CorrelationFromCovariance
    from simonplot.variable.PrebinnedVariable import DivideOutProfile, NormalizePerBlock, WithJacobian
    _SIMONPLOT_AVAILABLE = True
except Exception:
    _SIMONPLOT_AVAILABLE = False

def get_genreco_name(axnames, target):
    if target+'_gen' in axnames:
        return target+'_gen'
    elif target+'_reco' in axnames:
        return target+'_reco'
    elif target in axnames:
        return target
    else:
        raise ValueError("No axis found matching %s_gen, %s_reco, or %s" % (target, target, target))

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

    def _as_numpy(self, values) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values

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
    def values(self) -> Any:
        return self._values 
    
    @property
    def covmat(self) -> Any:
        return self._covmat
    
    @property
    def invcov(self) -> Any:
        if self._invcov is None:
            self.compute_invcov()
        assert self._invcov is not None, "This should be impossible!"
        return self._invcov
    
    @property
    def L(self) -> np.ndarray:
        if self._L is None:
            self.compute_sqrt()
        assert self._L is not None, "This should be impossible!"
        return self._L
    
    @property
    def Linv(self) -> np.ndarray:
        if self._Linv is None:
            self.compute_sqrt()
        assert self._Linv is not None, "This should be impossible!"
        return self._Linv

    @property
    def eigvals(self) -> np.ndarray:
        if self._eigvals is None:
            self.compute_invcov()
        assert self._eigvals is not None, "This should be impossible!"
        return self._eigvals

    @property
    def eigvecs(self) -> np.ndarray:
        if self._eigvecs is None:
            self.compute_invcov()
        assert self._eigvecs is not None, "This should be impossible!"
        return self._eigvecs

    @property
    def binning(self) -> ArbitraryBinning:
        return self._binning

    @property
    def device(self) -> str:
        return self._device

    def chi2(self, other : "Histogram") -> float:
        diff : np.ndarray = self._as_numpy(self.values) - self._as_numpy(other.values)
        cov : np.ndarray = self._as_numpy(self.covmat) + self._as_numpy(other.covmat)
        invcov : np.ndarray = smart_inverse(cov, False)
        result : float = np.linalg.multi_dot([diff, invcov, diff]).item()
        print(result)
        assert type(result) == float, "Chi2 should be a single number"
        return result

    @classmethod
    def from_dataset(cls, cfg: dsspec, histcfg: whichsystspec, whichhist : Literal['totalReco', 'totalGen', 'unmatchedReco', 'unmatchedGen', 'untransferedReco', 'untransferedGen'], rebinning : str | dict | None) -> "Histogram":
        values, covmat, binning = read_hist(
            cfg,
            histcfg,
            whichhist,
            read_cov=True
        )

        result = cls(values, covmat, binning)
        if rebinning is not None:
            result.rebin(rebinning)
        return result

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

    # NB this is an in-place operation
    def rebin(self, rebinning_spec : dict | str):
        self.to('numpy')
        assert(isinstance(self._values, np.ndarray))
        assert(isinstance(self._covmat, np.ndarray))

        self._values, _ = self._binning.rebin(self._values, rebinning_spec)
        self._covmat, self._binning = self._binning.rebin_cov2d(self._covmat, rebinning_spec)

        # I'm not sure if its possible to propagate the inverse covariance, eigendecomposition, or sqrt
        # so for now to be conservative we just invalidate them
        self._invcov = None
        self._L = None
        self._Linv = None
        self._eigvals = None
        self._eigvecs = None

    @classmethod
    def compare(
        cls,
        hist_l : Sequence['Histogram'],
        labels_l : Sequence[str],
        output_folder : str | None = None,
        extratext : str | None = None
    ) -> None:
        
        variable = BasicPrebinnedVariable()
        cut = NoopOperation()
        weight = ConstantVariable(1.0)
        binning = PrebinnedBinning()

        val_datasets_l = []
        err_datasets_l = []
        for i, hist in enumerate(hist_l):
            val_datasets_l.append(ValCovPairDataset(
                key=f'hist_{i}',
                color=None,
                label=labels_l[i],
                data=(hist.values, hist.covmat),
                binning=hist._binning,
                isMC=True,
            ))
            err_datasets_l.append(ValCovPairDataset(
                key=f'hist_{i}_err',
                color=None,
                label=labels_l[i],
                data=(np.sqrt(np.diag(hist.covmat)), np.zeros_like(hist.covmat)),
                binning=hist._binning,
                isMC=True,
            ))
            
        
        plot_histogram(
            variable,
            cut,
            weight,
            val_datasets_l,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            no_lumi_normalization=True,
            override_filename='values_comparison',
        )
        plot_histogram(
            variable,
            cut,
            weight,
            err_datasets_l,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            no_lumi_normalization=True,
            override_filename='errs_comparison',
            override_ylabel='Error [sqrt diag(cov)]'
        )

    def plot(
        self,
        output_folder: str | None = None,
        extratext: str | None = None,
        covmats : bool = True,
        prettymatrices : bool = False,
        projected_r_c_1D : bool = False,
        projected_c_1D : bool = False,
        transform : Literal['none', 'shapes', 'angular_modulation'] = 'none'
    ) -> None:
        
        if transform not in ['none', 'shapes', 'angular_modulation']:
            raise ValueError("Invalid transform option: %s" % transform)

        axnames = self.binning.axis_names

        ptname = get_genreco_name(axnames, 'Jpt')            
        Rname = get_genreco_name(axnames, 'R')
        rname = get_genreco_name(axnames, 'r')
        cname = get_genreco_name(axnames, 'c')

        plotbinning = self.binning.copy()
        plotbinning.clip_flow_bins(ptname, negativeinf=30.0) # pT underflow actually starts at 30 GeV

        if self.invcov is None:
            self.compute_invcov()
        
        values = self._as_numpy(self.values)
        covmat = self._as_numpy(self.covmat)
        invcov = self._as_numpy(self.invcov)
        invcov_cov = invcov @ covmat

        if transform == 'shapes':
            basevariable = NormalizePerBlock(
                BasicPrebinnedVariable(),
                [ptname, Rname]
            )
        elif transform == 'angular_modulation':
            basevariable = DivideOutProfile(
                BasicPrebinnedVariable(),
                [ptname, Rname, rname]
            )
        else:
            basevariable = BasicPrebinnedVariable()

        if transform != 'angular_modulation':
            basevariable = WithJacobian(
                basevariable,
                [rname, cname],
                radial_coords=[rname]
            )

        cut = NoopOperation()
        weight = ConstantVariable(1.0)
        binning = PrebinnedBinning()

        base_prefix = 'histogram'
        if transform != 'none':
            base_prefix += f'_{transform}'

        values_dataset = ValCovPairDataset(
            key=f'{base_prefix}_values',
            color=None,
            label=None,
            data=(values, covmat),
            binning=plotbinning,
            isMC=True,
        )
        cov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_cov',
            color=None,
            label=None,
            data=(values, covmat),
            binning=plotbinning,
            isMC=True,
        )
        invcov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_invcov',
            color=None,
            label=None,
            data=(values, invcov),
            binning=plotbinning,
            isMC=True,
        )
        invcov_cov_dataset = ValCovPairDataset(
            key=f'{base_prefix}_invcov_times_cov',
            color=None,
            label=None,
            data=(values, invcov_cov),
            binning=plotbinning,
            isMC=True,
        )

        if projected_r_c_1D:
            assert(isinstance(basevariable,splt.variable.WithJacobian))

            projcut = splt.cut.ProjectionOperation(
                [rname, cname]
            )
            plot_histogram(
                basevariable._var,  # type: ignore
                projcut,
                weight,
                values_dataset,
                binning,
                extratext=extratext,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_projected(r,c)_1D_histogram',
                no_lumi_normalization=True,
            )

        if projected_c_1D:
            projc_variable = splt.variable.BasicPrebinnedVariable()
            if transform == 'angular_modulation':
                raise ValueError("Angular modulation not supported when projecting out angular coordinate")
            elif transform == 'shapes':
                projc_variable = NormalizePerBlock(projc_variable, [])
            elif transform == 'none':
                pass
            else:
                raise ValueError("Invalid transform option: %s" % transform)

            projc_variable = splt.variable.WithJacobian(
                projc_variable,
                [rname],
                [rname]
            )

            Redges = plotbinning.axis_edges(Rname)
            ptedges = plotbinning.axis_edges(ptname)
            for iR in range(len(Redges)-1):
                projcuts = []
                labels = []
                for ipt in range(1, len(ptedges)-1): # skip pT underflow bin
                    projcuts.append(splt.cut.ProjectAndSliceOperation(
                        [cname],
                        {
                            Rname: (Redges[iR], Redges[iR+1]),
                            ptname : (ptedges[ipt], ptedges[ipt+1])
                        },
                        []
                    ))
                    if ptedges[ipt+1] == np.inf:
                        labels.append("$%d < p_T$ [GeV]" % (ptedges[ipt]))
                    else:
                        labels.append("$%d < p_T $ [GeV] $ < %d$" % (ptedges[ipt], ptedges[ipt+1]))

                extraextratext = '$%0.1f < R < %0.1f$' % (Redges[iR], Redges[iR+1])
                if extratext is not None:
                    itext = extraextratext + '\n' + extratext
                else:
                    itext = extraextratext
                    
                plot_histogram(
                    projc_variable,
                    projcuts,
                    weight,
                    values_dataset,
                    binning,
                    labels,
                    extratext=itext,
                    output_folder=output_folder,
                    override_filename=f'{base_prefix}_projected(c)_1D_histogram_R{iR}',
                    no_lumi_normalization=True,
                    no_ratiopad=True,
                    logx=True,
                    logy=True
                )
            
        plot_histogram(
            basevariable,
            cut,
            weight,
            values_dataset,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=f'{base_prefix}_values_histogram',
            no_lumi_normalization=True,
        )

        if covmats:
            draw_matrix(
                basevariable,
                cut,
                cov_dataset, # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_covariance_matrix',
            )

            draw_matrix(
                basevariable,
                cut,
                invcov_dataset, # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_inverse_covariance_matrix',
            )

            draw_matrix(
                basevariable,
                cut,
                invcov_cov_dataset, # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=False,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_inverse_covariance_times_covariance_matrix',
            )

        corr_variable = CorrelationFromCovariance(basevariable)

        if covmats:
            draw_matrix(
                corr_variable,
                cut,
                cov_dataset, # type: ignore
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
                invcov_dataset, # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=False,
                output_folder=output_folder,
                override_filename=f'{base_prefix}_inverse_covariance_correlation_matrix',
            )

        if prettymatrices:
            axnames = self.binning.axis_names

            ptname = get_genreco_name(axnames, 'Jpt')            
            Rname = get_genreco_name(axnames, 'R')
            rname = get_genreco_name(axnames, 'r')  
            cname = get_genreco_name(axnames, 'c')          

            ptedges = self.binning.axis_edges(ptname)
            Redges = self.binning.axis_edges(Rname)

            plotbinning.clip_flow_bins(rname, positiveinf=3.0) #  clip r overflow bin for triangle


            if transform == 'angular_modulation':
                # the slicing breaks the DivideOutProfile
                assert(isinstance(basevariable, DivideOutProfile))
                basevariable._axes = [rname]

            for ipt in range(len(ptedges)-1):
                for iR in range(len(Redges)-1):
                    cuti = SliceOperation(
                        {
                            ptname: (ptedges[ipt], ptedges[ipt+1]),
                            Rname: (Redges[iR], Redges[iR+1])
                        },
                        []
                    )
                    # if 'angular_modulation': want logc=False and sym=True
                    # else want logc=True and sym=False
                    draw_radial_histogram(
                        basevariable,
                        cuti,
                        values_dataset, # type: ignore
                        binning,
                        extratext = extratext,
                        logc = transform != 'angular_modulation',
                        sym = transform == 'angular_modulation',
                        output_folder = output_folder,
                        override_filename = '%s_radial2D_pt%d_R%d' % (base_prefix, ipt, iR)
                    )


    def to_torch(self):
        import torch
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
            import torch

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

            import torch

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

        import torch

        assert isinstance(self._values, torch.Tensor)
        assert isinstance(self._covmat, torch.Tensor)
        assert isinstance(self._invcov, torch.Tensor)

        self._values = self._values.detach()
        self._covmat = self._covmat.detach()
        self._invcov = self._invcov.detach()

        return self

class UnfoldedHistogram:
    def __init__(self, x : np.ndarray, baseline : np.ndarray, 
                 hessian : np.ndarray,
                 binning : ArbitraryBinning,
                 nuisance_names : list[str],
                 invhess : np.ndarray | None = None,
                 eigvals : np.ndarray | None = None,
                 eigvecs : np.ndarray | None = None,
                 L : np.ndarray | None = None,
                 Linv : np.ndarray | None = None):
        
        self._x = x
        self._baseline = baseline
        self._hessian = hessian
        self._binning = binning
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._L = L
        self._Linv = Linv
        self._invhess = invhess
        self._nuisance_names = nuisance_names

    @property
    def nuisance_names(self):
        return self._nuisance_names

    @property
    def x(self):
        return self._x

    @property
    def baseline(self):
        return self._baseline

    @property
    def hessian(self):
        return self._hessian

    @property
    def binning(self):
        return self._binning
    
    @property
    def invhess(self):
        if self._invhess is None:
            self.compute_invhess()
        return self._invhess
    
    @property
    def L(self):
        if self._L is None:
            self.compute_sqrt()
        return self._L
    
    @property
    def Linv(self):
        if self._Linv is None:
            self.compute_sqrt()
        return self._Linv
    
    @property
    def eigvals(self):
        if self._eigvals is None:
            self.compute_invhess()
        return self._eigvals
    
    @property
    def eigvecs(self):
        if self._eigvecs is None:
            self.compute_invhess()
        return self._eigvecs

    def compute_invhess(self):
        print("Inverting Hessian matrix...")
        assert isinstance(self._hessian, np.ndarray)
        self._invhess, self._eigvals, self._eigvecs = smart_inverse(self._hessian, True)

    def compute_sqrt(self):
        print("Computing sqrt of Hessian matrix...")
        assert isinstance(self._hessian, np.ndarray)
        self._L, self._Linv = smart_sqrt(self._hessian)

    def to_basic_histogram(self, nuisance_treatment : NuisanceTreatment):
        from simonpy.stats import marginalize, condition

        if nuisance_treatment.num != self._x.shape[0] - self._baseline.shape[0]:
            raise ValueError("Number of nuisances does not match")
        
        x = self._x.copy()
        invhess = self.invhess
        assert invhess is not None
        invhess = invhess.copy()

        Nbeta = self._baseline.shape[0]

        for treatment in nuisance_treatment.treated_nuisances:
            if treatment['fixed']:
                print("Fixing nuisance", self._nuisance_names[treatment['index']], "to value", treatment['value'])
                x, invhess = condition(x, invhess, Nbeta+treatment['index'], Nbeta+treatment['index']+1, treatment['value'])
            else:
                print("Marginalizing over nuisance", self._nuisance_names[treatment['index']])      
                x, invhess = marginalize(x, invhess, Nbeta+treatment['index'], Nbeta+treatment['index']+1)
    

        # apply multiplication by baseline
        x *= self._baseline
        invhess = invhess * np.outer(self._baseline, self._baseline)

        return Histogram(x, invhess, self._binning)
        
    @classmethod 
    def from_minimization_result(cls, where:str) -> 'UnfoldedHistogram':
        # Load the results from a minimization
        with open(os.path.join(where, 'result', 'x.npy'), 'rb') as f:
            x = np.load(f)

        with open(os.path.join(where, 'result', 'hessian.npy'), 'rb') as f:
            hessian = np.load(f)

        with open(os.path.join(where, 'baseline.npy'), 'rb') as f:
            baseline = np.load(f)

        binning = ArbitraryBinning()
        binning.load_from_file(os.path.join(where, 'binning.json'))

        with open(os.path.join(where, 'nuisance_names.txt'), 'r') as f:
            nuisance_names = [line.strip() for line in f]

        return cls(x, baseline, hessian, binning=binning, nuisance_names=nuisance_names)

    @classmethod
    def from_disk(cls, where:str) -> 'UnfoldedHistogram':
        with open(os.path.join(where, 'x.npy'), 'rb') as f:
            x = np.load(f)
        with open(os.path.join(where, 'baseline.npy'), 'rb') as f:
            baseline = np.load(f)
        with open(os.path.join(where, 'hessian.npy'), 'rb') as f:
            hessian = np.load(f)
        binning = ArbitraryBinning()
        binning.load_from_file(os.path.join(where, 'binning.json'))
        with open(os.path.join(where, 'nuisance_names.txt'), 'r') as f:
            nuisance_names = [line.strip() for line in f]

        extras = {}

        if os.path.exists(os.path.join(where, 'invhess.npy')):
            with open(os.path.join(where, 'invhess.npy'), 'rb') as f:
                extras['invhess'] = np.load(f)
        if os.path.exists(os.path.join(where, 'eigvals.npy')):
            with open(os.path.join(where, 'eigvals.npy'), 'rb') as f:
                extras['eigvals'] = np.load(f)
        if os.path.exists(os.path.join(where, 'eigvecs.npy')):
            with open(os.path.join(where, 'eigvecs.npy'), 'rb') as f:
                extras['eigvecs'] = np.load(f)
        if os.path.exists(os.path.join(where, 'L.npy')):
            with open(os.path.join(where, 'L.npy'), 'rb') as f:
                extras['L'] = np.load(f)
        if os.path.exists(os.path.join(where, 'Linv.npy')):
            with open(os.path.join(where, 'Linv.npy'), 'rb') as f:
                extras['Linv'] = np.load(f)

        return cls(x, baseline, hessian, binning=binning, nuisance_names=nuisance_names, **extras)
    
    def dump_to_disk(self, where: str):
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, 'x.npy'), 'wb') as f:
            np.save(f, self._x)
        with open(os.path.join(where, 'baseline.npy'), 'wb') as f:
            np.save(f, self._baseline)
        with open(os.path.join(where, 'hessian.npy'), 'wb') as f:
            np.save(f, self._hessian)
        self._binning.dump_to_file(os.path.join(where, 'binning.json'))
        with open(os.path.join(where, 'nuisance_names.txt'), 'w') as f:
            for name in self._nuisance_names:
                f.write(name + '\n')

        if self._invhess is not None:
            with open(os.path.join(where, 'invhess.npy'), 'wb') as f:
                np.save(f, self._invhess)
        if self._eigvals is not None:
            with open(os.path.join(where, 'eigvals.npy'), 'wb') as f:
                np.save(f, self._eigvals)
        if self._eigvecs is not None:
            with open(os.path.join(where, 'eigvecs.npy'), 'wb') as f:
                np.save(f, self._eigvecs)
        if self._L is not None:
            with open(os.path.join(where, 'L.npy'), 'wb') as f:
                np.save(f, self._L)
        if self._Linv is not None:
            with open(os.path.join(where, 'Linv.npy'), 'wb') as f:
                np.save(f, self._Linv)

    def _plot_valonly_histogram(self,
                                output_folder : str | None, extratext : str | None, 
                                variable : Any, cut : Any, weight : Any, binning : Any,
                                data : list[np.ndarray], label : list[str], thebinning : Any,
                                override_filename : str, override_ylabel : str | None = None,
                                **kwargs):
        datasets = []
        for d, l in zip(data, label):
            datasets.append(ValNoCovDataset(
                key=l,
                color=None,
                label=l,
                data=d,
                binning=thebinning,
                isMC=True,
            ))
        plot_histogram(
            variable,
            cut,
            weight,
            datasets,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename=override_filename,
            no_lumi_normalization=True,
            override_ylabel=override_ylabel,
            no_ratiopad = True,
            **kwargs
        )


    def plot(self, output_folder: str | None = None, extratext: str | None = None, covmats : bool = False) -> None:
        if self.invhess is None:
            self.compute_invhess()
        
        assert(isinstance(self._x, np.ndarray))
        assert(isinstance(self.invhess, np.ndarray))

        xprof = self._x[:self.baseline.shape[0]] * self._baseline
        cov_prof = self.invhess[:self.baseline.shape[0], :self.baseline.shape[0]] * np.outer(self._baseline, self._baseline)

        profdataset = ValCovPairDataset(
            key='profiled',
            color=None,
            label='Unfolded [marginalized nuisances]',
            data=(xprof, cov_prof),
            binning=self._binning,
            isMC=True,
        )

        variable = BasicPrebinnedVariable()
        cut = NoopOperation()
        weight = ConstantVariable(1.0)
        binning = PrebinnedBinning()

        plot_histogram(
            variable,
            cut,
            weight,
            profdataset,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename='unfolded_profiled',
            no_lumi_normalization=True,
        )

        if covmats:
            draw_matrix(
                variable,
                cut,
                profdataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename='unfolded_profiled_covariance_matrix',
            )

        corr_variable = CorrelationFromCovariance(variable)

        if covmats:
            draw_matrix(
                corr_variable,
                cut,
                profdataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=False,
                output_folder=output_folder,
                override_filename='unfolded_profiled_correlation_matrix',
            )

        theta = self._x[self.baseline.shape[0]:]
        covtheta = self.invhess[self.baseline.shape[0]:, self.baseline.shape[0]:]
        testH = hist.Hist(
            hist.axis.Integer(0, len(self._nuisance_names), name='nuisance', underflow=False, overflow=False)
        )
        thetabinning = ArbitraryBinning()
        thetabinning.setup_from_histogram(testH)
        thetabinning.setup_label_lookup(
            {'nuisance' : {i : self._nuisance_names[i] for i in range(len(self._nuisance_names))}}
        )

        covtheta_dataset = ValCovPairDataset(
            key='nuisance_covariance',
            color=None,
            label="Posterior",
            data=(theta, covtheta),
            binning=thetabinning,
            isMC=True,
        )

        gray_box = BoxSpec(-0.1, -1, len(self._nuisance_names)+0.2, 2, facecolor='gray', alpha=0.5, zorder=0, label='Prior')

        plot_histogram(
            variable,
            cut,
            weight,
            covtheta_dataset,
            binning,
            extratext=extratext,
            output_folder=output_folder,
            override_filename='nuisance_pulls',
            no_lumi_normalization=True,
            extra_stuff=[gray_box],
            override_ylabel="Nuisance posteriors",
            logy=False
        )
        
        if covmats:
            draw_matrix(
                variable,
                cut,
                covtheta_dataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename='unfolded_nuisance_covariance_matrix',
            )
            draw_matrix(
                corr_variable,
                cut,
                covtheta_dataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=False,
                output_folder=output_folder,
                override_filename='unfolded_nuisance_correlation_matrix',
            )

        covbetatheta = self.invhess[:self.baseline.shape[0], self.baseline.shape[0]:] * self._baseline[:, None]

        thetabetabinning = ArbitraryGenRecoBinning()
        thetabetabinning.setup_from_binnings(self._binning, thetabinning)

        covbetatheta_dataset = TransferMatrixDataset(
            key='betatheta_covariance',
            color=None,
            label='Covariance between unfolded and nuisances',
            data=covbetatheta,
            binning=thetabetabinning,
            isMC=True,
        )
        thetaerrs = np.sqrt(np.diag(covtheta))
        betaerrs = np.sqrt(np.diag(cov_prof))
        covbetatheta_dataset.setup_custom_diagonals(thetaerrs, betaerrs)

        if covmats:
            draw_matrix(
                variable,
                cut,
                covbetatheta_dataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename='unfolded_betatheta_covariance_matrix',
            )
            draw_matrix(
                corr_variable,
                cut,
                covbetatheta_dataset,  # type: ignore
                binning,
                extratext=extratext,
                sym=True,
                logc=True,
                output_folder=output_folder,
                override_filename='unfolded_betatheta_correlation_matrix',
            )
        
        # time for nuisance contributions
        from simonpy.stats import condition

        xi = self._x.copy()
        covxi = self.invhess.copy()
        Nbeta = self._baseline.shape[0]
        Ntheta = len(self._nuisance_names)

        xi[:Nbeta] *= self._baseline
        covxi[:Nbeta, :] *= self._baseline[:, None]
        covxi[:, :Nbeta] *= self._baseline[None, :]

        # "what if we never had nuisance parameters at all?" -> just condition on them all being 0
        x_statonly = xi.copy()
        cov_statonly = covxi.copy()
        for i in reversed(list(range(Ntheta))):
            x_statonly, cov_statonly = condition(x_statonly, cov_statonly, Nbeta+i, Nbeta+i+1, 0)

        flux_statonly, shape_statonly, fluxbinning = self.binning.get_fluxes_shapes(x_statonly, ['Jpt_gen', 'R_gen'])
        covflux_statonly, covshape_statonly, covfluxshape_statonly = self.binning.get_fluxes_shapes_cov2d(
            flux_statonly, shape_statonly, cov_statonly, ['Jpt_gen', 'R_gen']
        )

        flux_total, shape_total, _ = self.binning.get_fluxes_shapes(xi[:Nbeta], ['Jpt_gen', 'R_gen'])
        covflux_total, covshape_total, covfluxshape_total = self.binning.get_fluxes_shapes_cov2d(
            flux_total, shape_total, covxi[:Nbeta, :Nbeta], ['Jpt_gen', 'R_gen']
        )

        # contribution from nuisance i is the difference between the statonly sollution 
        # and the solution with ONLY nuisance i not conditioned out
        val_contributions = []
        cov_contributions = []
        
        flux_contributions = []
        shape_contributions = []
        covflux_contributions = []
        covshape_contributions = []
        covfluxshape_contributions = []

        for i in range(Ntheta):
            x_nui = xi.copy()
            cov_nui = covxi.copy()

            for j in reversed(list(range(Ntheta))):
                if i == j:
                    continue
                x_nui, cov_nui = condition(x_nui, cov_nui, Nbeta+j, Nbeta+j+1, 0)

            val_contributions.append(x_nui[:Nbeta] - x_statonly[:Nbeta])
            cov_contributions.append(cov_nui[:Nbeta, :Nbeta] - cov_statonly[:Nbeta, :Nbeta])

            fluxi, shapei, _ = self.binning.get_fluxes_shapes(x_nui[:Nbeta], ['Jpt_gen', 'R_gen'])
            covfluxi, covshapei, covfluxshapei = self.binning.get_fluxes_shapes_cov2d(
                fluxi, shapei, cov_nui[:Nbeta, :Nbeta], ['Jpt_gen', 'R_gen']
            )
            flux_contributions.append(fluxi - flux_statonly)
            shape_contributions.append(shapei - shape_statonly)
            covflux_contributions.append(covfluxi - covflux_statonly)
            covshape_contributions.append(covshapei - covshape_statonly)
            covfluxshape_contributions.append(covfluxshapei - covfluxshape_statonly)

        # now we can plot the contributions
        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[x / x_statonly for x in val_contributions],
            label=self.nuisance_names,
            thebinning = self.binning,
            override_filename='nuisance_contributions',
            override_ylabel='Proportional shift in unfolded result from nuisance',
            logy=False
        )
        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[flux / flux_statonly for flux in flux_contributions],
            label=self.nuisance_names,
            thebinning = fluxbinning,
            override_filename='nuisance_flux_contributions',
            override_ylabel='Proportional shift in unfolded result from nuisance [flux per (pT, R) bin]',
            logy=False
        )
        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[shape / shape_statonly for shape in shape_contributions],
            label=self.nuisance_names,
            thebinning = self.binning,
            override_filename='nuisance_shape_contributions',
            override_ylabel='Proportional shift in unfolded result from nuisance [normalized per (pT, R) bin]',
            logy=False
        )

        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[np.sqrt(np.diag(cov))/x_statonly for cov in (cov_contributions + [cov_statonly, cov_prof])],
            label=self.nuisance_names + ['Stat', 'Total'],
            thebinning = self.binning,
            override_filename='nuisance_err_contributions',
            override_ylabel='$\\sigma/\\mu$ contribution from nuisance',
            logy=True
        )
        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[np.sqrt(np.diag(cov))/flux_statonly for cov in (covflux_contributions + [covflux_statonly, covflux_total])],
            label=self.nuisance_names + ['Stat', 'Total'],
            thebinning = fluxbinning,
            override_filename='nuisance_flux_err_contributions',
            override_ylabel='$\\sigma/\\mu$ contribution from nuisance [flux per (pT, R) bin]',
            logy=True
        )

        # due to floating point rounding, it can happen that some of the contributions are very slightly negative
        # it doesn't feel right to clip an uncertainty to zero
        # -> take abs()
        diagcov = [np.diag(cov) for cov in (covshape_contributions + [covshape_statonly, covshape_total])]
        diagcov = [np.abs(diag) for diag in diagcov]
        thedata = [np.sqrt(diag)/shape_statonly for diag in diagcov]

        self._plot_valonly_histogram(
            output_folder, extratext, variable, cut, weight, binning,
            data=[np.sqrt(cov)/shape_statonly for cov in diagcov],
            label=self.nuisance_names + ['Stat', 'Total'],
            thebinning = self.binning,
            override_filename='nuisance_shape_err_contributions',
            override_ylabel='$\\sigma/\\mu$ contribution from nuisance [normalized per (pT, R) bin]',
            logy=True
        )