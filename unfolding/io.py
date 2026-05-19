from typing import overload
try:
    from typing import override
except ImportError:
    def override(f): return f

from unfolding.specs import dsspec, whichsystspec
import numpy as np
from general.datasets.datasets import lookup_count, lookup_dataset, lookup_skim_path, lookup_xsec
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
import json
from typing import Literal
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path

@overload
def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : Literal['transfer', 'totalReco', 'totalGen', 'unmatchedReco', 'unmatchedGen', 'untransferedReco', 'untransferedGen'],
              read_cov : Literal[True]) -> tuple[np.ndarray, np.ndarray, ArbitraryBinning]:
    ...

@overload
def read_hist(dset : dsspec,
              hist : whichsystspec,
              whichhist : Literal['transfer'],
              read_cov : Literal[False]) -> tuple[np.ndarray, ArbitraryGenRecoBinning]:
    ...

@overload
def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : Literal['totalReco', 'totalGen', 'unmatchedReco', 'unmatchedGen', 'untransferedReco', 'untransferedGen'],
              read_cov : Literal[False]) -> tuple[np.ndarray, ArbitraryBinning]:
    ...

def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : Literal['transfer', 'totalReco', 'totalGen', 'unmatchedReco', 'unmatchedGen', 'untransferedReco', 'untransferedGen'],
              read_cov : bool):
    
    table = dset['what'] + '_' + whichhist

    fs, valpath = get_hist_path(
        dset['location'],
        dset['config_suite'],
        dset['runtag'],
        dset['dataset'],
        hist['objsyst'],
        hist['wtsyst'],
        table,
        False,
        dset['statN'],
        dset['statK']
    )

    _, bincfgpath = get_hist_bincfg_path(
        dset['location'],
        dset['config_suite'],
        dset['runtag'],
        dset['dataset'],
        hist['objsyst'],
        table
    )

    with fs.open(valpath, 'rb') as f:
        values = np.load(f)

    if dset['target_lumi'] > 0 and dset['isMC']:
        # reweight by target lumi
        if dset['isStack']: # stacks are already scaled to a luminosity of 1/fb
            wt = dset['target_lumi']
        else:
            xsec = lookup_xsec(dset['runtag'], dset['dataset'])
            count = lookup_count(dset['runtag'], dset['dataset'])
            wt = 1000 * xsec * dset['target_lumi'] / count
    else:
        wt = 1.0

    if dset['statN'] > 0:
        if isinstance(dset['statK'], int):
            wt *= dset['statN']
        else:
            wt *= dset['statN'] / len(dset['statK'])
    
    values *= wt

    if read_cov and whichhist == 'transfer':
        raise ValueError("Covariance matrices not supported for transfer histograms")
    elif whichhist == 'transfer':
        binning = ArbitraryGenRecoBinning()
        with fs.open(bincfgpath, 'r') as f:
            binning.from_dict(json.load(f))
        return values, binning
    else:
        binning = ArbitraryBinning()
        with fs.open(bincfgpath, 'r') as f:
            binning.from_dict(json.load(f))

        if read_cov:
            _, covpath = get_hist_path(
                dset['location'],
                dset['config_suite'],
                dset['runtag'],
                dset['dataset'],
                hist['objsyst'],
                hist['wtsyst'],
                table,
                True,
                dset['statN'],
                dset['statK']
            )
            with fs.open(covpath, 'rb') as f:
                covmat = np.load(f)
            covmat *= wt * wt
            return values, covmat, binning
        else:
            return values, binning