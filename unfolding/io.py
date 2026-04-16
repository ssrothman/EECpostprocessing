from typing import overload, override

from unfolding.specs import dsspec, whichsystspec
import numpy as np
from general.datasets.datasets import lookup_count, lookup_dataset, lookup_skim_path
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
import json
from typing import Literal

@overload
def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : str,
              read_cov : Literal[True]) -> tuple[np.ndarray, np.ndarray, ArbitraryBinning]:
    ...
@overload
def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : str,
              read_cov : Literal[False]) -> tuple[np.ndarray, ArbitraryBinning]:
    ...

def read_hist(dset : dsspec, 
              hist : whichsystspec,
              whichhist : str,
              read_cov : bool):
    
    if dset['isStack']:
        stackcfg = lookup_dataset(
            'stacks',
            dset['dataset']
        )

        vals = []
        covs = []
        binnings = []

        for subdset in stackcfg['dsets']:
            subcfg = dset.copy()
            subcfg.update({
                'dataset' : subdset,
                'isStack' : False
            })
            subH = read_hist(
                subcfg,
                hist,
                whichhist=whichhist,
                read_cov=read_cov
            )
            vals.append(subH[0])
            binnings.append(subH[-1])

            if read_cov:
                covs.append(subH[1])

        for substack in stackcfg['stacks']:
            subcfg = dset.copy()
            subcfg.update({
                'dataset' : substack,
                'isStack' : True
            })
            subH = read_hist(
                subcfg,
                hist,
                whichhist=whichhist,
                read_cov=read_cov
            )
            vals.append(subH[0])
            binnings.append(subH[-1])

            if read_cov:
                covs.append(subH[1])
        
        # check that all binnings are the same
        for b in binnings[1:]:
            if b != binnings[0]:
                raise ValueError("Binnings of subdatasets do not match in stack %s" % dset['dataset'])
            
        val = vals[0]
        for v in vals[1:]:
            val += v

        if not read_cov:
            return val, binnings[0]
        else:
            cov = covs[0]
            for c in covs[1:]:
                cov += c
            return val, cov, binnings[0]
        
    else:
        fs, skimpath = lookup_skim_path(
            dset['location'],
            dset['config_suite'],
            dset['runtag'],
            dset['dataset'], 
            hist['objsyst'],
            dset['what'] + '_' + whichhist
        )
        
        valspath = skimpath + '_BINNED_%s' % hist['wtsyst']
        covpath = skimpath + '_BINNED_covmat_%s' % hist['wtsyst']
        
        if dset['statN'] > 0:
            valspath += '_%dstat%d' % (dset['statN'], dset['statK'])
            covpath += '_%dstat%d' % (dset['statN'], dset['statK'])

        valspath += '.npy'
        covpath += '.npy'

        binningpath = skimpath + '_bincfg.json'

        print("Reading", valspath, "...")
        with fs.open(valspath, 'rb') as f:
            values = np.load(f)

        if whichhist == 'transfer':
            binning = ArbitraryGenRecoBinning()
        else:
            binning = ArbitraryBinning()

        with fs.open(binningpath, 'r') as f:
            binning.from_dict(json.load(f))

        if dset['isMC']:
            xsec = lookup_dataset(
                dset['runtag'],
                dset['dataset']
            )['xsec']
            count = lookup_count(
                dset['location'],
                dset['config_suite'],
                dset['runtag'],
                dset['dataset'],
                'nominal'
            )

            wt = 1000 * xsec * dset['target_lumi'] / count 
            print("\txsec weight:", wt)
            values *= wt
        else:
            wt = 1.0

        if not read_cov:
            return values, binning
        
        else:
            with fs.open(covpath, 'rb') as f:
                covmat = np.load(f)

            if dset['isMC']:
                covmat *= wt * wt

            return values, covmat, binning

