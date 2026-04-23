from general.datasets import datasets
from general.datasets.datasets import lookup_count, lookup_dataset, cfg
from general.fslookup.skim_path import lookup_skim_path
from simonplot.plottables import ParquetDataset
from simonplot.plottables.Datasets import DatasetStack
import os
import json
import numpy as np

from simonplot.plottables.PrebinnedDatasets import PrebinnedRootHistogramDataset, ValCovPairDataset
from simonpy.AbitraryBinning import ArbitraryBinning

def build_prebinned_dataset_stack(configsuite : str,
                                  runtag : str,
                                  dataset : str,
                                  objsyst : str,
                                  wtsyst : str,
                                  table : str,
                                  location : str = 'local-submit',
                                  no_count : bool = False,
                                  label_override : str | None = None,
                                  color_override : str | None = None,
                                  extra_key : str | None = None,
                                  nocov : bool = False,
                                  statN : int = -1,
                                  statK : int = -1,
                                  showStack : bool = True) -> DatasetStack:
    
    stackcfg = cfg['stacks'][dataset]
    dsets = []
    for dset in stackcfg['dsets']:
        dsets.append(load_prebinned_dataset(
            configsuite,
            runtag, 
            dset, 
            objsyst, 
            wtsyst,
            table,
            location,
            nocov = nocov,
            statN = statN,
            statK = statK,
        ))
    for dset in stackcfg['stacks']:
        dsets.append(build_prebinned_dataset_stack(
            configsuite,
            runtag,
            dset,
            objsyst,
            wtsyst,
            table,
            location,
            no_count=no_count,
            nocov = nocov,
            statN = statN,
            statK = statK,
        ))

    thekey = dataset
    if extra_key is not None:
        thekey += '-' + extra_key

    return DatasetStack(
        key = thekey,
        color = color_override if color_override is not None else stackcfg['color'],
        label = label_override if label_override is not None else stackcfg['label'],
        datasets = dsets,
        showstack = showStack
    )



def build_pq_dataset_stack(configsuite : str,
                           runtag : str,
                           dataset : str,
                           objsyst : str,
                           table : str,
                           location : str = 'local-submit',
                           no_count : bool = False,
                           label_override : str | None = None,
                           color_override : str | None = None,
                           extra_key : str | None = None,
                           showStack : bool = True) -> DatasetStack:
    stackcfg = cfg['stacks'][dataset]
    dsets = []
    for dset in stackcfg['dsets']:
        dsets.append(build_pq_dataset(
            configsuite,
            runtag, 
            dset, 
            objsyst, 
            table,
            location,
            no_count = no_count
        ))
    for dset in stackcfg['stacks']:
        dsets.append(build_pq_dataset_stack(
            configsuite,
            runtag,
            dset,
            objsyst,
            table,
            location,
            no_count = no_count
        ))

    thekey = dataset
    if extra_key is not None:
        thekey += '-' + extra_key

    return DatasetStack(
        key = thekey,
        color = color_override if color_override is not None else stackcfg['color'],
        label = label_override if label_override is not None else stackcfg['label'],
        datasets = dsets,
        showstack = showStack
    )

def load_prebinned_root_histogram(path : str,
                                  runtag : str,
                                  dataset : str,
                                  
                                  label_override : str | None = None,
                                  color_override : str | None = None,

                                  extra_key : str | None = None,

                                  configsuite : str = 'ignored',
                                  objsyst : str = 'ignored',
                                  table : str = 'ignored',
                                  location : str = 'ignored',):
    dsetcfg = lookup_dataset(runtag, dataset)

    color = color_override if color_override is not None else dsetcfg['color']
    label = label_override if label_override is not None else dsetcfg['label']

    thekey = dataset
    if extra_key is not None:
        thekey += '-' + extra_key

    theds = PrebinnedRootHistogramDataset(
        key = thekey,
        path = path,
        color = color,
        label = label,
        isMC = 'xsec' in dsetcfg
    )
    if 'xsec' in dsetcfg:
        theds.set_xsec(dsetcfg['xsec'])
    elif 'lumi' in dsetcfg:
        theds.set_lumi(dsetcfg['lumi'])

    return theds

def load_prebinned_dataset(configsuite : str,
                           runtag : str,
                           dataset :str,
                           objsyst : str,
                           wtsyst : str,
                           table : str,
                           location : str = 'local-submit',
                           label_override : str | None = None,
                           color_override : str | None = None,
                           extra_key : str | None = None,
                           statN : int = -1,
                           statK : int = -1,
                           nocov : bool = False) -> ValCovPairDataset:
    
    dsetcfg = lookup_dataset(runtag, dataset)
    
    fs, tablepath = lookup_skim_path(
        location,
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )

    valpath = tablepath + '_BINNED_%s' % wtsyst
    covpath = tablepath + '_BINNED_covmat_%s' % wtsyst

    if statN > 0:
        valpath += '_%dstat%d' % (statN, statK)
        covpath += '_%dstat%d' % (statN, statK)

    valpath += '.npy'
    covpath += '.npy'

    binningpath = tablepath + '_bincfg.json'

    vals = np.load(fs.open(valpath, 'rb'))
    if nocov:
        covmat = np.zeros((vals.shape[0], vals.shape[0]))
    else:
        covmat = np.load(fs.open(covpath, 'rb'))
        
    with fs.open(binningpath, 'r') as f:
        bincfg = json.load(f)
    
    binning = ArbitraryBinning()
    binning.from_dict(bincfg)

    thekey = dataset
    if extra_key is not None:
        thekey += '-' + extra_key

    theds = ValCovPairDataset(
        key = thekey,
        color = color_override if color_override is not None else dsetcfg['color'],
        label = label_override if label_override is not None else dsetcfg['label'],
        data = (vals, covmat),
        binning = binning,
        isMC = 'xsec' in dsetcfg
    )

    if 'xsec' in dsetcfg:
        theds.set_xsec(dsetcfg['xsec'])
    elif 'lumi' in dsetcfg:
        theds.set_lumi(dsetcfg['lumi'])

    return theds

def build_pq_dataset(configsuite : str,
                     runtag: str, 
                     dataset: str, 
                     objsyst : str, 
                     table: str, 
                     location : str = 'local-submit',
                     no_count : bool = False,
                     label_override : str | None = None,
                     color_override : str | None = None,
                     extra_key : str | None = None) -> ParquetDataset:
    
    dsetcfg = lookup_dataset(runtag, dataset)
    
    fs, tablepath = lookup_skim_path(
        location,
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )

    thekey = dataset
    if extra_key is not None:
        thekey += '-' + extra_key

    pqds = ParquetDataset(
        key = thekey,
        color = color_override if color_override is not None else dsetcfg['color'],
        label = label_override if label_override is not None else dsetcfg['label'],
        path = tablepath,
        filesystem=fs,
    )

    if not no_count and objsyst != 'DATA':
        pqds.override_num_events(
            lookup_count(
                runtag, 
                dataset, 
            )
        )
        
    if 'lumi' in dsetcfg:
        pqds.set_lumi(dsetcfg['lumi'])
    elif 'xsec' in dsetcfg:
        pqds.set_xsec(dsetcfg['xsec'])
    else:
        raise RuntimeError(f"Dataset {dataset} in runtag {runtag} has neither lumi nor xsec defined!")
    
    return pqds