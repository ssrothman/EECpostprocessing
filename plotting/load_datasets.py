from general.datasets import datasets
from general.datasets.datasets import lookup_dataset, cfg
from general.fslookup.skim_path import lookup_skim_path
from simonplot.plottables import ParquetDataset
from simonplot.plottables.Datasets import DatasetStack
import os
import json
import numpy as np

from simonplot.plottables.PrebinnedDatasets import ValCovPairDataset
from simonpy.AbitraryBinning import ArbitraryBinning

def build_pq_dataset_stack(configsuite : str,
                           runtag : str,
                           stackname : str,
                           objsyst : str,
                           table : str,
                           location : str = 'local-submit',
                           no_count : bool = False,
                           label_override : str | None = None,
                           color_override : str | None = None) -> DatasetStack:
    stackcfg = cfg['stacks'][stackname]
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

    return DatasetStack(
        key = stackname + '-' + objsyst,
        color = color_override if color_override is not None else stackcfg['color'],
        label = label_override if label_override is not None else stackcfg['label'],
        datasets = dsets
    )

def load_prebinned_dataset(configsuite : str,
                           runtag : str,
                           dataset :str,
                           objsyst : str,
                           wtsyst : str,
                           table : str,
                           location : str = 'local-submit',
                           label_override : str | None = None,
                           color_override : str | None = None) -> ValCovPairDataset:
    
    dsetcfg = lookup_dataset(runtag, dataset)
    
    fs, tablepath = lookup_skim_path(
        location,
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )

    print(tablepath)
    valpath = tablepath + '_BINNED_%s.npy' % wtsyst
    covpath = tablepath + '_BINNED_covmat_%s.npy' % wtsyst
    binningpath = tablepath + '_bincfg.json'

    vals = np.load(fs.open(valpath, 'rb'))
    covmat = np.load(fs.open(covpath, 'rb'))
    with fs.open(binningpath, 'r') as f:
        bincfg = json.load(f)
    
    binning = ArbitraryBinning()
    binning.from_dict(bincfg)

    theds = ValCovPairDataset(
        key = dataset + '-' + objsyst + '-' + wtsyst,
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
                     color_override : str | None = None) -> ParquetDataset:
    
    dsetcfg = lookup_dataset(runtag, dataset)
    
    fs, tablepath = lookup_skim_path(
        location,
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )

    pqds = ParquetDataset(
        key = dataset + '-' + objsyst,
        color = color_override if color_override is not None else dsetcfg['color'],
        label = label_override if label_override is not None else dsetcfg['label'],
        path = tablepath,
        filesystem=fs,
    )

    if not no_count:
        countfs, countpath = lookup_skim_path(
            location,
            configsuite,
            runtag,
            dataset,
            objsyst,
            'count'
        )

        with countfs.open(os.path.join(countpath, 'merged.json'), 'r') as f:
            countdict = json.load(f)

        pqds.override_num_events(countdict['n_events'])
        
    if 'lumi' in dsetcfg:
        pqds.set_lumi(dsetcfg['lumi'])
    elif 'xsec' in dsetcfg:
        pqds.set_xsec(dsetcfg['xsec'])
    else:
        raise RuntimeError(f"Dataset {dataset} in runtag {runtag} has neither lumi nor xsec defined!")
    
    return pqds