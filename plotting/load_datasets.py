from datasets import datasets
from datasets.datasets import lookup_dataset, cfg
from fslookup.skim_path import lookup_skim_path
from simonplot.plottables import ParquetDataset
from simonplot.plottables.Datasets import DatasetStack

def build_pq_dataset_stack(configsuite : str,
                           runtag : str,
                           stackname : str,
                           objsyst : str,
                           table : str,
                           location : str = 'local-submit') -> DatasetStack:
    stackcfg = cfg['stacks'][stackname]
    dsets = []
    for dset in stackcfg['dsets']:
        dsets.append(build_pq_dataset(
            configsuite,
            runtag, 
            dset, 
            objsyst, 
            table,
            location
        ))
    for dset in stackcfg['stacks']:
        dsets.append(build_pq_dataset_stack(
            configsuite,
            runtag,
            dset,
            objsyst,
            table,
            location
        ))

    return DatasetStack(
        key = stackname,
        color = stackcfg['color'],
        label = stackcfg['label'],
        datasets = dsets
    )

def build_pq_dataset(configsuite : str,
                     runtag: str, 
                     dataset: str, 
                     objsyst : str, 
                     table: str, 
                     location : str = 'local-submit') -> ParquetDataset:
    
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
        key = dataset,
        color = dsetcfg['color'],
        label = dsetcfg['label'],
        path = tablepath,
        filesystem=fs,
    )

    if 'lumi' in dsetcfg:
        pqds.set_lumi(dsetcfg['lumi'])
    elif 'xsec' in dsetcfg:
        pqds.set_xsec(dsetcfg['xsec'])
    else:
        raise RuntimeError(f"Dataset {dataset} in runtag {runtag} has neither lumi nor xsec defined!")
    
    return pqds