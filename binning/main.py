from typing import Sequence
import hist
from tqdm import tqdm
import numpy as np
import directcov
import pyarrow.dataset as ds
import pyarrow.compute as pc
from correctionlib import Correction 

def build_transfer_hist(gencfg, recocfg):
    axes = []
    for axis_cfg in gencfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name']+"_gen",
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name']+"_gen",
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}")
    for axis_cfg in recocfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name']+"_reco",
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name']+"_reco",
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}") 
    H = hist.Hist(*axes, storage=hist.storage.Weight())
    return H

def build_hist(cfg):
    axes = []
    for axis_cfg in cfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name'],
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name'],
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}")
        
    H = hist.Hist(*axes, storage=hist.storage.Weight())
    return H

def get(batch, name):
    return batch[name].to_numpy(zero_copy_only=False)

def build_iterator(dset : ds.Dataset,
                    names : Sequence[str], 
                    weightname: str, itemwt: str | None,
                    statN : int,
                    statK : int,
                    reweight : Correction | None = None):
    columns = list(names) + [weightname, 'event_id']
    if itemwt is not None:
        columns.append(itemwt)
    if reweight is not None:
        for input in reweight.inputs:
            columns.append(input.name)
            
    thefilter = statsplit_filter(statN, statK)

    return tqdm(dset.to_batches(
        columns = columns,
        batch_readahead = 2,
        fragment_readahead = 2,
        use_threads = True,
        batch_size = 1<<20,
        filter = thefilter
    ))

def pa_mod(val, divisor):
    #pyarrow doesn't know about pyarrow.compute??
    quotient = pc.floor(pc.divide(val, divisor)) # pyright: ignore[reportAttributeAccessIssue]
    remainder = pc.subtract(val, pc.multiply(quotient, divisor)) # pyright: ignore[reportAttributeAccessIssue]
    return remainder

def statsplit_filter(statN, statK):
    if statN < 0:
        return pa_mod(ds.field('event_id'), statN) == statK
    else:
        return None
    
def fill_cov(H, dset : ds.Dataset,
             weightname: str,
             itemwt: str | None = None,
             statN : int = -1,
             statK : int = -1,
             reweight : Correction | None = None) -> np.ndarray:
    
    iterator = build_iterator(
        dset, H.axes.name,
        weightname, itemwt,
        statN, statK,
        reweight
    )
    total_rows = dset.count_rows()
    rows_so_far = 0

    shape = [ax.extent for ax in H.axes]
    cov = directcov.DirectCov(shape, 1)

    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        indices = [
            H.axes[name].index(get(batch, name)) for name in H.axes.name
        ]
        indices = np.stack(indices, axis=1) # pyright: ignore[reportArgumentType, reportCallIssue]
        
        for i, name in enumerate(H.axes.name):
            if H.axes[name].traits.underflow:
                indices[:,i] += 1

        weight = get(batch, weightname)
        if itemwt is not None:
            weight = weight * get(batch, itemwt)
        if reweight is not None:
            rwin = {}
            for i, input in enumerate(reweight.inputs):
                rwin[input] = get(batch, input.name)
            weight = weight * reweight.evaluate(**rwin)

        evtid = get(batch, 'event_id')

        cov.fillEvents(indices, weight, evtid)
            
    cov.finalize()

    return np.array(cov, copy=True)

def fill_hist(H : hist.Hist,
              dset : ds.Dataset, 
              weightname : str,
              itemwt : str | None = None,
              statN : int = -1,
              statK : int = -1,
              reweight : Correction | None = None) -> hist.Hist:
    
    iterator = build_iterator(
        dset, H.axes.name, 
        weightname, itemwt,
        statN, statK,
        reweight
    )
    
    total_rows = dset.count_rows()
    rows_so_far = 0

    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        weight = get(batch, weightname)
        if itemwt is not None:
            weight = weight * get(batch, itemwt)
        if reweight is not None:
            rwin = {}
            for i, input in enumerate(reweight.inputs):
                rwin[input] = get(batch, input.name)
            weight = weight * reweight.evaluate(**rwin)

        H.fill(
            **{name: get(batch, name) for name in H.axes.name},
            weight=weight
        )

    return H
