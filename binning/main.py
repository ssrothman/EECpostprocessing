import json
from typing import Any, List, Sequence
import hist
from tqdm import tqdm
import numpy as np
try:
    import directcov
except ImportError:
    directcov = None
import pyarrow.dataset as ds
import pyarrow.compute as pc
from correctionlib import Correction 
import os.path
import os

def narrow_axis(axis, narrow_to: List[int]):
    if len(narrow_to) != 2:
        raise ValueError("narrow_to must be a list of two integers [minbin, maxbin]")

    underflow = axis.traits.underflow or narrow_to[0] > 0
    overflow = axis.traits.overflow or narrow_to[1] < axis.size - 1

    if isinstance(axis, hist.axis.Regular):
        theaxis = hist.axis.Regular(
            narrow_to[1] - narrow_to[0],
            axis.edges[narrow_to[0]],
            axis.edges[narrow_to[1]],
            name=axis.name,
            transform=axis.transform,
            underflow=underflow,
            overflow=overflow
        )
    elif isinstance(axis, hist.axis.Variable):
        edges = axis.edges[narrow_to[0]:narrow_to[1]]
        theaxis = hist.axis.Variable(
            edges,
            name=axis.name,
            underflow=underflow,
            overflow=overflow
        )
    else:
        raise ValueError(f"Narrowing not supported for axis type: {type(axis)}")
    
    print("Narrowed axis", axis.name, "from", axis, "to", theaxis)        
    return theaxis

def rebin_axis(axis, rebin : int):
    if isinstance(axis, hist.axis.Regular):
        theaxis = hist.axis.Regular(
            axis.size // rebin,
            np.min(axis.edges),
            np.max(axis.edges),
            name=axis.name,
            transform=axis.transform,
            underflow=axis.traits.underflow,
            overflow=axis.traits.overflow
        )
    elif isinstance(axis, hist.axis.Variable):
        edges = axis.edges
        rebinned_edges = [
            edges[i] for i in range(0, len(edges), rebin)
        ]
        if edges[-1] != rebinned_edges[-1]:
            rebinned_edges.append(edges[-1])
        theaxis = hist.axis.Variable(
            rebinned_edges,
            name=axis.name,
            underflow=axis.traits.underflow,
            overflow=axis.traits.overflow
        )
    else:
        raise ValueError(f"Rebinning not supported for axis type: {type(axis)}")

    print("Rebinned axis", axis.name, "from", axis, "to", theaxis)        
    return theaxis

def build_transfer_config(gencfg : List[dict], recocfg : List[dict]) -> List[dict]:
    axes = []
    for axis_cfg in recocfg:
        copycfg = axis_cfg.copy()
        copycfg['name'] = axis_cfg['name'] + '_reco'
        axes.append(copycfg)
    for axis_cfg in gencfg:
        copycfg = axis_cfg.copy()
        copycfg['name'] = axis_cfg['name'] + '_gen'
        axes.append(copycfg)
    return axes

def build_hist(cfg : List[dict]):
    axes = []
    prebinned_edges = {}
    for axis_cfg in cfg:
        axistype = axis_cfg['type']
        if '-' in axistype:
            axistype, subtype = axistype.split('-', 1)
        else:
            subtype = None

        if axistype == 'Regular':
            theaxis = hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name'],
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            )
        elif axistype == 'Variable':
            theaxis = hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name'],
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            )
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}")
        
        if 'clamp' in axis_cfg and axis_cfg['clamp']:
            theaxis.clamp = True
        else:
            theaxis.clamp = False

        if subtype == 'Prebinned':
            prebinned_axis = theaxis

            if 'rebin' in axis_cfg:
                theaxis = rebin_axis(prebinned_axis, axis_cfg['rebin'])

            if 'narrow_to' in axis_cfg:
                theaxis = narrow_axis(theaxis, axis_cfg['narrow_to'])
            
            pbedges = prebinned_axis.edges
            if prebinned_axis.traits.underflow:
                pbedges = [np.inf] + list(pbedges)
            if prebinned_axis.traits.overflow:
                pbedges = list(pbedges) + [np.inf]

            prebinned_edges[axis_cfg['name']] = np.asarray(pbedges)

        axes.append(theaxis)

    H = hist.Hist(*axes, storage=hist.storage.Weight())
    return H, prebinned_edges

def get(batch, name : str):
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
    #pyright doesn't know about pyarrow.compute??
    quotient = pc.floor(pc.divide(val, divisor)) # pyright: ignore[reportAttributeAccessIssue]
    remainder = pc.subtract(val, pc.multiply(quotient, divisor)) # pyright: ignore[reportAttributeAccessIssue]
    return remainder

def statsplit_filter(statN, statK):
    if statN > 0:
        return pa_mod(ds.field('event_id'), statN) == statK
    else:
        return None
    
def fill_cov(H, prebinned : dict[str, np.ndarray],
             dset : ds.Dataset,
             weightname: str,
             itemwt : str | None = None,
             statN : int = -1,
             statK : int = -1,
             reweight : Correction | None = None) -> np.ndarray:
    if directcov is None:
        raise ImportError("directcov is not installed")
    
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
        if batch.num_rows == 0:
            continue

        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        values = [
            prebinned[name][get(batch, name)] if name in prebinned else get(batch, name) for name in H.axes.name
        ]
        indices = [
            H.axes[i].index(values[i]) for i in range(len(H.axes))
        ]
        indices = np.stack(indices, axis=1) # pyright: ignore[reportArgumentType, reportCallIssue]

        for i, ax in enumerate(H.axes):
            if ax.clamp:
                indices[:,i] = np.clip(indices[:,i], 0, ax.extent-1)

        for i, name in enumerate(H.axes.name):
            if H.axes[name].traits.underflow:
                indices[:,i] += 1

        # mask out entries that fall outside any non-clamped axis
        mask = np.ones(len(indices), dtype=bool)
        for i, ax in enumerate(H.axes):
            if not ax.clamp:
                mask &= (indices[:,i] >= 0) & (indices[:,i] < ax.extent)
        indices = indices[mask]

        weight = get(batch, weightname)
        if itemwt is not None:
            weight = weight * get(batch, itemwt)
        if reweight is not None:
            rwin = {}
            for i, input in enumerate(reweight.inputs):
                rwin[input] = get(batch, input.name)
            weight = weight * reweight.evaluate(**rwin)

        weight = weight[mask]
        evtid = get(batch, 'event_id')[mask]

        cov.fillEvents(indices, weight, evtid)
            
    cov.finalize()

    return np.array(cov, copy=True)

def fill_hist(H : hist.Hist,
              prebinned : dict[str, np.ndarray],
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

    if total_rows == 0:
        return H

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

        filldict = {
            name : get(batch, name) for name in H.axes.name
        }
        for name in prebinned:
            filldict[name] = prebinned[name][filldict[name]]
            if H.axes[name].clamp:
                filldict[name] = np.clip(filldict[name], H.axes[name].edges[0], H.axes[name].edges[-1])

        H.fill(
            **filldict,
            weight=weight
        )

    return H
