import pyarrow.compute as pc

from pyrandom123 import philox2x32, threefry2x32
from pyrandom123 import util as r123util

import numpy as np
import awkward as ak

def pa_mod(val, divisor):
    quotient = pc.floor(pc.divide(val, divisor))
    remainder = pc.subtract(val, pc.multiply(quotient, divisor))
    return remainder

def get_evthash(table):
    run = table['run'].to_numpy(zero_copy_only=False).astype(np.uint64)
    lumi = table['lumi'].to_numpy(zero_copy_only=False).astype(np.uint64)
    event = table['event'].to_numpy(zero_copy_only=False).astype(np.uint64)

    return ((run << 48) + (lumi << 32) + event).astype(np.uint64);

def get_rng_keys(Nboot, rng_offset, dtype=np.uint32):
    Nboot = dtype(Nboot)
    rng_offset = dtype(rng_offset)
    return np.arange(Nboot, dtype=dtype) + dtype(rng_offset)

def get_boot_wts(r123type, table, Nboot, rng_offset,
                 return_extra=False):
    if r123type == 'philox':
        RNG = philox2x32
        RNGkeytype = np.uint32
    elif r123type == 'threefry':
        RNG = threefry2x32
        RNGkeytype = np.uint64
    else:
        raise ValueError(f'Unknown r123type: {r123type}')

    evthash = get_evthash(table)
    evtrepeats = ak.to_numpy(ak.run_lengths(evthash))
    evtidxs = np.cumsum(evtrepeats) - 1

    rng_counters = evthash[evtidxs]
    rng_keys = get_rng_keys(Nboot, rng_offset, dtype=RNGkeytype)

    rng_uints = RNG.get_uint64(rng_counters[None, :], rng_keys[:, None])
    
    if return_extra:
        return r123util.uint64_to_poisson1(rng_uints), evtrepeats, \
                rng_counters, rng_keys
    else:
        return r123util.uint64_to_poisson1(rng_uints), evtrepeats

def fill_bootstrapped(H, filldict, weight, 
                      r123type, Nboot, rng_offset, table, 
                      skipNominal, collect_debug_info=False):
    if not skipNominal:
        H.fill(**filldict, weight=weight, bootstrap=0)

    if collect_debug_info:
        boots, repeats, counters, keys = get_boot_wts(r123type, table, 
                                                      Nboot, rng_offset, 
                                                      True)
    else:
        boots, repeats = get_boot_wts(r123type, table, Nboot, rng_offset, False)

    for i in range(Nboot):
        boot = np.repeat(boots[i], repeats)
        H.fill(**filldict, weight=weight*boot, bootstrap=i+1)

    if collect_debug_info:
        return boots, repeats, counters, keys

def fill_bootstrapped_fast(H, filldict, weight,
                           r123type, Nboot, rng_offset, table, 
                           skipNominal, collect_debug_info=False):
    axnames = H.axes[1:].name

    indices = [H.axes[name].index(filldict[name]) for name in axnames]
    mask = np.ones(indices[0].shape, dtype=bool)
    for i, name in enumerate(axnames):
        if H.axes[name].traits.underflow:
            indices[i] += 1
        else:
            mask &= indices[i] >= 0

        if not H.axes[name].traits.overflow:
            mask &= indices[i] < H.axes[name].size

    indices = tuple(indices[i][mask] for i in range(len(indices)))

    if not skipNominal:
        np.add.at(H.view(flow=True)[0], 
                  indices,
                  weight)

    if collect_debug_info:
        boots, repeats, counters, keys = get_boot_wts(r123type, table, 
                                                      Nboot, rng_offset, 
                                                      True)
    else:
        boots, repeats = get_boot_wts(r123type, table, Nboot, rng_offset, False)

    for i in range(Nboot):
        boot = np.repeat(boots[i], repeats)
        np.add.at(H.view(flow=True)[i+1], 
                  indices,
                  weight*boot)

    if collect_debug_info:
        return boots, repeats, counters, keys
