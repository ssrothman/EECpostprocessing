import awkward as ak
import numpy as np

def unflatMatrix(arr, nrows, ncols):
    nrows = ak.flatten(nrows, axis=None)
    ncols = ak.flatten(ncols, axis=None)
    ntot = nrows*ncols

    ans = ak.unflatten(arr, ntot, axis=-1)
    ans = ak.unflatten(ans, np.repeat(ncols, nrows), axis=-1)

    return ans

def unflatVector(arr, ncols):
    ncols = ak.flatten(ncols, axis=None)
    ans = ak.unflatten(arr, ncols, axis=-1)
    return ans

def ensure_mask(mask, arr):
    if mask is None:
        return ak.ones_like(arr, dtype=bool)
    else:
        return mask

def cleanDivide(num, denom):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(num, denom)
        c = np.nan_to_num(c, copy=False, nan=0, posinf=0, neginf=0)
        return c
