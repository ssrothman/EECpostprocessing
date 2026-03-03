import awkward as ak
import numpy as np

def unflatMatrix(arr : ak.Array, nrows : ak.Array, ncols : ak.Array) -> ak.Array:
    nrows = ak.flatten(nrows, axis=None)
    ncols = ak.flatten(ncols, axis=None)
    ntot = nrows*ncols

    ans = ak.unflatten(arr, ntot, axis=-1)
    ans = ak.unflatten(ans, np.repeat(ncols, nrows), axis=-1)

    return ans

def unflatVector(arr : ak.Array, ncols : ak.Array) -> ak.Array:
    ncols = ak.flatten(ncols, axis=None)
    ans = ak.unflatten(arr, ncols, axis=-1)
    return ans
