import awkward as ak
import numpy as np
import warnings

def getMuons(x, name):
    ans = x[name]
    ans = ak.pad_none(ans, 2)
    if hasattr(ans, "RoccoR"):
        ans['pt'] = ans.pt * ans.RoccoR
    else:
        warnings.warn("No RoccoR found, using pt as is")
    return ans

def getRawMuons(x, name):
    return ak.pad_none(x[name], 2)
