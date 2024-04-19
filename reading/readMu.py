import awkward as ak
import numpy as np
import warnings

def getMuons(x, name, noRoccoR):
    ans = x[name]
    ans = ak.pad_none(ans, 2)
    if not noRoccoR:
        if hasattr(ans, "RoccoR"):
            ans['pt'] = ans.pt * ans.RoccoR
        else:
            warnings.warn("No RoccoR found, using pt as is")
    else:
        warnings.warn("No RoccoR applied")
    return ans

def getRawMuons(x, name):
    return ak.pad_none(x[name], 2)
