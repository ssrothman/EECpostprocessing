import awkward as ak
import numpy as np
import warnings

def getMuons(x, name, noRoccoR=False):
    ans = x[name]
    ans = ak.pad_none(ans, 2)

    if noRoccoR or not hasattr(ans, "RoccoR"):
        ans['rawPt'] = ans.pt
        ans['pt'] = ans.pt

        if not noRoccoR:
            warnings.warn("Could not find RoccoR - not applying")
    else:
        ans['rawPt'] = ans.pt 
        ans['pt'] = ans.pt * ans.RoccoR

    return ans
