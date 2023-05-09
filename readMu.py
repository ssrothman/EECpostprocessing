import awkward as ak
import numpy as np

def getMuons(x, name):
    ans = x[name]
    ans = ak.pad_none(ans, 2)
    ans['pt'] = ans.pt * ans.RoccoR
    return ans
