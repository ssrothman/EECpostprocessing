import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights

def getEventWeight(x):
    ans = Weights(len(x))
    ans.add('generator', x.genWeight)
    return ans
