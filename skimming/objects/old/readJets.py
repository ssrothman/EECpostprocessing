import awkward as ak
import numpy as np

from util.util import unflatMatrix, unflatVector

def getParts(x, name):
    arr = x[name]
    nPart = x[name+"BK"].nPart
    parts = unflatVector(arr, nPart)
    return parts[parts.pt > 0]

def getSimonJets(x, name, CHSname):
    ans = x[name+"BK"]
    
    parts = getParts(x, name)
    ans['nPart'] = ak.num(parts, axis=-1)

    if CHSname is not None and CHSname != '':
        matchedCHS = getMatchedCHSjets(x, CHSname, name)
        ans['nCHS'] = ak.num(matchedCHS.pt, axis=2)

    return ans

def getJets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iJet
    return x[jetsname][iJet]

def getCHSjets(x, jetsname):
    return x[jetsname]

def getMatchedCHSjets(x, CHSname, simonname):
    nCHS = x[simonname+"BK"].nCHS
    iCHS = x[simonname+'CHS'].idx

    bad = iCHS == 99999999
    iCHS = ak.where(bad, 0, iCHS)

    result = ak.unflatten(x[CHSname][iCHS], ak.flatten(nCHS), axis=1)
    bad = ak.unflatten(bad, ak.flatten(nCHS), axis=1)

    result['pt'] = ak.where(bad, 0, result.pt)
    result = result[result.pt > 0]
    return result
