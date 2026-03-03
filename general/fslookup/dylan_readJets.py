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
    #if CHSname is not None and CHSname != '':
    #    matchedCHS = getMatchedCHSjets(x, CHSname, name)
    #    print(matchedCHS.pt)
    #    print((matchedCHS.pt).type)
    #    print((matchedCHS.pt[1]))
    #    ans['nCHS'] = ak.num(matchedCHS.pt, axis=2)

    return ans

def getJets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iJet
    return x[jetsname][iJet]

def getCHSjets(x, jetsname):
    return x[jetsname]

def getMatchedCHSjets(x, CHSname, simonname):
    return x[simonname + "BK"].iJet
    #iJet = x[simonjetsname+"BK"].iJet
    #return x[simonjetsname+"BK"][iJet]
    #nCHS = x[simonname+"BK"].nCHS
    nCHS = x[simonname+"BK"].nPart
    #iCHS = x[simonname+'CHS'].idx
    #print(nCHS, iCHS, simonname, x, x[simonname+'CHS'])
    #print(nCHS, simonname, x, x[simonname+'CHS'])

    #bad = iCHS == 99999999
    #iCHS = ak.where(bad, 0, iCHS)

    print(x[CHSname])
    print(nCHS)
    print(ak.flatten(nCHS))
    print(x[CHSname].type)
    print(x[CHSname][0])
    print(x[CHSname][0].type)
    print(len(nCHS))
    #print(ak.flatten(nCHS).shape)
    #print(x[CHSname].shape)


    result = ak.unflatten(x[CHSname], ak.flatten(nCHS), axis=1)
    print(result)
    bad = ak.unflatten(bad, ak.flatten(nCHS), axis=1)

    result['pt'] = ak.where(bad, 0, result.pt)
    result = result[result.pt > 0]
    return result
