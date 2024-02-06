import awkward as ak
import numpy as np

from util.util import unflatMatrix, unflatVector

def getParts(x, name):
    arr = x[name]
    nPart = x[name+"BK"].nPart
    return unflatVector(arr, nPart)

def getSimonJets(x, name):
    return x[name+"BK"]

def getJets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iJet
    return x[jetsname][iJet]

def getCHSjets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iCHS

    #have to handle cases where there is no matched CHS jet
    #in this case we set iJet=0, so that we can still index with 
    #and then we go back through and zero everything out
    #so that we don't get any false positives
    iJet2 = ak.where(iJet == 9999, 0, iJet)
    ans = x[jetsname][iJet2]
    for field in ans.fields:
        ans[field] = ak.where(iJet == 9999, 0, ans[field])

    return ans
