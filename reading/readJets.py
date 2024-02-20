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
    nCHS = x[simonjetsname+"BK"].nCHS
    iCHS = x[simonjetsname+"CHS"].idx
    bad = iCHS == 99999999
    iCHS = ak.where(bad, 0, iCHS)
    jets = x[jetsname][iCHS]

    #zero out everything that wasn't actually a match
    for field in jets.fields:
        jets[field] = ~bad * jets[field]

    jets = ak.unflatten(jets, ak.flatten(nCHS, axis=None), axis=1)
    return jets
