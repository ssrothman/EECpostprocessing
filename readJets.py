import awkward as ak
import numpy as np

from util import unflatMatrix, unflatVector

def getParts(x, name):
    arr = x[name]
    nPart = x[name+"BK"].nPart
    return unflatVector(arr, nPart)

def getSimonJets(x, name):
    return ak.zip({
        "pt": x[name+"BK"].jetPt,
        "eta": x[name+"BK"].jetEta,
        'phi': x[name+"BK"].jetPhi,
        'nPart': x[name+"BK"].nPart,     
    })

def getJets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iJet
    return x[jetsname][iJet]
