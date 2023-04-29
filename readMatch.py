import awkward as ak
import numpy as np

from util import unflatMatrix, unflatVector

def getParts(x, name):
    arr = x[name]
    nPart = x[name+"BK"].nPart
    return unflatVector(arr, nPart)

def getJets(x, name):
    return ak.zip({
        "pt": x[name+"BK"].jetPt,
        "eta": x[name+"BK"].jetEta,
        'phi': x[name+"BK"].jetPhi,
        'nPart': x[name+"BK"].nPart,     
    })
