import awkward as ak
import numpy as np

from util.util import unflatMatrix, unflatVector

def getParts(x, name):
    arr = x[name]
    nPart = x[name+"BK"].nPart
    return unflatVector(arr, nPart)

def getSimonJets(x, name):
    if "GenParticles" in name:
        return ak.zip({
            "pt": x[name+"BK"].jetPt,
            "rawPt": x[name+"BK"].jetRawPt,
            "eta": x[name+"BK"].jetEta,
            'phi': x[name+"BK"].jetPhi,
            'nPart': x[name+"BK"].nPart,     
        })
    else:
        return ak.zip({
            "pt": x[name+"BK"].jetPt,
            "rawPt": x[name+"BK"].jetRawPt,
            "eta": x[name+"BK"].jetEta,
            'phi': x[name+"BK"].jetPhi,
            'nPart': x[name+"BK"].nPart,     
            #'genPt': x[name+"BK"].genPt,
            #'genEta': x[name+"BK"].genEta,
            #'genPhi': x[name+"BK"].genPhi,
        })


def getJets(x, jetsname, simonjetsname):
    iJet = x[simonjetsname+"BK"].iJet
    return x[jetsname][iJet]

def getResolutionStudy(x, name):
    res = x[name]
    resBK = x[name+"BK"]
    
    nPart = resBK.nPart[resBK.nPart!=0]
    return unflatVector(res, nPart)

def getResolutionStudyIdx(x, name):
    return x[name+"BK"].jetIdx[x[name+"BK"].nPart!=0]
