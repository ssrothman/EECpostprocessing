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
    return x[jetsname][iJet]
