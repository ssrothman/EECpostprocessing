import numpy as np
import awkward as ak

from util.util import unflatMatrix, unflatVector

def getNProj(x, name):
    return x[name+"BK"].nproj

def getProj(x, name, which):
    wts = x[name+'proj'][which]
    nproj = getNProj(x, name)
    return unflatVector(wts, nproj)

def getJetIdx(x, name):
    return x[name+"BK"].iJet

def getRecoIdx(x, name):
    return x[name+"BK"].iReco

def getGenIdx(x, name):
    return x[name+"BK"].iGen

def getTransferP(x, name, which):
    vals = x[name+'proj'][which]
    
    nproj = getNProj(x, name)
    nmat = nproj*nproj

    return unflatMatrix(vals, nproj, nproj)
