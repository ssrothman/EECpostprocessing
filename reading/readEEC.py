import numpy as np
import awkward as ak

from util.util import unflatMatrix, unflatVector

def getNProj(x, name):
    return x[name+"BK"].nproj

def getNres3(x, name):
    return x[name+"BK"].nres3

def getProj(x, name, which):
    wts = x[name+'proj'][which]
    nproj = getNProj(x, name)
    return unflatVector(wts, nproj)

def getRes3(x, name):
    wts = x[name+'res3'].value
    nres3 = getNres3(x, name)
    return unflatVector(wts, nres3)

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

def getTransferRes3(x, name):
    vals = x[name+'res3'].value

    nres3 = getNres3(x, name)
    nmat = nres3*nres3

    return unflatMatrix(vals, nres3, nres3)
