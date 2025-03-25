import numpy as np
import awkward as ak

from util.util import unflatMatrix, unflatVector, unflatRecursive

def getBK(x, name):
    return x[name+"BK"]

def getNProj(x, name):
    return getBK(x, name).nproj

def getProj(x, name, which):
    wts = x[name+'proj'][which]
    nproj = getBK(x, name).nproj
    return unflatVector(wts, nproj)

def getAllProj(x, name):
    projs = [getProj(x, name, 'value%d'%order)[:,:,None,:] for order in range(2,7)]
    return ak.concatenate(projs, axis=2)

def getRes3(x, name):
    wts = x[name+'res3'].value
    BK = getBK(x, name)
    
    nRL = BK.nres3_RL
    nxi = BK.nres3_xi
    nphi = BK.nres3_phi

    return unflatRecursive(wts, [nRL, nxi, nphi])

def getTransferRes3(x, name):
    vals = x[name+'res3'].value
    BK = getBK(x, name)

    nRL = BK.nres3_RL
    nxi = BK.nres3_xi
    nphi = BK.nres3_phi

    return unflatRecursive(vals, [nRL, nxi, nphi, nRL, nxi, nphi])

def getRes4dipole(x, name):
    wts = x[name+'dipole'].value
    BK = getBK(x, name)

    nRL = BK.nR_dipole
    nr = BK.nr_dipole
    nct = BK.nc_dipole

    return unflatRecursive(wts, [nRL, nr, nct])

def getRes4tee(x, name):
    wts = x[name+'tee'].value
    BK = getBK(x, name)

    nRL = BK.nR_tee
    nr = BK.nr_tee
    nct = BK.nc_tee

    return unflatRecursive(wts, [nRL, nr, nct])

def getRes4triangle(x, name):
    wts = x[name+'triangle'].value
    BK = getBK(x, name)

    nRL = BK.nR_triangle
    nr = BK.nr_triangle
    nct = BK.nc_triangle

    return unflatRecursive(wts, [nRL, nr, nct])

def getTransferRes4dipole(x, name):
    result = x[name+'dipole']
    nEntry = x[name+'BK'].nEntries_dipole
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen >= 0]
    return result

def getTransferRes4tee(x, name):
    result = x[name+'tee']
    nEntry = x[name+'BK'].nEntries_tee
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen >= 0]
    return result

def getTransferRes4triangle(x, name):
    result = x[name+'triangle']
    nEntry = x[name+'BK'].nEntries_triangle
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen >= 0]
    return result

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

