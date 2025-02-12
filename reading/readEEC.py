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
    wts = x[name+'res4dipole'].value
    BK = getBK(x, name)

    nRL = BK.nres4_dipole_RL
    nr = BK.nres4_dipole_r
    nct = BK.nres4_dipole_ct

    return unflatRecursive(wts, [nRL, nr, nct])

def getRes4tee(x, name):
    wts = x[name+'res4tee'].value
    BK = getBK(x, name)

    nRL = BK.nres4_tee_RL
    nr = BK.nres4_dipole_r
    nct = BK.nres4_dipole_ct

    return unflatRecursive(wts, [nRL, nr, nct])

def getRes4triangle(x, name):
    wts = x[name+'res4triangle'].value
    BK = getBK(x, name)

    nRL = BK.nres4_triangle_RL
    nr = BK.nres4_triangle_r
    nct = BK.nres4_triangle_ct

    return unflatRecursive(wts, [nRL, nr, nct])

def getRes4minR(x, name):
    wts = x[name+'res4minR'].value
    BK = getBK(x, name)

    nR = BK.nres4_minR_RL
    nr1 = BK.nres4_minR_r1
    nr2 = BK.nres4_minR_r2
    nphi = BK.nres4_minR_phi

    return unflatRecursive(wts, [nR, nr1, nr2, nphi])

def getTransferRes4Shapes(x, name):
    vals = x[name+'res4shapes'].value
    BK = getBK(x, name)

    nshape = BK.nres4shapes_shape
    nRL = BK.nres4shapes_RL
    nr = BK.nres4shapes_r
    nct = BK.nres4shapes_ct

    return unflatRecursive(vals, [nshape, nRL, nr, nct, nshape, nRL, nr, nct])

def getRes4fixed(x, name):
    wts = x[name+'res4fixed'].value
    BK = getBK(x, name)

    nshape = BK.nres4fixed_shape
    nRL = BK.nres4fixed_RL

    return unflatRecursive(wts, [nshape, nRL])

def getTransferRes4Fixed(x, name):
    vals = x[name+'res4fixed'].value
    BK = getBK(x, name)

    nshape = BK.nres4fixed_shape
    nRL = BK.nres4fixed_RL

    return unflatRecursive(vals, [nshape, nRL, nshape, nRL])

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

