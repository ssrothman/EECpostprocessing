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
    result = x[name+'dipole']
    nEntry = x[name+'BK'].nEntry_dipole
    result = unflatVector(result, nEntry)
    result = result[result.wt > 0]
    return result

def getRes4tee(x, name):
    result = x[name+'tee']
    nEntry = x[name+'BK'].nEntry_tee
    result = unflatVector(result, nEntry)
    result = result[result.wt > 0]
    return result

def getRes4triangle(x, name):
    result = x[name+'triangle']
    nEntry = x[name+'BK'].nEntry_triangle
    result = unflatVector(result, nEntry)
    result = result[result.wt > 0]
    return result

def getTransferRes4dipole(x, name):
    result = x[name+'dipole']
    nEntry = x[name+'BK'].nEntries_dipole
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen > 0]
    return result

def getTransferRes4tee(x, name):
    result = x[name+'tee']
    nEntry = x[name+'BK'].nEntries_tee
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen > 0]
    return result

def getTransferRes4triangle(x, name):
    result = x[name+'triangle']
    nEntry = x[name+'BK'].nEntries_triangle
    result = unflatVector(result, nEntry)
    result = result[result.wt_gen > 0]
    return result

def getTransferShapeRes4tee(x, name):
    idx = ak.where(ak.num(x[name+'BK']))[0][0]
    BK0 = x[name+'BK'][idx]
    return {
        'R_reco' : BK0.nR_tee_reco[0],
        'r_reco' : BK0.nr_tee_reco[0],
        'c_reco' : BK0.nc_tee_reco[0],
        'R_gen' : BK0.nR_tee_gen[0],
        'r_gen' : BK0.nr_tee_gen[0],
        'c_gen' : BK0.nc_tee_gen[0]
    }

def getTransferShapeRes4dipole(x, name):
    idx = ak.where(ak.num(x[name+'BK']))[0][0]
    BK0 = x[name+'BK'][idx]
    return {
        'R_reco' : BK0.nR_dipole_reco[0],
        'r_reco' : BK0.nr_dipole_reco[0],
        'c_reco' : BK0.nc_dipole_reco[0],
        'R_gen' : BK0.nR_dipole_gen[0],
        'r_gen' : BK0.nr_dipole_gen[0],
        'c_gen' : BK0.nc_dipole_gen[0]
    }

def getTransferShapeRes4triangle(x, name):
    idx = ak.where(ak.num(x[name+'BK']))[0][0]
    BK0 = x[name+'BK'][idx]
    return {
        'R_reco' : BK0.nR_triangle_reco[0],
        'r_reco' : BK0.nr_triangle_reco[0],
        'c_reco' : BK0.nc_triangle_reco[0],
        'R_gen' : BK0.nR_triangle_gen[0],
        'r_gen' : BK0.nr_triangle_gen[0],
        'c_gen' : BK0.nc_triangle_gen[0]
    }

def getPtDenom(x, name):
    return x[name+"BK"].pt_denom

def getPtDenomGen(x, name):
    return x[name+"BK"].pt_denom_gen

def getPtDenomReco(x, name):
    return x[name+"BK"].pt_denom_reco

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

