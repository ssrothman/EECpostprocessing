import numpy as np
import awkward as ak

from util import unflatMatrix, unflatVector

def getptrans(x, name):
    arr = x.GenMatch.matrix
    nrows = x.GenMatchBK.n_rows
    ncols = x.GenMatchBK.n_cols
    return unflatMatrix(arr, nrows, ncols)

def getcovPxP(x, name):
    arr = x[name+'COVPxP'].value
    nrows = x[name+'BK'].nWts
    ncols = nrows
    return unflatMatrix(arr, nrows, ncols)

def getcov3x3(x, name):
    arr = x[name+'COV3x3'].value
    nrows = x[name+'BK'].nRes3
    ncols = nrows
    return unflatMatrix(arr, nrows, ncols)

def getcov3xP(x, name):
    arr = x[name+'COV3xP'].value
    nrows = x[name+'BK'].nRes3
    ncols = x[name+'BK'].nWts
    return unflatMatrix(arr, nrows, ncols)

def getcov4x4(x, name):
    arr = x[name+'COV4x4'].value
    nrows = x[name+'BK'].nRes4
    ncols = nrows
    return unflatMatrix(arr, nrows, ncols)

def getcov4x3(x, name):
    arr = x[name+'COV4x3'].value
    nrows = x[name+'BK'].nRes4
    ncols = x[name+'BK'].nRes3
    return unflatMatrix(arr, nrows, ncols)

def getcov4xP(x, name):
    arr = x[name+'COV4xP'].value
    nrows = x[name+'BK'].nRes4
    ncols = x[name+'BK'].nWts
    return unflatMatrix(arr, nrows, ncols)

def getproj(x, name):
    arr = x[name+'WTS'].value
    nrows = x[name+'BK'].nOrders
    ncols = x[name+'BK'].nDR
    return unflatMatrix(arr, nrows, ncols)

def getprojdR(x, name):
    arr = x[name+'DRS'].value
    nrows = x[name+'BK'].nDR
    return unflatVector(arr, nrows)

def getprojOrder(x, name):
    arr = x[name+"WTSBK"].order
    nrows = x[name+'BK'].nOrders
    return unflatVector(arr, nrows)

def getprojJetIdx(x, name):
    arr = x[name+"BK"].iJet
    return arr

def getres3(x, name):
    arr = x[name+'RES3']
    nrows = x[name+'BK'].nRes3
    return unflatVector(arr, nrows)

def getres4(x, name):
    arr = x[name+'RES4']
    nrows = x[name+'BK'].nRes4
    return unflatVector(arr, nrows)

def gettransferP(x, name): 
    '''
    NB index in last two dimensions is backwards: [gen, reco]
    '''
    arr = x[name+'PROJ'].value
    nOrd = ak.flatten(x[name+'BK'].nOrder, axis=None)
    nTrans = x[name+"BK"].nTransP
    arr = unflatMatrix(arr, nOrd, nTrans)
    nrows = ak.flatten(x[name+'BK'].nGenP, axis=None)
    ncols = ak.flatten(x[name+'BK'].nRecoP, axis=None)
    cts = np.repeat(nrows, ncols*nOrd)
    return ak.unflatten(arr, cts, axis=-1)

def gettransferRecoIdx(x, name):
    arr = x[name+'BK'].iReco
    return arr

def gettransferGenIdx(x, name):
    arr = x[name+'BK'].iGen
    return arr

def getTransfer3(x, name):
    '''
    NB index in last two dimensions is backwards: [gen, reco]
    '''
    arr = x[name+'RES3'].value
    nrows = x[name+'BK'].nGen3
    ncols = x[name+'BK'].nReco3
    return unflatMatrix(arr, nrows, ncols)

def getTransfer4(x, name):
    '''
    NB index in last two dimensions is backwards: [gen, reco]
    '''
    arr = x[name+'RES4'].value
    nrows = x[name+'BK'].nGen4
    ncols = x[name+'BK'].nReco4
    return unflatMatrix(arr, nrows, ncols)
