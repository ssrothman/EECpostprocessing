import awkward as ak
import numpy as np
import hist
from .util import cleanDivide

def getOrderIdx(H, order, returnBdiag = False):
    if 'order' in H.axes.name:
        idx = H.axes['order'].index(order)
    elif 'ordera' in H.axes.name:
        idx = H.axes['ordera'].index(order)
    else:
        raise ValueError('No order axis found')
    if returnBdiag:
        return idx, 'order' in H.axes.name
    else:
        return idx

def getForOrder1d(H, order):
    idx = getOrderIdx(H, order)
    return H[{'order' : idx}].project('dR')

def getForOrder2d(H, order):
    idx, bdiag = getOrderIdx(H, order, returnBdiag=True)
    if bdiag:
        return H[{'order' : idx}].project('dRa', 'dRb')
    else:
        return H[{'ordera' : idx, 'orderb' : idx}].project('dRa', 'dRb')

def getTransferMatrix(Htrans, Hgen, Hreco, order = None):
    if order is not None:
        Htrans = getForOrder2d(Htrans, order)
        Hreco = getForOrder1d(Hreco, order)
        Hgen = getForOrder1d(Hgen, order)

    trans = Htrans.values(flow=True)
    gen = Hgen.values(flow=True)

    invgen = cleanDivide(1, gen)
    
    transMat = np.einsum('ij,j->ij', trans, invgen)

    norm = Hreco.sum(flow=True)

    return transMat#/norm

def getProjectedValsCov(h, hcov, order=None, normalize=True, flow=False):
    if order is not None:
        h = getForOrder1d(h, order)
        hcov = getForOrder2d(hcov, order)
    
    norm = 1
    if normalize:
        norm = h.sum(flow=True)

    vals = h.values(flow=flow)/norm
    cov = hcov.values(flow=flow)/np.square(norm)

    return vals, cov

def getBackgroundTemplate(Htrans, Hgen, Hcovgen, Hreco, Hcovreco, order=None, 
                          normByGen=True):
    genvals, gencov = getProjectedValsCov(Hgen, Hcovgen, order=order, 
                                          flow=True, normalize=False)
    
    transMat = getTransferMatrix(Htrans, Hgen, Hreco, order)

    forward, covforward = forwardTransfer(transMat, genvals, gencov)

    recovals, recocov = getProjectedValsCov(Hreco, Hcovreco, order=order,
                                            flow=True, normalize=False)

    background = recovals - forward
    backgroundcov = covforward + recocov

    if normByGen:
        norm = genvals.sum()
    else:
        norm = recovals.sum()

    templ = background/norm
    templcov = backgroundcov/np.square(norm)

    return templ, templcov

def forwardTransfer(transMat, genvals, gencov):

    reco = np.einsum('ij,j->i', transMat, genvals)
    covreco = np.einsum('ij,jk,lk->il', transMat, gencov, transMat)

    return reco, covreco
