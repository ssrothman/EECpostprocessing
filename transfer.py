import awkward as ak
import numpy as np
import hist

def cleanDivide(num, denom):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(num, denom)
        c = np.nan_to_num(c, copy=False, nan=0, posinf=0, neginf=0)
        return c

def getTransferMatrix(Htrans, Hgen):
    trans = Htrans.values(flow=True)
    gen = Hgen.values(flow=True)

    invgen = cleanDivide(1, gen)
    
    transMat = np.einsum('ij,j->ij', trans, invgen)
    return transMat

def forwardTransfer(transMat, Hgen, Hcovgen):
    gen = Hgen.values(flow=True)
    covgen = Hcovgen.values(flow=True)

    reco = np.einsum('ij,j->i', transMat, gen)
    covreco = np.einsum('ij,jk,lk->il', transMat, covgen, transMat)

    return reco, covreco
