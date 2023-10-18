import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from util.util import ensure_mask

def getdRAxis(name='dR', label='$\Delta R$'):
    #return hist.axis.Regular(20, 1e-3, 1.0,
    #                         name=name, label=label, 
    #                         transform=hist.axis.transform.log)
    return hist.axis.Variable([0, 0.0002, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.137, 0.157,0.179, 0.205, 0.234, 0.268, 0.306, 0.35, 0.4, 0.8, 10], name=name, label=label)

def getOrderAxis(name='order', label='Order'):
    return hist.axis.IntCategory([2, 3, 4, 5, 6, -12, -22, -13], 
                                 name=name, label=label)

def getPtAxis(name='pt', label='$p_T$'):
    return hist.axis.Regular(20, 30.0, 1000.0,
                             name=name, label=label)


def getHistP():
    return hist.Hist(
        getdRAxis(),
        getOrderAxis(),
        storage=hist.storage.Double())

def getHist3():
    return hist.Hist(
        getdRAxis("dR1", "$\Delta R_1$"),
        getdRAxis("dR2", "$\Delta R_2$"),
        getdRAxis("dR3", "$\Delta R_3$"),
        storage=hist.storage.Double())

def getHist4():
    return hist.Hist(
        getdRAxis("dR1", "$\Delta R_1$"),
        getdRAxis("dR2", "$\Delta R_2$"),
        getdRAxis("dR3", "$\Delta R_3$"),
        getdRAxis("dR4", "$\Delta R_4$"),
        getdRAxis("dR5", "$\Delta R_5$"),
        getdRAxis("dR6", "$\Delta R_6$"),
        storage=hist.storage.Double())

def getHistPxP():
    return hist.Hist(
        getdRAxis("dRa", "$\Delta R_a$"),
        getdRAxis("dRb", "$\Delta R_b$"),
        getOrderAxis("ordera", "Order a"), 
        getOrderAxis("orderb", "Order b"),
        storage=hist.storage.Double())

#block-diagonal PxP 
def getHistPxP_bdiag():
    return hist.Hist(
        getdRAxis("dRa", "$\Delta R_a$"),
        getdRAxis("dRb", "$\Delta R_b$"),
        getOrderAxis("order", "Order"),
        storage=hist.storage.Double())

def getHist3x3():
    return hist.Hist(
        getdRAxis("dRa1", "$\Delta R_{1a}$"),
        getdRAxis("dRb1", "$\Delta R_{1b}$"),
        getdRAxis("dRa2", "$\Delta R_{2a}$"),
        getdRAxis("dRb2", "$\Delta R_{2b}$"),
        getdRAxis("dRa3", "$\Delta R_{3a}$"),
        getdRAxis("dRb3", "$\Delta R_{3b}$"),
        storage=hist.storage.Double())

def getHist3xP():
    return hist.Hist(
        getdRAxis("dRP", "$\Delta R_P$"),
        getOrderAxis("orderP", "Order P"),
        getdRAxis("dR31", "$\Delta R_{31}$"),
        getdRAxis("dR32", "$\Delta R_{32}$"),
        getdRAxis("dR33", "$\Delta R_{33}$"),
        storage=hist.storage.Double())

#we need to rethink the 4x4
#this is naively order petabytes in memory
#I'm just gonna leave it for now though
#maybe we'll project it down?
def getHist4x4():
    return hist.Hist(
        getdRAxis("dRa1", "$\Delta R_{1a}$"),
        getdRAxis("dRb1", "$\Delta R_{1b}$"),
        getdRAxis("dRa2", "$\Delta R_{2a}$"),
        getdRAxis("dRb2", "$\Delta R_{2b}$"),
        getdRAxis("dRa3", "$\Delta R_{3a}$"),
        getdRAxis("dRb3", "$\Delta R_{3b}$"),
        getdRAxis("dRa4", "$\Delta R_{4a}$"),
        getdRAxis("dRb4", "$\Delta R_{4b}$"),
        getdRAxis("dRa5", "$\Delta R_{5a}$"),
        getdRAxis("dRb5", "$\Delta R_{5b}$"),
        getdRAxis("dRa6", "$\Delta R_{6a}$"),
        getdRAxis("dRb6", "$\Delta R_{6b}$"),
        storage=hist.storage.Double())

def getHist4x3():
    return hist.Hist(
        getdRAxis("dR31", "$\Delta R_{31}$"),
        getdRAxis("dR32", "$\Delta R_{32}$"),
        getdRAxis("dR33", "$\Delta R_{33}$"),
        getdRAxis("dR41", "$\Delta R_{41}$"),
        getdRAxis("dR42", "$\Delta R_{42}$"),
        getdRAxis("dR43", "$\Delta R_{43}$"),
        getdRAxis("dR44", "$\Delta R_{44}$"),
        getdRAxis("dR45", "$\Delta R_{45}$"),
        getdRAxis("dR46", "$\Delta R_{46}$"),
        storage=hist.storage.Double())

def getHist4xP():
    return hist.Hist(
        getdRAxis("dRP", "$\Delta R_P$"),
        getOrderAxis("orderP", "Order P"),
        getdRAxis("dR41", "$\Delta R_{41}$"),
        getdRAxis("dR42", "$\Delta R_{42}$"),
        getdRAxis("dR43", "$\Delta R_{43}$"),
        getdRAxis("dR44", "$\Delta R_{44}$"),
        getdRAxis("dR45", "$\Delta R_{45}$"),
        getdRAxis("dR46", "$\Delta R_{46}$"),
        storage=hist.storage.Double())

def fillHistP(h, r, evtwts=1, mask=None):
    mask = ensure_mask(mask, r.proj)

    dR = r.projdR_forhist[mask]
    order = r.projOrder_forhist[mask]
    wts = (r.proj*evtwts)[mask]
    h.fill(dR = ak.flatten(dR, axis=None), 
           order = ak.flatten(order, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes3(h, r, evtwts=1, mask=None):
    mask = ensure_mask(mask, r.res3.wts)

    dR1 = r.res3.dR1[mask]
    dR2 = r.res3.dR2[mask]
    dR3 = r.res3.dR3[mask]
    wts = (r.res3.wts*evtwts)[mask]
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes4(h, r, evtwts=1, mask=None):
    mask = ensure_mask(mask, r.res4.wts)
    dR1 = r.res4.dR1[mask]
    dR2 = r.res4.dR2[mask]
    dR3 = r.res4.dR3[mask]
    dR4 = r.res4.dR4[mask]
    dR5 = r.res4.dR5[mask]
    dR6 = r.res4.dR6[mask]
    wts = (r.res4.wts*evtwts)[mask]
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           dR4 = ak.flatten(dR4, axis=None),
           dR5 = ak.flatten(dR5, axis=None),
           dR6 = ak.flatten(dR6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCovPxP(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.covPxP)
    dR = r.projdR_forcov[mask]
    order = r.projOrder_forcov[mask]
    wts = (r.covPxP * evtwt)[mask]

    dRa, ordera, _ = ak.broadcast_arrays(dR, order, wts)
    dRb, orderb, _ = ak.broadcast_arrays(dR[:,:,None,:], 
                                         order[:,:,None,:],
                                         wts)

    h.fill(dRa = ak.flatten(dRa, axis=None),
           ordera = ak.flatten(ordera, axis=None),
           dRb = ak.flatten(dRb, axis=None),
           orderb = ak.flatten(orderb, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov3x3(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.cov3x3)

    dR1 = r.res3.dR1[mask]
    dR2 = r.res3.dR2[mask]
    dR3 = r.res3.dR3[mask]
    wts = (r.cov3x3 * evtwt)[mask]

    dRa1, dRa2, dRa3, _ = ak.broadcast_arrays(dR1[:,:,:,None], 
                                              dR2[:,:,:,None], 
                                              dR3[:,:,:,None],
                                              wts)

    dRb1, dRb2, dRb3, _ = ak.broadcast_arrays(dR1[:,:,None, :], 
                                              dR2[:,:,None, :], 
                                              dR3[:,:,None, :],
                                              wts)

    h.fill(dRa1 = ak.flatten(dRa1, axis=None),
           dRa2 = ak.flatten(dRa2, axis=None),
           dRa3 = ak.flatten(dRa3, axis=None),
           dRb1 = ak.flatten(dRb1, axis=None),
           dRb2 = ak.flatten(dRb2, axis=None),
           dRb3 = ak.flatten(dRb3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov3xP(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.cov3xP)

    dRP = r.projdR_forcov[mask]
    orderP = r.projOrder_forcov[mask]

    dR1 = r.res3.dR1[mask]
    dR2 = r.res3.dR2[mask]
    dR3 = r.res3.dR3[mask]

    wts = (r.cov3xP*wts)[mask]

    dRP, orderP = ak.broadcast_arrays(dRP[:,:,None,:], 
                                      orderP[:,:,None,:], 
                                      wts)
    dR1, dR2, dR3 = ak.broadcast_arrays(dR1[:,:,:,None],
                                        dR2[:,:,:,None],
                                        dR3[:,:,:,None],
                                        wts)

    h.fill(dRP = ak.flatten(dRP, axis=None),
           orderP = ak.flatten(orderP, axis=None),
           dR31 = ak.flatten(dR1, axis=None),
           dR32 = ak.flatten(dR2, axis=None),
           dR33 = ak.flatten(dR3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov4x4(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.cov4x4)
    dR1 = r.res4.dR1[mask]
    dR2 = r.res4.dR2[mask]
    dR3 = r.res4.dR3[mask]
    dR4 = r.res4.dR4[mask]
    dR5 = r.res4.dR5[mask]
    dR6 = r.res4.dR6[mask]
    wts = (r.cov4x4 * evtwt)[mask]

    dRa1, dRa2, dRa3, dRa4, dRa5, dRa6, _ = ak.broadcast_arrays(
            dR1[:,:,:,None],
            dR2[:,:,:,None],
            dR3[:,:,:,None],
            dR4[:,:,:,None],
            dR5[:,:,:,None],
            dR6[:,:,:,None],
            wts)
    dRb1, dRb2, dRb3, dRb4, dRb5, dRb6, _ = ak.broadcast_arrays(
            dR1[:,:,None,:],
            dR2[:,:,None,:],
            dR3[:,:,None,:],
            dR4[:,:,None,:],
            dR5[:,:,None,:],
            dR6[:,:,None,:],
            wts)

    h.fill(dRa1 = ak.flatten(dRa1, axis=None),
           dRa2 = ak.flatten(dRa2, axis=None),
           dRa3 = ak.flatten(dRa3, axis=None),
           dRa4 = ak.flatten(dRa4, axis=None),
           dRa5 = ak.flatten(dRa5, axis=None),
           dRa6 = ak.flatten(dRa6, axis=None),
           dRb1 = ak.flatten(dRb1, axis=None),
           dRb2 = ak.flatten(dRb2, axis=None),
           dRb3 = ak.flatten(dRb3, axis=None),
           dRb4 = ak.flatten(dRb4, axis=None),
           dRb5 = ak.flatten(dRb5, axis=None),
           dRb6 = ak.flatten(dRb6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov4x3(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.cov4x3)

    dR41 = r.res4.dR1[mask]
    dR42 = r.res4.dR2[mask]
    dR43 = r.res4.dR3[mask]
    dR44 = r.res4.dR4[mask]
    dR45 = r.res4.dR5[mask]
    dR46 = r.res4.dR6[mask]

    dR31 = r.res3.dR1[mask]
    dR32 = r.res3.dR2[mask]
    dR33 = r.res3.dR3[mask]

    wts = (r.cov4x3 * evtwt)[mask]

    dR41, dR42, dR43, dR44, dR45, dR46, _ = ak.broadcast_arrays(
            dR41[:,:,:,None],
            dR42[:,:,:,None],
            dR43[:,:,:,None],
            dR44[:,:,:,None],
            dR45[:,:,:,None],
            dR46[:,:,:,None],
            wts)

    dR31, dR32, dR33 = ak.broadcast_arrays(
            dR31[:,:,None,:],
            dR32[:,:,None,:],
            dR33[:,:,None,:],
            wts)

    h.fill(dR41 = ak.flatten(dR41, axis=None),
           dR42 = ak.flatten(dR42, axis=None),
           dR43 = ak.flatten(dR43, axis=None),
           dR44 = ak.flatten(dR44, axis=None),
           dR45 = ak.flatten(dR45, axis=None),
           dR46 = ak.flatten(dR46, axis=None),
           dR31 = ak.flatten(dR31, axis=None),
           dR32 = ak.flatten(dR32, axis=None),
           dR33 = ak.flatten(dR33, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov4xP(h, r, evtwt=1, mask=None):
    mask = ensure_mask(mask, r.cov4xP)
    dRP = r.projdR_forcov[mask]
    orderP = r.projOrder_forcov[mask]

    dR1 = r.res4.dR1[mask]
    dR2 = r.res4.dR2[mask]
    dR3 = r.res4.dR3[mask]
    dR4 = r.res4.dR4[mask]
    dR5 = r.res4.dR5[mask]
    dR6 = r.res4.dR6[mask]

    wts = (r.cov4xP * evtwt)[mask]

    dRP, orderP = ak.broadcast_arrays(dRP[:,:,None,:],
                                      orderP[:,:,None,:],
                                      wts)

    dR1, dR2, dR3, dR4, dR5, dR6 = ak.broadcast_arrays(
            dR1[:,:,:,None],
            dR2[:,:,:,None],
            dR3[:,:,:,None],
            dR4[:,:,:,None],
            dR5[:,:,:,None],
            dR6[:,:,:,None],
            wts)

    h.fill(dRP = ak.flatten(dRP, axis=None),
           orderP = ak.flatten(orderP, axis=None),
           dR41 = ak.flatten(dR1, axis=None),
           dR42 = ak.flatten(dR2, axis=None),
           dR43 = ak.flatten(dR3, axis=None),
           dR44 = ak.flatten(dR4, axis=None),
           dR45 = ak.flatten(dR5, axis=None),
           dR46 = ak.flatten(dR6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistTransferP(h, rReco, rGen, rTrans, evtwt=1, mask=None):
    mask = ensure_mask(mask, rTrans.transferP)
    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx
    mask = mask[iJTa]

    dRa = rReco.projdR_forhist[iJTa][mask]
    order = rReco.projOrder_forhist[iJTa][mask]

    dRb = rGen.projdR_forhist[mask] #gen only even gets computed for matched 
                                    #so no need to mask with iJTb (?)


    wts = (rTrans.transferP * evtwt)[mask]

    dRa, order, _ = ak.broadcast_arrays(dRa[:,:,:,None,:],
                                         order[:,:,:,None,:],
                                         wts)
    dRb, _ = ak.broadcast_arrays(dRb, wts)

    h.fill(dRa = ak.flatten(dRa, axis=None),
           order = ak.flatten(order, axis=None),
           dRb = ak.flatten(dRb, axis=None),
           weight = ak.flatten(wts, axis=None))

def fillHistTransfer3(h, rReco, rGen, rTrans, evtwt=1, mask=None):
    mask = ensure_mask(mask, rTrans.transfer3)

    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx
    mask = mask[iJTa]

    dRa1 = rReco.res3.dR1[iJTa][mask]
    dRa2 = rReco.res3.dR2[iJTa][mask]
    dRa3 = rReco.res3.dR3[iJTa][mask]

    dRb1 = rGen.res3.dR1[mask]
    dRb2 = rGen.res3.dR2[mask]
    dRb3 = rGen.res3.dR3[mask]

    wts = (rTrans.transfer3 * evtwt)[mask]

    dRa1, dRa2, dRa3, _ = ak.broadcast_arrays(dRa1[:,:,None,:],
                                              dRa2[:,:,None,:],
                                              dRa3[:,:,None,:],
                                              wts)
    dRb1, dRb2, dRb3, _ = ak.broadcast_arrays(dRb1[:,:,:,None],
                                              dRb2[:,:,:,None],
                                              dRb3[:,:,:,None],
                                              wts)
    h.fill(dRa1 = ak.flatten(dRa1, axis=None),
           dRa2 = ak.flatten(dRa2, axis=None),
           dRa3 = ak.flatten(dRa3, axis=None),
           dRb1 = ak.flatten(dRb1, axis=None),
           dRb2 = ak.flatten(dRb2, axis=None),
           dRb3 = ak.flatten(dRb3, axis=None),
           weight = ak.flatten(wts, axis=None))

def fillHistTransfer4(h, rReco, rGen, rTrans, evtwt=1, mask=None):
    mask = ensure_mask(mask, rTrans.transfer4)

    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx
        
    mask = mask[iJTa]

    dRa1 = rReco.res4.dR1[iJTa][mask]
    dRa2 = rReco.res4.dR2[iJTa][mask]
    dRa3 = rReco.res4.dR3[iJTa][mask]
    dRa4 = rReco.res4.dR4[iJTa][mask]
    dRa5 = rReco.res4.dR5[iJTa][mask]
    dRa6 = rReco.res4.dR6[iJTa][mask]
    
    dRb1 = rGen.res4.dR1[mask]
    dRb2 = rGen.res4.dR2[mask]
    dRb3 = rGen.res4.dR3[mask]
    dRb4 = rGen.res4.dR4[mask]
    dRb5 = rGen.res4.dR5[mask]
    dRb6 = rGen.res4.dR6[mask]

    wts = (rTrans.transfer4 * evtwt)[mask]

    dRa1, dRa2, dRa3, dRa4, dRa5, dRa6, _ = ak.broadcast_arrays(
            dRa1[:,:,None, :],
            dRa2[:,:,None, :],
            dRa3[:,:,None, :],
            dRa4[:,:,None, :],
            dRa5[:,:,None, :],
            dRa6[:,:,None, :],
            wts)

    dRb1, dRb2, dRb3, dRb4, dRb5, dRb6, _ = ak.broadcast_arrays(
            dRb1[:,:,:,None],
            dRb2[:,:,:,None],
            dRb3[:,:,:,None],
            dRb4[:,:,:,None],
            dRb5[:,:,:,None],
            dRb6[:,:,:,None],
            wts)

    h.fill(dRa1 = ak.flatten(dRa1, axis=None),
           dRa2 = ak.flatten(dRa2, axis=None),
           dRa3 = ak.flatten(dRa3, axis=None),
           dRa4 = ak.flatten(dRa4, axis=None),
           dRa5 = ak.flatten(dRa5, axis=None),
           dRa6 = ak.flatten(dRa6, axis=None),
           dRb1 = ak.flatten(dRb1, axis=None),
           dRb2 = ak.flatten(dRb2, axis=None),
           dRb3 = ak.flatten(dRb3, axis=None),
           dRb4 = ak.flatten(dRb4, axis=None),
           dRb5 = ak.flatten(dRb5, axis=None),
           dRb6 = ak.flatten(dRb6, axis=None),
           weight = ak.flatten(wts, axis=None))
