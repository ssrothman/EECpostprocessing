import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

def getdRAxis(name='dR', label='$\Delta R$'):
    return hist.axis.Regular(20, 1e-3, 1.0,
                             name=name, label=label, 
                             transform=hist.axis.transform.log)

def getOrderAxis(name='order', label='Order'):
    return hist.axis.IntCategory([2, 3, 4, 5, 6, -21, -22, -23], 
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

def fillHistP(h, r, evtwts):
    dR = r.projdR_forhist
    order = r.projOrder_forhist
    wts = r.proj*evtwts
    h.fill(dR = ak.flatten(dR, axis=None), 
           order = ak.flatten(order, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes3(h, r, evtwts):
    dR1 = r.res3.dR1
    dR2 = r.res3.dR2
    dR3 = r.res3.dR3
    wts = r.res3.wts*evtwts
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes4(h, r, evtwts):
    dR1 = r.res4.dR1
    dR2 = r.res4.dR2
    dR3 = r.res4.dR3
    dR4 = r.res4.dR4
    dR5 = r.res4.dR5
    dR6 = r.res4.dR6
    wts = r.res4.wts*evtwts
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           dR4 = ak.flatten(dR4, axis=None),
           dR5 = ak.flatten(dR5, axis=None),
           dR6 = ak.flatten(dR6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCovPxP(h, r, evtwt):
    dR = r.projdR_forcov
    order = r.projOrder_forcov
    covPxP = r.covPxP

    dRa, ordera, _ = ak.broadcast_arrays(dR, order, covPxP)
    dRb, orderb, _ = ak.broadcast_arrays(dR[:,:,None,:], 
                                         order[:,:,None,:],
                                         covPxP)
    wts = covPxP * evtwt
    h.fill(dRa = ak.flatten(dRa, axis=None),
           ordera = ak.flatten(ordera, axis=None),
           dRb = ak.flatten(dRb, axis=None),
           orderb = ak.flatten(orderb, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov3x3(h, r, evtwt):
    dR1 = r.res3.dR1
    dR2 = r.res3.dR2
    dR3 = r.res3.dR3
    cov3x3 = r.cov3x3

    dRa1, dRa2, dRa3, _ = ak.broadcast_arrays(dR1[:,:,:,None], 
                                              dR2[:,:,:,None], 
                                              dR3[:,:,:,None],
                                              cov3x3)

    dRb1, dRb2, dRb3, _ = ak.broadcast_arrays(dR1[:,:,None, :], 
                                              dR2[:,:,None, :], 
                                              dR3[:,:,None, :],
                                              cov3x3)

    wts = cov3x3 * evtwt
    h.fill(dRa1 = ak.flatten(dRa1, axis=None),
           dRa2 = ak.flatten(dRa2, axis=None),
           dRa3 = ak.flatten(dRa3, axis=None),
           dRb1 = ak.flatten(dRb1, axis=None),
           dRb2 = ak.flatten(dRb2, axis=None),
           dRb3 = ak.flatten(dRb3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov3xP(h, r, evtwt):
    dRP = r.projdR_forcov
    orderP = r.projOrder_forcov

    dR1 = r.res3.dR1
    dR2 = r.res3.dR2
    dR3 = r.res3.dR3

    cov3xP = r.cov3xP

    dRP, orderP = ak.broadcast_arrays(dRP[:,:,None,:], 
                                      orderP[:,:,None,:], 
                                      cov3xP)
    dR1, dR2, dR3 = ak.broadcast_arrays(dR1[:,:,:,None],
                                        dR2[:,:,:,None],
                                        dR3[:,:,:,None],
                                        cov3xP)
    wts = cov3xP * evtwt
    h.fill(dRP = ak.flatten(dRP, axis=None),
           orderP = ak.flatten(orderP, axis=None),
           dR31 = ak.flatten(dR1, axis=None),
           dR32 = ak.flatten(dR2, axis=None),
           dR33 = ak.flatten(dR3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov4x4(h, r, evtwt):
    dR1 = r.res4.dR1
    dR2 = r.res4.dR2
    dR3 = r.res4.dR3
    dR4 = r.res4.dR4
    dR5 = r.res4.dR5
    dR6 = r.res4.dR6
    cov4x4 = r.cov4x4

    dRa1, dRa2, dRa3, dRa4, dRa5, dRa6, _ = ak.broadcast_arrays(
            dR1[:,:,:,None],
            dR2[:,:,:,None],
            dR3[:,:,:,None],
            dR4[:,:,:,None],
            dR5[:,:,:,None],
            dR6[:,:,:,None],
            cov4x4)
    dRb1, dRb2, dRb3, dRb4, dRb5, dRb6, _ = ak.broadcast_arrays(
            dR1[:,:,None,:],
            dR2[:,:,None,:],
            dR3[:,:,None,:],
            dR4[:,:,None,:],
            dR5[:,:,None,:],
            dR6[:,:,None,:],
            cov4x4)
    wts = cov4x4 * evtwt
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

def fillHistCov4x3(h, r, evtwt):
    dR41 = r.res4.dR1
    dR42 = r.res4.dR2
    dR43 = r.res4.dR3
    dR44 = r.res4.dR4
    dR45 = r.res4.dR5
    dR46 = r.res4.dR6

    dR31 = r.res3.dR1
    dR32 = r.res3.dR2
    dR33 = r.res3.dR3

    cov4x3 = r.cov4x3

    dR41, dR42, dR43, dR44, dR45, dR46, _ = ak.broadcast_arrays(
            dR41[:,:,:,None],
            dR42[:,:,:,None],
            dR43[:,:,:,None],
            dR44[:,:,:,None],
            dR45[:,:,:,None],
            dR46[:,:,:,None],
            cov4x3)

    dR31, dR32, dR33 = ak.broadcast_arrays(
            dR31[:,:,None,:],
            dR32[:,:,None,:],
            dR33[:,:,None,:],
            cov4x3)

    wts = cov4x3 * evtwt

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

def fillHistCov4xP():
    dRP = r.projdR_forcov
    orderP = r.projOrder_forcov

    dR1 = r.res4.dR1
    dR2 = r.res4.dR2
    dR3 = r.res4.dR3
    dR4 = r.res4.dR4
    dR5 = r.res4.dR5
    dR6 = r.res4.dR6

    cov4xP = r.cov4xP

    dRP, orderP = ak.broadcast_arrays(dRP[:,:,None,:],
                                      orderP[:,:,None,:],
                                      cov4xP)

    dR1, dR2, dR3, dR4, dR5, dR6 = ak.broadcast_arrays(
            dR1[:,:,:,None],
            dR2[:,:,:,None],
            dR3[:,:,:,None],
            dR4[:,:,:,None],
            dR5[:,:,:,None],
            dR6[:,:,:,None],
            cov4xP)

    wts = cov4xP * evtwt

    h.fill(dRP = ak.flatten(dRP, axis=None),
           orderP = ak.flatten(orderP, axis=None),
           dR41 = ak.flatten(dR1, axis=None),
           dR42 = ak.flatten(dR2, axis=None),
           dR43 = ak.flatten(dR3, axis=None),
           dR44 = ak.flatten(dR4, axis=None),
           dR45 = ak.flatten(dR5, axis=None),
           dR46 = ak.flatten(dR6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistTransferP(h, rReco, rGen, rTrans, evtwt):
    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx

    dRa = rReco.projdR_forhist[iJTa]
    order = rReco.projOrder_forhist[iJTa]

    dRb = rGen.projdR_forhist[iJTb]


    wts = rTrans.transferP * evtwt

    dRa, order, _ = ak.broadcast_arrays(dRa[:,:,:,None,:],
                                         order[:,:,:,None,:],
                                         wts)
    dRb, _ = ak.broadcast_arrays(dRb, wts)

    h.fill(dRa = ak.flatten(dRa, axis=None),
           order = ak.flatten(order, axis=None),
           dRb = ak.flatten(dRb, axis=None),
           weight = ak.flatten(wts, axis=None))

def fillHistTransfer3(h, rReco, rGen, rTrans, evtwt):
    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx

    dRa1 = rReco.res3.dR1[iJTa]
    dRa2 = rReco.res3.dR2[iJTa]
    dRa3 = rReco.res3.dR3[iJTa]

    dRb1 = rGen.res3.dR1[iJTb]
    dRb2 = rGen.res3.dR2[iJTb]
    dRb3 = rGen.res3.dR3[iJTb]

    wts = rTrans.transfer3 * evtwt

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

def fillHistTransfer4(h, rReco, rGen, rTrans, evtwt):
    iJTa = rTrans.transferRecoIdx
    iJTb = rTrans.transferGenIdx

    dRa1 = rReco.res4.dR1[iJTa]
    dRa2 = rReco.res4.dR2[iJTa]
    dRa3 = rReco.res4.dR3[iJTa]
    dRa4 = rReco.res4.dR4[iJTa]
    dRa5 = rReco.res4.dR5[iJTa]
    dRa6 = rReco.res4.dR6[iJTa]

    dRb1 = rGen.res4.dR1[iJTb]
    dRb2 = rGen.res4.dR2[iJTb]
    dRb3 = rGen.res4.dR3[iJTb]
    dRb4 = rGen.res4.dR4[iJTb]
    dRb5 = rGen.res4.dR5[iJTb]
    dRb6 = rGen.res4.dR6[iJTb]

    wts = rTrans.transfer4 * evtwt

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

def getEECnorm(r, evtwt):
    return ak.sum(ak.num(r.proj, axis=1) * evtwt, axis=None)

def plotProjectedEEC(H, Hcov, order, norm):
    orderIdx = H.axes['order'].index(order)
    Hproj = H[{'order' : orderIdx}].project('dR')
    Hcovproj = Hcov[{'ordera' : orderIdx, 'orderb' : orderIdx}].project('dRa', 'dRb')

    vals = Hproj.values()/norm
    errs = np.diag(Hcovproj.values())/norm
    
    x = H.axes['dR'].centers

    plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(x, vals, yerr=errs, fmt='o', label=f'order {order}')
