import hist
import awkward as ak
import numpy as np

def getdRAxis(name='dR', label='$\Delta R$'):
    return hist.axis.Regular(10, 1e-5, 1.0,
                             name=name, label=label, 
                             transform=hist.axis.transform.log)

def getOrderAxis(name='order', label='Order'):
    return hist.axis.Integer(0, 8, 
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

def getHistCovPxP():
    return hist.Hist(
        getdRAxis("dRa", "$\Delta R_a$"),
        getdRAxis("dRb", "$\Delta R_b$"),
        getOrderAxis("ordera", "Order a"), 
        getOrderAxis("orderb", "Order b"),
        storage=hist.storage.Double())

def getHistCov3x3():
    return hist.Hist(
        getdRAxis("dR1a", "$\Delta R_{1a}$"),
        getdRAxis("dR1b", "$\Delta R_{1b}$"),
        getdRAxis("dR2a", "$\Delta R_{2a}$"),
        getdRAxis("dR2b", "$\Delta R_{2b}$"),
        getdRAxis("dR3a", "$\Delta R_{3a}$"),
        getdRAxis("dR3b", "$\Delta R_{3b}$"),
        storage=hist.storage.Double())

def getHistCov4x4():
    return hist.Hist(
        getdRAxis("dR1a", "$\Delta R_{1a}$"),
        getdRAxis("dR1b", "$\Delta R_{1b}$"),
        getdRAxis("dR2a", "$\Delta R_{2a}$"),
        getdRAxis("dR2b", "$\Delta R_{2b}$"),
        getdRAxis("dR3a", "$\Delta R_{3a}$"),
        getdRAxis("dR3b", "$\Delta R_{3b}$"),
        getdRAxis("dR4a", "$\Delta R_{4a}$"),
        getdRAxis("dR4b", "$\Delta R_{4b}$"),
        #getdRAxis("dR5a", "$\Delta R_{5a}$"),
        #getdRAxis("dR5b", "$\Delta R_{5b}$"),
        #getdRAxis("dR6a", "$\Delta R_{6a}$"),
        #getdRAxis("dR6b", "$\Delta R_{6b}$"),
        storage=hist.storage.Double())

def fillHistP(h, dRs, wts, evtwts):
    dR, _ = ak.broadcast_arrays(dRs[:, :, None, :], wts)
    order = ak.local_index(wts, axis=2)
    order, _ = ak.broadcast_arrays(order, wts)
    wts = wts*evtwts
    h.fill(dR = ak.flatten(dR, axis=None), 
           order = ak.flatten(order, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes3(h, res, evtwts):
    dR1 = res.dR1
    dR2 = res.dR2
    dR3 = res.dR3
    wts = res.wts*evtwts
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistRes4(h, res, evtwts):
    dR1 = res.dR1
    dR2 = res.dR2
    dR3 = res.dR3
    dR4 = res.dR4
    dR5 = res.dR5
    dR6 = res.dR6
    wts = res.wts*evtwts
    h.fill(dR1 = ak.flatten(dR1, axis=None),
           dR2 = ak.flatten(dR2, axis=None),
           dR3 = ak.flatten(dR3, axis=None),
           dR4 = ak.flatten(dR4, axis=None),
           dR5 = ak.flatten(dR5, axis=None),
           dR6 = ak.flatten(dR6, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCovPxP(h, covPxP, dRs, evtwt, proj):
    dR, _ = ak.broadcast_arrays(dRs[:, :, None, :], proj)
    dR = ak.flatten(dR, axis=-1)

    order = ak.local_index(proj, axis=2)
    order, _ = ak.broadcast_arrays(order, proj)
    order = ak.flatten(order, axis=-1)

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

def fillHistCov3x3(h, cov3x3, res3, evtwt):
    dR1a, dR2a, dR3a, _ = ak.broadcast_arrays(res3.dR1[:,:,:,None], 
                                              res3.dR2[:,:,:,None], 
                                              res3.dR3[:,:,:,None],
                                              cov3x3)

    dR1b, dR2b, dR3b, _ = ak.broadcast_arrays(res3.dR1[:,:,None, :], 
                                              res3.dR2[:,:,None, :], 
                                              res3.dR3[:,:,None, :],
                                              cov3x3)

    wts = cov3x3 * evtwt
    h.fill(dR1a = ak.flatten(dR1a, axis=None),
           dR2a = ak.flatten(dR2a, axis=None),
           dR3a = ak.flatten(dR3a, axis=None),
           dR1b = ak.flatten(dR1b, axis=None),
           dR2b = ak.flatten(dR2b, axis=None),
           dR3b = ak.flatten(dR3b, axis=None),
           weight=ak.flatten(wts, axis=None))

def fillHistCov4x4(h, cov4x4, res4, evtwt):
    dR1a, dR2a, dR3a, dR4a, dR5a, dR6a, _ = ak.broadcast_arrays(
            res4.dR1[:,:,:,None],
            res4.dR2[:,:,:,None],
            res4.dR3[:,:,:,None],
            res4.dR4[:,:,:,None],
            res4.dR5[:,:,:,None],
            res4.dR6[:,:,:,None],
            cov4x4)
    dR1b, dR2b, dR3b, dR4b, dR5b, dR6b, _ = ak.broadcast_arrays(
            res4.dR1[:,:,None,:],
            res4.dR2[:,:,None,:],
            res4.dR3[:,:,None,:],
            res4.dR4[:,:,None,:],
            res4.dR5[:,:,None,:],
            res4.dR6[:,:,None,:],
            cov4x4)
    wts = cov4x4 * evtwt
    h.fill(dR1a = ak.flatten(dR1a, axis=None),
           dR2a = ak.flatten(dR2a, axis=None),
           dR3a = ak.flatten(dR3a, axis=None),
           dR4a = ak.flatten(dR4a, axis=None),
           #dR5a = ak.flatten(dR5a, axis=None),
           #dR6a = ak.flatten(dR6a, axis=None),
           dR1b = ak.flatten(dR1b, axis=None),
           dR2b = ak.flatten(dR2b, axis=None),
           dR3b = ak.flatten(dR3b, axis=None),
           dR4b = ak.flatten(dR4b, axis=None),
           #dR5b = ak.flatten(dR5b, axis=None),
           #dR6b = ak.flatten(dR6b, axis=None),
           weight=ak.flatten(wts, axis=None))
