import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from util.util import ensure_mask

import reading.reader

nBinWT = 50
minwt = 1e-6

def getProjHist(nDR):
    return hist.Hist(
        hist.axis.Regular(10, 0, 500, name='pt', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbin', label='$\Delta R$ bin'),
        hist.axis.Regular(nBinWT, minwt, 1, name='EECwt', label='EEC weight',
                          transform=hist.axis.transform.log),
        storage=hist.storage.Double(),
    )

def binProj(H, rEEC, rJet, nDR, wt, mask=None, minus=None):
    proj = rEEC.proj
    if minus is not None:
        proj = proj - minus.proj
    iReco = rEEC.iReco
    iJet = rEEC.iJet

    #mask = ensure_mask(mask, iReco)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return

    wts, _ = ak.broadcast_arrays(wt, mask)

    pt = rJet.simonjets.pt[iJet][mask]
    vals = proj[mask]

    wts = wts[mask]
    
    dRbin = ak.local_index(vals, axis=2)
    pt, wts, _ = ak.broadcast_arrays(pt, wts, dRbin)

    mask2 = vals>0
    pt = ak.flatten(pt[mask2], axis=None)
    dRbin = ak.flatten(dRbin[mask2], axis=None)
    vals = ak.flatten(vals[mask2], axis=None)
    wts = ak.flatten(wts[mask2], axis=None)

    H.fill(
        pt = pt,
        dRbin = dRbin,
        EECwt = vals,
        weight = wts,
    )

def getTransferHistP(nDR):
    return hist.Hist(
        hist.axis.Regular(10, 0, 500, name='ptReco', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinReco', label='$\Delta R$ bin'),
        hist.axis.Regular(nBinWT,minwt,1,name='EECwtReco', label='EEC weight',
                          transform=hist.axis.transform.log),
        hist.axis.Regular(10, 0, 500, name='ptGen', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinGen', label='$\Delta R$ bin'),
        hist.axis.Regular(nBinWT,minwt,1,name='EECwtGen',label='EEC weight',
                          transform=hist.axis.transform.log),
        storage=hist.storage.Double(),
    )

def binTransferP(H, rTransfer, rRecoEEC, rRecoEECPU, rGenEEC, rGenEECUNMATCH,
                 rRecoJet, rGenJet, nDR, wt, mask=None, includeInefficiency=True):

    print(rTransfer._name)
    print(rRecoEEC._name)
    print(rRecoEECPU._name)
    print(rGenEEC._name)
    print(rGenEECUNMATCH._name)
    print(rRecoJet._simonjetsname)
    print(rGenJet._simonjetsname)
    print(includeInefficiency)

    transferval = rTransfer.proj
    iReco = rTransfer.iReco
    iGen = rTransfer.iGen

    if not includeInefficiency:
        gen = rGenEEC.proj - rGenEECUNMATCH.proj
    else:
        gen = rGenEEC.proj
    reco = rRecoEEC.proj - rRecoEECPU.proj
    print("INITIAL RECO SUM", ak.sum(reco))

    mask = ensure_mask(mask, rRecoJet.simonjets.pt)


    mask = mask[iReco]

    if(ak.sum(mask)==0):
        return

    reco = reco[iReco][mask]
    gen = gen[mask] #we only compute EEC for matched gen jets
                    #so this is already aligned with the transfers

    print("--"*20)
    print("max diff", ak.max(ak.sum(transferval, axis=2) - reco))
    print("min diff", ak.min(ak.sum(transferval, axis=2) - reco))
    print("--"*20)
    recobackup = reco[:]

    wts, _ = ak.broadcast_arrays(wt, mask)
    wts = wts[mask]

    recoPt = rRecoJet.simonjets.pt[iReco]
    genPt = rGenJet.simonjets.pt[iGen]

    recoPt = recoPt[mask]
    genPt = genPt[mask]
    transferval = transferval[mask]
    print("masked reco sum", ak.sum(reco))

    iDRGen = ak.local_index(transferval, axis=2)
    iDRReco = ak.local_index(transferval, axis=3)
    genwt = gen[iDRGen]
    recowt = reco[iDRReco[:,:,0,:]]

    #return recoPt, genPt, iDRGen, genwt, recowt, wts, iDRReco

    recoPt, genPt, iDRGen, genwt, recowt, wts, _ = ak.broadcast_arrays(
                                                        recoPt, genPt, 
                                                        iDRGen[:,:,:,None], 
                                                        genwt[:,:,:,None], 
                                                        recowt[:,:,None,:], wts,
                                                        iDRReco)
    mask2 = (transferval>0) & (recowt>0) #&(genwt>0)

    H.fill(
        ptReco = ak.flatten(recoPt[mask2], axis=None),
        dRbinReco = ak.flatten(iDRReco[mask2], axis=None),
        EECwtReco = ak.flatten(recowt[mask2], axis=None),
        ptGen = ak.flatten(genPt[mask2], axis=None),
        dRbinGen = ak.flatten(iDRGen[mask2], axis=None),
        EECwtGen = ak.flatten(genwt[mask2], axis=None),
        weight = ak.flatten(wts[mask2]*transferval[mask2]/recowt[mask2], axis=None),
    )

    wt = wts*transferval/recowt
    print("--"*20)
    print("max diff", ak.max(ak.sum(transferval, axis=2) - recobackup))
    print("min diff", ak.min(ak.sum(transferval, axis=2) - recobackup))
    print("--"*20)
    diff2 = ak.sum(recowt[mask2]*wt[mask2], axis=None) - ak.sum(recobackup, axis=None)
    print("diff2", diff2)
    print("--"*20)
    dRbin = 0
    wtbin = 29
    minwt = H.axes['EECwtReco'].edges[wtbin]
    maxwt = H.axes['EECwtReco'].edges[wtbin+1]
    wtmask = (recowt>minwt) & (recowt<=maxwt)
    dRmask = iDRReco == dRbin
    print("SUM IN BIN", ak.sum(wt[wtmask & dRmask]))
    print("pts", ak.flatten(recoPt[wtmask & dRmask & (wt!=0)], axis=None))
    print("genPts", ak.flatten(genPt[wtmask & dRmask & (wt!=0)], axis=None))
    print("genIDR", ak.flatten(iDRGen[wtmask & dRmask & (wt!=0)], axis=None))
    print("genWTbin", H.axes['EECwtGen'].index(ak.to_numpy(ak.flatten(genwt[wtmask & dRmask & (wt!=0)], axis=None))))
    print("genWT", ak.flatten(genwt[wtmask & dRmask & (wt!=0)], axis=None))
    fullmask = wtmask & dRmask & (wt!=0) 
    print("run", rTransfer._x.run[ak.where(ak.any(ak.any(ak.any(fullmask, axis=-1), axis=-1),axis=-1))])
    print("lumi", rTransfer._x.luminosityBlock[ak.where(ak.any(ak.any(ak.any(fullmask, axis=-1), axis=-1),axis=-1))])
    print("event", rTransfer._x.event[ak.where(ak.any(ak.any(ak.any(fullmask, axis=-1), axis=-1),axis=-1))])
    print("event", ak.where(ak.any(ak.any(ak.any(fullmask, axis=-1), axis=-1),axis=-1)))
    #print("jet", ak.where(ak.any(ak.any(fullmask, axis=-1), axis=-1)))
    wtmask2 = (recobackup>minwt) & (recobackup<=maxwt)
    print("TARGET VAL", ak.sum(wtmask2[:,:,dRbin]))
    print("In the histogram...", np.sum(H.values(flow=True)[1,0,30,:,:,:]))
    
def getCovHistP(nDR):
    return hist.Hist(
        hist.axis.Regular(10, 0, 500, name='pt1', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False,
                          name='dRbin1', label='$\Delta R$ bin'),
        hist.axis.Regular(10, 0, 500, name='pt2', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False,
                          name='dRbin2', label='$\Delta R$ bin'),
        storage=hist.storage.Double(),
    )

def binCovP(H, rEEC, rJet, nDR, wt, mask=None):
    from time import time
    proj = rEEC.proj
    iJet = rEEC.iJet
    iReco = rEEC.iReco

    mask = ensure_mask(mask, rJet.simonjets.pt)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return

    wts, _ = ak.broadcast_arrays(wt, mask)

    pt = rJet.simonjets.pt[iJet][mask]
    numPt = ak.num(pt)
    ptIdx = H.axes['pt1'].index(ak.flatten(pt, axis=None))
    idxvals = np.unique(ptIdx)
    centers = H.axes['pt1'].centers
    ptIdx = ak.unflatten(ptIdx, numPt)

    proj = proj[mask] * wts[mask]

    t0 = time()
    projsums = []
    for pt in idxvals:
        s = ak.sum(proj[ptIdx==pt], axis=1)
        s = s[ak.num(proj)>0]
        size = ak.max(ak.num(s), axis=None)
        s = ak.fill_none(ak.pad_none(s, size, axis=-1, clip=True), 0)
        projsums.append(ak.to_numpy(s))
    #print("\t\tmaking projsums took", time()-t0, "seconds")

    covtime = 0
    filltime = 0
    for pt1 in range(len(projsums)):
        for pt2 in range(len(projsums)):
            t0 = time()
            cov = np.einsum('ij,ik->ijk', projsums[pt1], projsums[pt2])
            covtime += time()-t0
            t0 = time()
            indices = np.indices(cov.shape)
            dR1 = indices[1]
            dR2 = indices[2]

            H.fill(
                pt1 = np.ones(len(np.ravel(dR2)))*H.axes['pt1'].value(pt1),
                dRbin1 = np.ravel(dR1),
                pt2 = np.ones(len(np.ravel(dR2)))*H.axes['pt2'].value(pt2),
                dRbin2 = np.ravel(dR2),
                weight = np.ravel(cov),
            )
            #for dR1 in range(nDR):
            #    for dR2 in range(nDR):
            #        H.fill(
            #            pt1 = np.ones(cov.shape[0])*H.axes['pt1'].value(pt1),
            #            dRbin1 = np.ones(cov.shape[0])*dR1,
            #            pt2 = np.ones(cov.shape[0])*H.axes['pt2'].value(pt2),
            #            dRbin2 = np.ones(cov.shape[0])*dR2,
            #            weight = cov[:,dR1,dR2],
            #            threads=1,
            #        )
            filltime += time()-t0
    #print("\t\tcovtime = ", covtime)
    #print("\t\tfilltime = ", filltime)

def doProjected(x, nameEEC, nameJet, nDR, wt, mask=None, minus=None):
    Hval = getProjHist(nDR)
    Hcov = getCovHistP(nDR)
    
    rEEC = reading.reader.EECreader(x, nameEEC)
    rJet = reading.reader.jetreader(x, '', nameJet)

    if minus is None:
        rMinus = None
    else:
        rMinus = reading.reader.EECreader(x, minus)

    from time import time
    t0 = time()
    binProj(Hval, rEEC, rJet, nDR, wt, mask, minus=rMinus)
    #print("\tbinProj took", time()-t0, "seconds")
    t0 = time()
    #binCovP(Hcov, rEEC, rJet, nDR, wt, mask)
    #print("\tbinCovP took", time()-t0, "seconds")

    return Hval, Hcov

def doTransfer(x, nameTransfer, nameRecoEEC, nameGenEEC, nameRecoJet, nameGenJet, 
               nDR, wt, mask=None, includeInefficiency=False):
    Htrans = getTransferHistP(nDR)

    rTransfer = reading.reader.transferreader(x, nameTransfer)
    rGenEEC = reading.reader.EECreader(x, nameGenEEC)
    rGenEECUNMATCH = reading.reader.EECreader(x, nameGenEEC+"UNMATCH")
    rRecoEEC = reading.reader.EECreader(x, nameRecoEEC)
    rRecoEECPU = reading.reader.EECreader(x, nameRecoEEC+"PU")
    rRecoJet = reading.reader.jetreader(x, '', nameRecoJet)
    rGenJet = reading.reader.jetreader(x, '', nameGenJet)

    binTransferP(Htrans, rTransfer, rRecoEEC, rRecoEECPU, rGenEEC, rGenEECUNMATCH,
                 rRecoJet, rGenJet, nDR, wt, mask, includeInefficiency)

    return Htrans

def doAll(x, nameTransfer, nameRecoEEC, nameGenEEC, 
          nameRecoJet, nameGenJet, nDR, wt, mask,
          includeInefficiency):

    print("top of doALL")
    from time import time
    t0 = time()
    Hreco, HcovReco = doProjected(x, nameRecoEEC, nameRecoJet, nDR, wt, mask)
    #print("reco took %0.4f seconds"%(time()-t0))
    HrecoPure, HcovRecoPure = doProjected(x, nameRecoEEC, nameRecoJet, nDR, wt, mask, 
                                          minus = nameRecoEEC+"PU")

    t0 = time()
    recoJets = reading.reader.jetreader(x, '', nameRecoJet).parts
    recoEEC = reading.reader.EECreader(x, nameRecoEEC).proj
    recoPUEEC = reading.reader.EECreader(x, nameRecoEEC+"PU").proj
    PUjets = ak.all(recoJets.nmatch == 0, axis=-1)
    #print("finding PUjets took %0.4f seconds"%(time()-t0))

    t0 = time()
    HrecoPUjets, HcovRecoPUjets = doProjected(x, nameRecoEEC+"PU", 
                                              nameRecoJet, nDR, wt, 
                                              (mask & PUjets))
    #print("recoPUjets took %0.4f seconds"%(time()-t0))
    t0 = time()
    HrecoUNMATCH, HcovRecoUNMATCH = doProjected(x, nameRecoEEC+"PU", 
                                                nameRecoJet, nDR, wt, 
                                                (mask & (~PUjets)))
    #print("recoUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    Hgen, HcovGen = doProjected(x, nameGenEEC, nameGenJet, nDR, wt, mask)
    #print("gen took %0.4f seconds"%(time()-t0))
    HgenPure, HcovGenPure = doProjected(x, nameGenEEC, nameGenJet, nDR, wt, mask,
                                        minus = nameGenEEC+"UNMATCH")
    t0 = time()
    HgenUNMATCH, HcovGenUNMATCH = doProjected(x, nameGenEEC+"UNMATCH", 
                                              nameGenJet, nDR, wt, mask)
    #print("genUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    Htrans = doTransfer(x, nameTransfer, nameRecoEEC, nameGenEEC, nameRecoJet, nameGenJet, nDR, wt, mask, includeInefficiency)
    #print("Htrans took %0.4f seconds"%(time()-t0))

    print("DONE")
    return {
        'Hreco' : Hreco,
        'HrecoPure' : HrecoPure,
        'HrecoPUjets' : HrecoPUjets,
        'HrecoUNMATCH' : HrecoUNMATCH,
        'Hgen' : Hgen,
        'HgenPure' : HgenPure,
        'HgenUNMATCH' : HgenUNMATCH,
        'Htrans' : Htrans,
        'HcovReco' : HcovReco,
        'HcovRecoPure' : HcovRecoPure,
        'HcovRecoPUjets' : HcovRecoPUjets,
        'HcovRecoUNMATCH' : HcovRecoUNMATCH,
        'HcovGen' : HcovGen,
        'HcovGenPure' : HcovGenPure,
        'HcovGenUNMATCH' : HcovGenUNMATCH,
    }
    

