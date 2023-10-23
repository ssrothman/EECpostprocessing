import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from util.util import ensure_mask

import reading.reader

def getProjHist(nDR):
    return hist.Hist(
        hist.axis.Regular(10, 0, 500, name='pt', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbin', label='$\Delta R$ bin'),
        storage=hist.storage.Double(),
    )

def binProj(H, rEEC, rJet, nDR, wt, mask=None):
    proj = rEEC.proj
    iReco = rEEC.iReco
    iJet = rEEC.iJet

    #mask = ensure_mask(mask, iReco)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return


    wts, _ = ak.broadcast_arrays(wt, mask)

    pt = rJet.simonjets.pt[iJet][mask]
    vals = (proj * wts)[mask]
    
    dRbin = ak.local_index(vals, axis=2)
    pt, _ = ak.broadcast_arrays(pt, dRbin)

    mask2 = vals > 0

    H.fill(
        pt = ak.flatten(pt[mask2], axis=None),
        dRbin = ak.flatten(dRbin[mask2], axis=None),
        weight = ak.flatten(vals[mask2], axis=None),
    )

def getTransferHistP(nDR):
    return hist.Hist(
        hist.axis.Regular(10, 0, 500, name='ptReco', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinReco', label='$\Delta R$ bin'),
        hist.axis.Regular(10, 0, 500, name='ptGen', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinGen', label='$\Delta R$ bin'),
        storage=hist.storage.Double(),
    )

def binTransferP(H, rTransfer, rRecoJet, rGenJet, nDR, wt, mask=None):
    proj = rTransfer.proj
    iReco = rTransfer.iReco
    iGen = rTransfer.iGen

    mask = ensure_mask(mask, rRecoJet.simonjets.pt)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return

    wts, _ = ak.broadcast_arrays(wt, mask)

    recoPt = rRecoJet.simonjets.pt[iReco]
    genPt = rGenJet.simonjets.pt[iGen]

    proj = proj[mask]

    iGen = ak.local_index(proj, axis=2)
    iReco = ak.local_index(proj, axis=3)

    recoPt, genPt, iGen, _ = ak.broadcast_arrays(recoPt, genPt, iGen, iReco)

    mask2 = proj > 0

    H.fill(
        ptReco = ak.flatten(recoPt[mask2], axis=None),
        dRbinReco = ak.flatten(iReco[mask2], axis=None),
        ptGen = ak.flatten(genPt[mask2], axis=None),
        dRbinGen = ak.flatten(iGen[mask2], axis=None),
        weight = ak.flatten((proj*wts)[mask2], axis=None),
    )

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
    print("\t\tmaking projsums took", time()-t0, "seconds")

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

            mask2 = cov > 0

            dR1 = dR1[mask2]
            dR2 = dR2[mask2]
            cov = cov[mask2]

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
    print("\t\tcovtime = ", covtime)
    print("\t\tfilltime = ", filltime)

def doProjected(x, nameEEC, nameJet, nDR, wt, mask=None):
    Hval = getProjHist(nDR)
    Hcov = getCovHistP(nDR)
    
    rEEC = reading.reader.EECreader(x, nameEEC)
    rJet = reading.reader.jetreader(x, '', nameJet)

    from time import time
    t0 = time()
    binProj(Hval, rEEC, rJet, nDR, wt, mask)
    print("\tbinProj took", time()-t0, "seconds")
    t0 = time()
    binCovP(Hcov, rEEC, rJet, nDR, wt, mask)
    print("\tbinCovP took", time()-t0, "seconds")

    return Hval, Hcov

def doTransfer(x, nameTransfer, nameRecoJet, nameGenJet, nDR, wt, mask=None):
    Htrans = getTransferHistP(nDR)

    rTransfer = reading.reader.transferreader(x, nameTransfer)
    rRecoJet = reading.reader.jetreader(x, '', nameRecoJet)
    rGenJet = reading.reader.jetreader(x, '', nameGenJet)

    binTransferP(Htrans, rTransfer, rRecoJet, rGenJet, nDR, wt, mask)

    return Htrans

def doAll(x, nameTransfer, nameRecoEEC, nameGenEEC, 
          nameRecoJet, nameGenJet, nDR, wt, mask):

    print("top of doALL")
    from time import time
    t0 = time()
    Hreco, HcovReco = doProjected(x, nameRecoEEC, nameRecoJet, nDR, wt, mask)
    print("reco took %0.4f seconds"%(time()-t0))

    t0 = time()
    recoJets = reading.reader.jetreader(x, '', nameRecoJet).parts
    recoEEC = reading.reader.EECreader(x, nameRecoEEC).proj
    recoPUEEC = reading.reader.EECreader(x, nameRecoEEC+"PU").proj
    PUjets = ak.all(recoJets.nmatch == 0, axis=-1)
    print("finding PUjets took %0.4f seconds"%(time()-t0))

    t0 = time()
    HrecoPUjets, HcovRecoPUjets = doProjected(x, nameRecoEEC+"PU", 
                                              nameRecoJet, nDR, wt, 
                                              (mask & PUjets))
    print("recoPUjets took %0.4f seconds"%(time()-t0))
    t0 = time()
    HrecoUNMATCH, HcovRecoUNMATCH = doProjected(x, nameRecoEEC+"PU", 
                                                nameRecoJet, nDR, wt, 
                                                (mask & (~PUjets)))
    print("recoUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    Hgen, HcovGen = doProjected(x, nameGenEEC, nameGenJet, nDR, wt, mask)
    print("gen took %0.4f seconds"%(time()-t0))
    HgenUNMATCH, HcovGenUNMATCH = doProjected(x, nameGenEEC+"UNMATCH", 
                                              nameGenJet, nDR, wt, mask)
    print("genUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    Htrans = doTransfer(x, nameTransfer, nameRecoJet, nameGenJet, nDR, wt, mask)
    print("Htrans took %0.4f seconds"%(time()-t0))

    print("DONE")
    return {
        'Hreco' : Hreco,
        'HrecoPUjets' : HrecoPUjets,
        'HrecoUNMATCH' : HrecoUNMATCH,
        'Hgen' : Hgen,
        'HgenUNMATCH' : HgenUNMATCH,
        'Htrans' : Htrans,
        'HcovReco' : HcovReco,
        'HcovRecoPUjets' : HcovRecoPUjets,
        'HcovRecoUNMATCH' : HcovRecoUNMATCH,
        'HcovGen' : HcovGen,
        'HcovGenUNMATCH' : HcovGenUNMATCH,
    }
    

