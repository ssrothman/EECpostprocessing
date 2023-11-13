import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from util.util import ensure_mask

import reading.reader

etabins = [0, 0.4, 0.9, 1.4, 2.0]

def getProjHist(nDR):
    return hist.Hist(
        hist.axis.Variable(etabins, name='eta', label = 'Jet $|\eta|$'),
        hist.axis.Regular(10, 0, 500, name='pt', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbin', label='$\Delta R$ bin'),
        hist.axis.Variable([0, 20, 30, 40, 50, 80], 
                           name='nPU', label='Number of PU vertices'),
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

    pt = rJet.simonjets.jetPt[iJet][mask]
    vals = (proj * wts)[mask]
    
    dRbin = ak.local_index(vals, axis=2)
    pt, _ = ak.broadcast_arrays(pt, dRbin)

    mask2 = vals > 0

    nPU, _ = ak.broadcast_arrays(rEEC._x.Pileup.nPU, dRbin)

    eta = np.abs(rJet.simonjets.jetEta[iJet][mask])
    eta, _ = ak.broadcast_arrays(eta, dRbin)

    H.fill(
        pt = ak.flatten(pt[mask2], axis=None),
        eta = ak.flatten(eta[mask2], axis=None),
        dRbin = ak.flatten(dRbin[mask2], axis=None),
        nPU = ak.flatten(nPU[mask2], axis=None),
        weight = ak.flatten(vals[mask2], axis=None),
    )

def getTransferHistP(nDR):
    HSR = hist.Hist(
        #hist.axis.Regular(20, 1e-6, 1, name='EECwtReco', label='EEC weight',
        #                  transform=hist.axis.transform.log),
        hist.axis.Variable(etabins, name='eta', label = 'Jet $|\eta|$'),
        hist.axis.Regular(10, 0, 500, name='ptReco', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinReco', label='$\Delta R$ bin'),
        hist.axis.Regular(10, 0, 500, name='ptGen', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinGen', label='$\Delta R$ bin'),
        storage=hist.storage.Double(),
    )
    HSG = hist.Hist(
        #hist.axis.Regular(20, 1e-6, 1, name='EECwtReco', label='EEC weight',
        #                  transform=hist.axis.transform.log),
        hist.axis.Variable(etabins, name='eta', label = 'Jet $|\eta|$'),
        hist.axis.Regular(10, 0, 500, name='ptReco', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinReco', label='$\Delta R$ bin'),
        hist.axis.Regular(10, 0, 500, name='ptGen', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False, 
                          name='dRbinGen', label='$\Delta R$ bin'),
        storage=hist.storage.Double(),
    )
    HFR = hist.Hist(
        hist.axis.Variable(etabins, name='eta', label = 'Jet $|\eta|$'),
        hist.axis.Regular(10, 0, 500, name='ptReco', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False,
                          name='dRbinReco', label='$\Delta R$ bin'),
        #hist.axis.Regular(20, 1e-6, 1, name='EECwtReco', label='EEC weight',
        #                  transform=hist.axis.transform.log),
        storage=hist.storage.WeightedMean(),
    )
    HFG = hist.Hist(
        hist.axis.Variable(etabins, name='eta', label = 'Jet $|\eta|$'),
        hist.axis.Regular(10, 0, 500, name='ptGen', label='Jet $p_T$ [GeV]'),
        hist.axis.Integer(0, nDR,  underflow=False, overflow=False,
                          name='dRbinGen', label='$\Delta R$ bin'),
        #hist.axis.Regular(20, 1e-6, 1, name='EECwtGen', label='EEC weight',
        #                  transform=hist.axis.transform.log),
        storage=hist.storage.WeightedMean(),
    )
    return HSR, HSG, HFR, HFG

def binTransferP(HSR, HSG, HFR, HFG, rTransfer, rGenEEC, rGenEECUNMATCH, 
                 rRecoJet, rGenJet, nDR, wt, mask):
    proj = rTransfer.proj
    iReco = rTransfer.iReco
    iGen = rTransfer.iGen

    mask = ensure_mask(mask, rRecoJet.simonjets.jetPt)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return

    wts, _ = ak.broadcast_arrays(wt, mask)

    recoPt = rRecoJet.simonjets.jetPt[iReco][mask]
    genPt = rGenJet.simonjets.jetPt[iGen][mask]

    recoEta = np.abs(rRecoJet.simonjets.jetEta[iReco][mask])

    genwt = (rGenEEC.proj - rGenEECUNMATCH.proj)[mask]
    genwt_backup = genwt

    proj = proj[mask]

    iDRGen = ak.local_index(proj, axis=2)
    iDRReco = ak.local_index(proj, axis=3)

    wts = wts[mask]
    recoPt, genPt, recoEta, iDRGen, genwt, wts, _ = ak.broadcast_arrays(recoPt, 
                                                               genPt, 
                                                               recoEta,
                                                               iDRGen, 
                                                               genwt, 
                                                               wts, 
                                                               iDRReco)

    mask2 = proj > 0


    reco_bygen = ak.sum(proj, axis=3) #indexed by gen
    gen_bygen = ak.firsts(genwt, axis=3) #indexed by gen
    factor_bygen = reco_bygen/gen_bygen #indexed by gen
    short_dRbinGen = ak.firsts(iDRGen, axis=3)
    
    reco_byreco = ak.sum(proj, axis=2) #indexed by reco
    invfactor = ak.nan_to_num(1/factor_bygen)
    gen_byreco = ak.sum(proj*invfactor, axis=2) #indexed by reco
    factor_byreco = reco_byreco/gen_byreco #indexed by reco
    short_dRbinReco = ak.firsts(iDRReco, axis=2)

    #doesn't matter whether we reduce by reco or gen, 
    #these are at jet level anyway
    short_ptReco = ak.firsts(recoPt, axis=-1)
    short_ptGen = ak.firsts(genPt, axis=-1) 
    short_etaReco = ak.firsts(recoEta, axis=-1)

    mask3 = np.isfinite(factor_byreco)
    HFR.fill(
        eta = ak.flatten(short_etaReco[mask3], axis=None),
        ptReco = ak.flatten(short_ptReco[mask3], axis=None),
        dRbinReco = ak.flatten(short_dRbinReco[mask3], axis=None),
        #EECwtReco = ak.flatten(reco_byreco[mask3], axis=None),
        sample = ak.flatten(factor_byreco[mask3], axis=None),
        weight = ak.flatten(gen_byreco[mask3], axis=None)
    )

    mask4 = np.isfinite(factor_bygen)
    HFG.fill(
        eta = ak.flatten(short_etaReco[mask4], axis=None),
        ptGen = ak.flatten(short_ptGen[mask4], axis=None),
        dRbinGen = ak.flatten(short_dRbinGen[mask4], axis=None),
        #EECwtGen = ak.flatten(gen_bygen[mask4], axis=None),
        sample = ak.flatten(factor_bygen[mask4], axis=None),
        weight = ak.flatten(gen_bygen[mask4], axis=None)
    )
    print()
    print("sum(genbygen):", ak.sum(gen_bygen, axis=None))
    print("sum(genbyreco):", ak.sum(gen_byreco, axis=None))
    print("sum(gen):", ak.sum(genwt_backup, axis=None))
    print()
    print("sum(recobyreco):", ak.sum(reco_byreco, axis=None))
    print("sum(recobygen):", ak.sum(reco_bygen, axis=None))
    print("sum(reco):", ak.sum(proj, axis=None))
    print()

    HSR.fill(
        #EECwtReco = ak.flatten(proj[mask2], axis=None),
        eta = ak.flatten(recoEta[mask2], axis=None),
        ptReco = ak.flatten(recoPt[mask2], axis=None),
        dRbinReco = ak.flatten(iDRReco[mask2], axis=None),
        ptGen = ak.flatten(genPt[mask2], axis=None),
        dRbinGen = ak.flatten(iDRGen[mask2], axis=None),
        weight = ak.flatten((proj*wts)[mask2], axis=None),
    )

    HSG.fill(
        #EECwtReco = ak.flatten(proj[mask2], axis=None),
        eta = ak.flatten(recoEta[mask2], axis=None),
        ptReco = ak.flatten(recoPt[mask2], axis=None),
        dRbinReco = ak.flatten(iDRReco[mask2], axis=None),
        ptGen = ak.flatten(genPt[mask2], axis=None),
        dRbinGen = ak.flatten(iDRGen[mask2], axis=None),
        weight = ak.flatten((proj*invfactor*wts)[mask2], axis=None),
    )

    #denom = ak.sum(proj, axis=-1)
    #factor = denom/genwt
    #denom, _ = ak.broadcast_arrays(denom, factor)

    #HFG.fill(
    #    ptGen = ak.flatten(ak.firsts(genPt[mask2], axis=-1), axis=None),
    #    dRbinGen = ak.flatten(ak.firsts(iDRGen[mask2], axis=-1), axis=None),
    #    EECwtGen = ak.flatten(ak.firsts(genwt[mask2], axis=-1), axis=None),
    #    sample = ak.flatten(ak.firsts(factor[mask2], axis=-1), axis=None),
    #    weight = ak.flatten(ak.firsts(wts[mask2]*genwt[mask2], axis=-1), axis=None)
    #)

    #plt.hist([ak.flatten(factor_byreco, axis=None), ak.flatten(factor_bygen, axis=None)], bins=100, range=[0,3], histtype='step')
    #plt.show()

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

    mask = ensure_mask(mask, rJet.simonjets.jetPt)
    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return

    wts, _ = ak.broadcast_arrays(wt, mask)

    pt = rJet.simonjets.jetPt[iJet][mask]
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
                pt1 = np.ones(len(np.ravel(dR2)))*H.axes['pt1'].value(idxvals[pt1]),
                dRbin1 = np.ravel(dR1),
                pt2 = np.ones(len(np.ravel(dR2)))*H.axes['pt2'].value(idxvals[pt2]),
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

def doProjected(x, nameEEC, config, gen, wt, mask):
    Hval = getProjHist(config.nDR)
    Hcov = getCovHistP(config.nDR)
    
    rEEC = reading.reader.EECreader(x, nameEEC)
    if gen:
        rJet = reading.reader.jetreader(x, None, config.names.gensimonjets, None)
    else:
        rJet = reading.reader.jetreader(x, None, config.names.simonjets, None)

    from time import time
    t0 = time()
    binProj(Hval, rEEC, rJet, config.nDR, wt, mask)
    print("\tbinProj took", time()-t0, "seconds")
    t0 = time()
    binCovP(Hcov, rEEC, rJet, config.nDR, wt, mask)
    print("\tbinCovP took", time()-t0, "seconds")

    return Hval, Hcov

def doTransfer(x, EECname, config, wt, mask):
    HtransSR, HtransSG, HtransFR, HtransFG = getTransferHistP(config.nDR)

    rTransfer = reading.reader.transferreader(x, "%sTransfer"%EECname)
    rRecoJet = reading.reader.jetreader(x, None, config.names.simonjets, None)
    rGenJet = reading.reader.jetreader(x, None, config.names.gensimonjets, None)

    rGenEEC = reading.reader.EECreader(x, 'Gen%s'%EECname)
    rGenEECUNMATCH = reading.reader.EECreader(x, "Gen%sUNMATCH"%EECname)

    binTransferP(HtransSR, HtransSG, HtransFR, HtransFG, 
                 rTransfer, rGenEEC, rGenEECUNMATCH, 
                 rRecoJet, rGenJet, config.nDR, wt, mask)

    return HtransSR, HtransSG, HtransFR, HtransFG

def doAll(x, EECname, MatchName, config, wt, mask):
    print("top of doALL")
    from time import time
    t0 = time()
    Hreco, HcovReco = doProjected(x, "Reco%s"%EECname, config, False, wt, mask)
    print("reco took %0.4f seconds"%(time()-t0))

    t0 = time()
    #recoJets = reading.reader.jetreader(x, '', x.names.).parts
    #recoEEC = reading.reader.EECreader(x, nameRecoEEC).proj
    #recoPUEEC = reading.reader.EECreader(x, nameRecoEEC+"PU").proj
    #PUjets = ak.all(recoJets.nmatch == 0, axis=-1)
    matchedjets = reading.reader.jetreader(x, None, '%sParticles'%MatchName, None)
    PUjets = matchedjets.simonjets.genPt < 0
    print("finding PUjets took %0.4f seconds"%(time()-t0))

    t0 = time()
    HrecoPUjets, HcovRecoPUjets = doProjected(x, "Reco%sPU"%EECname, config, 
                                              False, wt, (mask & PUjets))
    print("recoPUjets took %0.4f seconds"%(time()-t0))
    t0 = time()
    HrecoUNMATCH, HcovRecoUNMATCH = doProjected(x, "Reco%sPU"%EECname, config,
                                                False, wt, (mask & (~PUjets)))
    print("recoUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    Hgen, HcovGen = doProjected(x, "Gen%s"%EECname, config, True, wt, mask)
    print("gen took %0.4f seconds"%(time()-t0))
    HgenUNMATCH, HcovGenUNMATCH = doProjected(x, "Gen%sUNMATCH"%EECname, config, 
                                              True, wt, mask)
    print("genUNMATCH took %0.4f seconds"%(time()-t0))

    t0 = time()
    HtransSR, HtransSG, HtransFR, HtransFG = doTransfer(
            x, EECname, config, 
            wt, mask)
    print("Htrans took %0.4f seconds"%(time()-t0))

    print("DONE")
    return {
        'Hreco' : Hreco,
        'HrecoPUjets' : HrecoPUjets,
        'HrecoUNMATCH' : HrecoUNMATCH,
        'Hgen' : Hgen,
        'HgenUNMATCH' : HgenUNMATCH,
        'HtransSR' : HtransSR,
        'HtransSG' : HtransSG,
        'HtransFR' : HtransFR,
        'HtransFG' : HtransFG,
        'HcovReco' : HcovReco,
        'HcovRecoPUjets' : HcovRecoPUjets,
        'HcovRecoUNMATCH' : HcovRecoUNMATCH,
        'HcovGen' : HcovGen,
        'HcovGenUNMATCH' : HcovGenUNMATCH,
    }
    

