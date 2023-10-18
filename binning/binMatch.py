import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt
import reading.reader as reader

from binning.util import *
from util.util import ensure_mask

def getResHists(minPt, maxPt, etaBins, detarange, dptrange):
    return {
        'dpt' : getdptHist(minPt, maxPt, etaBins, dptrange, 51),
        'deta' : getdetaHist(minPt, maxPt, etaBins, detarange, 51),
        'dphi' : getdphiHist(minPt, maxPt, etaBins, detarange, 51),
        'dR' : getdRHist(minPt, maxPt, etaBins, detarange*2, 51)
    }

def getdptHist(minPt, maxPt, etaBins, dptrange, bins):
    return hist.Hist(
        getLinAxis('pt', '$p_T$ [GeV]', minval=minPt, maxval=maxPt, bins=25),
        getVariableAxis('eta', '$|\eta|$', vals=etaBins),
        getLinAxis('dpt', '$\Delta p_T$ [%]', minval=-dptrange, maxval=dptrange, bins=bins),
        storage=hist.storage.Weight(),
    )

def getdetaHist(minPt, maxPt, etaBins, detarange, bins):
    return hist.Hist(
        getLinAxis('pt', '$p_T$ [GeV]', minval=minPt, maxval=maxPt, bins=25),
        getVariableAxis('eta', '$|\eta|$', vals=etaBins),
        getLinAxis('deta', '$\Delta \eta$', minval=-detarange, maxval=detarange, bins=bins),
        storage=hist.storage.Weight(),
    )

def getdphiHist(minPt, maxPt, etaBins, dphirange, bins):
    return hist.Hist(
        getLinAxis('pt', '$p_T$ [GeV]', minval=minPt, maxval=maxPt, bins=25),
        getVariableAxis('eta', '$|\eta|$', vals=etaBins),
        getLinAxis('dphi', '$\Delta \phi$', minval=-dphirange, maxval=dphirange, bins=bins),
        storage=hist.storage.Weight(),
    )

def getdRHist(minPt, maxPt, etaBins, drange, bins):
    return hist.Hist(
        getLinAxis('pt', '$p_T$ [GeV]', minval=minPt, maxval=maxPt, bins=25),
        getVariableAxis('eta', '$|\eta|$', vals=etaBins),
        getLinAxis('dR', '$\Delta R$', minval=0, maxval=drange, bins=bins),
        storage=hist.storage.Weight(),
    )

def getEM0Hists():
    return getResHists(0.0, 100, [0, 1.1, 1.6, 3.0], 0.075, 30)

def getHAD0Hists():
    return getResHists(0.0, 100, [0, 1.3, 1.74, 3.0], 0.2, 100)

def getHADCHHists():
    return getResHists(0.0, 100, [0, 0.9, 1.4, 3.0], 0.003, 5)

def getELEHists():
    return getResHists(0.0, 100, [0, 1.1, 1.6, 3.0], 0.003, 10)

def getMUHists():
    return getResHists(0.0, 100, [0, 1.1, 1.6, 3.0], 0.003, 5)

def getMatchHist():
    return hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=0.1, maxval=100, bins=50),
        getLogAxis('jetpt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=10),
        getLinAxis('ptfrac', '$p_{T}/p_{T,Jet}$ fraction', minval=0, maxval=1, bins=10),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.5]),
        getCatAxis('pdgid', 'Particle ID', cats=[11, 13, 22, 130, 211]),
        getIntAxis('nmatch', 'Number of Matches', minval=0, maxval=3),
        getLinAxis('dRjet', '$\Delta R$ to jet axis', minval=0, maxval=0.5, bins=5),
        storage = hist.storage.Weight(),
    )

def getJetHists():
    ans = {}
    ptbins = 10
    varbins = 101
    varrange=1

    ans['jec'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.0]),
        getLinAxis('jec', 'JEC_{cmssw}', minval=0.7, maxval=1.3, bins=varbins),
        getLinAxis('trueJEC', '$p_{T,gen}/p_{T,raw}$', minval=0.7, maxval=1.3, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    ans['puTerm'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.0]),
        getLinAxis('puTerm', '$1-p_{T,PU}/p_{T,raw}$', minval=-varrange, maxval=varrange, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    ans['responseTerm'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.0]),
        getLinAxis('responseTerm', '$(p_{T,matched}^{Reco}/p_{T,matched}^{Gen})^{-1}$', minval=-varrange, maxval=varrange, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    ans['unrecoTerm'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.5]),
        getLinAxis('unrecoTerm', '$(1-p_{T,unreco}/p_{T,gen})^{-1}$', minval=-varrange, maxval=varrange, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    ans['jecPred'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.5]),
        getLinAxis('jecPred', '$log(JEC)$', minval=-varrange, maxval=varrange, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    ans['jecRatio'] = hist.Hist(
        getLogAxis('pt', '$p_T$ [GeV]', minval=30, maxval=1000, bins=ptbins),
        getVariableAxis('eta', '$|\eta|$', vals=[0, 0.9, 1.4, 2.5]),
        getLinAxis('jecRatio', '$p_T^{Reco}/p_T^{Gen}$', minval=0.0, maxval=2.0, bins=varbins),
        getIntAxis('PU', 'PU', minval=0, maxval=1),
        storage = hist.storage.Weight(),
    )
    return ans

def fillJetHists(H, jetreader, wt=1, mask=None):
    parts = jetreader.parts
    #jets = jetreader.jets
    simonjets = jetreader.simonjets
    
    mask = ensure_mask(mask, simonjets.pt)
    
    wt, _ = ak.broadcast_arrays(wt, simonjets.pt)

    PU = ak.all(parts.nmatch == 0, axis=-1)

    jec_actual = simonjets.genPt / simonjets.rawPt
    jec_cmssw = simonjets.pt / simonjets.rawPt

    matchedPt_gen = ak.sum(parts.matchPt, axis=-1)
    matchedPt_reco = ak.sum(parts.pt[parts.nmatch>0], axis=-1)

    puTerm = np.log(matchedPt_reco/simonjets.rawPt)
    responseTerm = np.log(matchedPt_gen/matchedPt_reco)
    unrecoTerm = np.log(simonjets.genPt/matchedPt_gen)

    jec_simon = puTerm + responseTerm + unrecoTerm

    jec_ratio = simonjets.pt / simonjets.genPt

    H['jec'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        jec = ak.flatten(jec_cmssw[mask], axis=None),
        trueJEC = ak.flatten(jec_actual[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )
    H['puTerm'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        puTerm = ak.flatten(puTerm[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )
    H['responseTerm'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        responseTerm = ak.flatten(responseTerm[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )
    H['unrecoTerm'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        unrecoTerm = ak.flatten(unrecoTerm[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )
    H['jecRatio'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        jecRatio = ak.flatten(jec_ratio[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )
    H['jecPred'].fill(
        pt = ak.flatten(simonjets.pt[mask], axis=None),
        eta = np.abs(ak.flatten(simonjets.eta[mask], axis=None)),
        jecPred = ak.flatten(jec_simon[mask], axis=None),
        PU = ak.flatten(PU[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

def fillResHists(Hdict, jetreader, wt=1, mask=None):
    data = jetreader.parts

    mask = ensure_mask(mask, data.pt)
    #mask = mask & (data.nmatch > 0)

    dpt = 100*(data.matchPt - data.pt)/data.pt
    deta = data.matchEta - data.eta
    dphi = data.matchPhi - data.phi

    wt, _ = ak.broadcast_arrays(wt, data.pt)

    Hdict['dpt'].fill(
        pt = ak.flatten(data.pt[mask], axis=None),
        eta = np.abs(ak.flatten(data.eta[mask], axis=None)),
        dpt = ak.flatten(dpt[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

    Hdict['deta'].fill(
        pt = ak.flatten(data.pt[mask], axis=None),
        eta = np.abs(ak.flatten(data.eta[mask], axis=None)),
        deta = ak.flatten(deta[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

    Hdict['dphi'].fill(
        pt = ak.flatten(data.pt[mask], axis=None),
        eta = np.abs(ak.flatten(data.eta[mask], axis=None)),
        dphi = ak.flatten(dphi[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

    Hdict['dR'].fill(
        pt = ak.flatten(data.pt[mask], axis=None),
        eta = np.abs(ak.flatten(data.eta[mask], axis=None)),
        dR = ak.flatten(np.sqrt(deta*deta + dphi*dphi)[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

def fillMatchHist(H, jetreader, wt=1, mask=None):
    data = jetreader.parts
    jets = jetreader.simonjets

    mask = ensure_mask(mask, data.pt)
    mask = mask & (~ak.all(data.nmatch == 0, axis=-1)) #mask out PU jets

    jetpt, _ = ak.broadcast_arrays(jets.pt, data.pt)
    ptfrac = data.pt/jetpt

    wt, _ = ak.broadcast_arrays(wt, data.pt)

    detajet = data.eta - jets.eta
    dphijet = data.phi - jets.phi
    dphijet = np.where(dphijet > np.pi, dphijet - 2*np.pi, dphijet)
    dphijet = np.where(dphijet < -np.pi, dphijet + 2*np.pi, dphijet)
    dRjet = np.sqrt(detajet**2 + dphijet**2)

    pdgid = ak.where(data.pdgid == 22, 22,
            ak.where(data.pdgid == 11, 11,
            ak.where(data.pdgid == 13, 13,
            ak.where((data.pdgid>=100) & (data.charge==0), 130,
            ak.where((data.pdgid>=100) & (data.charge!=0), 211, 0)))))

    H.fill(
        pt = ak.flatten(data.pt[mask], axis=None),
        jetpt = ak.flatten(jetpt[mask], axis=None),
        ptfrac = ak.flatten(ptfrac[mask], axis=None),
        eta = np.abs(ak.flatten(data.eta[mask], axis=None)),
        pdgid = ak.flatten(pdgid[mask], axis=None),
        nmatch = ak.flatten(data.nmatch[mask], axis=None),
        dRjet = ak.flatten(dRjet[mask], axis=None),
        weight = ak.flatten(wt[mask], axis=None),
    )

def getMatchingHists(events, jetMaskReco, jetMaskGen, weight, name):
    nameReco = name+"Particles"
    nameGen = name+"GenParticles"

    jetreaderReco = reader.jetreader(events, 
                                     'selectedPatJetsAK4PFPuppi', 
                                     nameReco)
    jetreaderGen = reader.jetreader(events, 
                                    'ak4GenJetsNoNu', 
                                    nameGen)
    
    H_EM0 = getEM0Hists()
    H_HAD0 = getHAD0Hists()
    H_HADCH = getHADCHHists()
    H_ELE = getELEHists()
    H_MU = getMUHists()

    H_EM0_HAD0 = getEM0Hists() #EM0 matched to gen HAD0
    H_EM0_ELE = getEM0Hists() #EM0 matched to gen ELE
    H_EM0_HADCH = getEM0Hists() #EM0 matched to gen HADCH
    H_EM0_EM0 = getEM0Hists() #pure EM0

    H_HAD0_EM0 = getHAD0Hists() #HAD0 matched to gen EM0
    H_HAD0_HADCH = getHAD0Hists() #HAD0 matched to gen HADCH
    H_HAD0_HAD0 = getHAD0Hists() #pure HAD0

    H_HADCH_ELE = getHADCHHists() #HADCH matched to gen ELE
    H_HADCH_MU = getHADCHHists() #HADCH matched to gen MU
    H_HADCH_HADCH = getHADCHHists() #pure HADCH

    H_ELE_HADCH = getELEHists() #ELE matched to gen HADCH
    H_ELE_MU = getELEHists() #ELE matched to gen MU
    H_ELE_ELE = getELEHists() #pure ELE

    H_MU_HADCH = getMUHists() #MU matched to gen HADCH
    H_MU_ELE = getMUHists() #MU matched to gen ELE
    H_MU_MU = getMUHists() #pure MU

    H_matchReco = getMatchHist()
    H_matchGen = getMatchHist()

    H_jets = getJetHists()

    EM0 = jetreaderReco.parts.pdgid==22
    HAD0 = (jetreaderReco.parts.pdgid>=100) & (jetreaderReco.parts.charge==0)
    HADCH = (jetreaderReco.parts.pdgid>=100) & (jetreaderReco.parts.charge!=0)
    ELE = jetreaderReco.parts.pdgid==11
    MU = jetreaderReco.parts.pdgid==13

    matched_EM0 = jetreaderReco.parts.nmatchEM0 > 0
    matched_HAD0 = jetreaderReco.parts.nmatchHAD0 > 0
    matched_HADCH = jetreaderReco.parts.nmatchHADCH > 0
    matched_ELE = jetreaderReco.parts.nmatchEle > 0
    matched_MU = jetreaderReco.parts.nmatchMuon > 0

    pure_EM0 = jetreaderReco.parts.nmatchEM0 == jetreaderReco.parts.nmatch
    pure_HAD0 = jetreaderReco.parts.nmatchHAD0 == jetreaderReco.parts.nmatch
    pure_HADCH = jetreaderReco.parts.nmatchHADCH == jetreaderReco.parts.nmatch
    pure_ELE = jetreaderReco.parts.nmatchEle == jetreaderReco.parts.nmatch
    pure_MU = jetreaderReco.parts.nmatchMuon == jetreaderReco.parts.nmatch

    fillResHists(H_EM0, jetreaderReco, wt=weight, mask=jetMaskReco & EM0)
    fillResHists(H_HAD0, jetreaderReco, wt=weight, mask=jetMaskReco & HAD0)
    fillResHists(H_HADCH, jetreaderReco, wt=weight, mask=jetMaskReco & HADCH)
    fillResHists(H_ELE, jetreaderReco, wt=weight, mask=jetMaskReco & ELE)
    fillResHists(H_MU, jetreaderReco, wt=weight, mask=jetMaskReco & MU)

    fillResHists(H_EM0_HAD0, jetreaderReco, wt=weight, 
                 mask=jetMaskReco & EM0 & matched_HAD0)
    fillResHists(H_EM0_ELE, jetreaderReco, wt=weight,
                 mask=jetMaskReco & EM0 & matched_ELE)
    fillResHists(H_EM0_HADCH, jetreaderReco, wt=weight,
                 mask=jetMaskReco & EM0 & matched_HADCH)
    fillResHists(H_EM0_EM0, jetreaderReco, wt=weight,
                 mask=jetMaskReco & EM0 & pure_EM0)
    fillResHists(H_HAD0_EM0, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HAD0 & matched_EM0)
    fillResHists(H_HAD0_HADCH, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HAD0 & matched_HADCH)
    fillResHists(H_HAD0_HAD0, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HAD0 & pure_HAD0)
    fillResHists(H_HADCH_ELE, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HADCH & matched_ELE)
    fillResHists(H_HADCH_MU, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HADCH & matched_MU)
    fillResHists(H_HADCH_HADCH, jetreaderReco, wt=weight,
                 mask=jetMaskReco & HADCH & pure_HADCH)
    fillResHists(H_ELE_HADCH, jetreaderReco, wt=weight,
                 mask=jetMaskReco & ELE & matched_HADCH)
    fillResHists(H_ELE_MU, jetreaderReco, wt=weight,
                 mask=jetMaskReco & ELE & matched_MU)
    fillResHists(H_ELE_ELE, jetreaderReco, wt=weight,
                 mask=jetMaskReco & ELE & pure_ELE)
    fillResHists(H_MU_HADCH, jetreaderReco, wt=weight,
                 mask=jetMaskReco & MU & matched_HADCH)
    fillResHists(H_MU_ELE, jetreaderReco, wt=weight,
                 mask=jetMaskReco & MU & matched_ELE)
    fillResHists(H_MU_MU, jetreaderReco, wt=weight,
                 mask=jetMaskReco & MU & pure_MU)

    fillMatchHist(H_matchReco, jetreaderReco, 
                  wt=weight, mask=jetMaskReco)
    fillMatchHist(H_matchGen, jetreaderGen, 
                  wt=weight, mask=jetMaskGen)

    fillJetHists(H_jets, jetreaderReco, wt=weight, mask=jetMaskReco)

    return {
        'EM0': H_EM0,
        'HAD0': H_HAD0,
        'HADCH': H_HADCH,
        'ELE': H_ELE,
        'MU': H_MU,
        'EM0_HAD0': H_EM0_HAD0,
        'EM0_ELE': H_EM0_ELE,
        'EM0_HADCH': H_EM0_HADCH,
        'EM0_EM0': H_EM0_EM0,
        'HAD0_EM0': H_HAD0_EM0,
        'HAD0_HADCH': H_HAD0_HADCH,
        'HAD0_HAD0': H_HAD0_HAD0,
        'HADCH_ELE': H_HADCH_ELE,
        'HADCH_MU': H_HADCH_MU,
        'HADCH_HADCH': H_HADCH_HADCH,
        'ELE_HADCH': H_ELE_HADCH,
        'ELE_MU': H_ELE_MU,
        'ELE_ELE': H_ELE_ELE,
        'MU_HADCH': H_MU_HADCH,
        'MU_ELE': H_MU_ELE,
        'MU_MU': H_MU_MU,
        'matchReco': H_matchReco,
        'matchGen': H_matchGen,
        'jets': H_jets,
    }

