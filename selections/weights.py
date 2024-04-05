import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

EPS = 1e-6

def getMuonSF(cset, name, year, eta, pt, bad):
    nom = cset[name].evaluate(
        year,
        eta,
        pt,
        'sf'
    )
    up = cset[name].evaluate(
        year,
        eta,
        pt,
        'systup'
    )
    dn = cset[name].evaluate(
        year,
        eta,
        pt,
        'systdown'
    )
    
    nom = np.where(bad, 1, nom)
    up = np.where(bad, 1, up)
    dn = np.where(bad, 1, dn)

    return nom, up, dn

def getAllMuonSFs(weights, prefire, muons, config, isMC):
    if not isMC:
        return

    if config.eventSelection.PreFireWeight:
        weights.add("wt_prefire", prefire.Nom, 
                           prefire.Up, 
                           prefire.Dn)

        cset = CorrectionSet.from_file(config.muon_sfpath)

    idsfnames = {
        'loose': 'NUM_LooseID_DEN_genTracks',
        'medium': 'NUM_MediumID_DEN_genTracks',
        'tight': 'NUM_TightID_DEN_genTracks'
    }
    isosfnames = {
        'loose' : {
            'loose' : 'NUM_LooseRelIso_DEN_LooseID',
            'medium' : 'NUM_LooseRelIso_DEN_MediumID',
            'tight' : 'NUM_LooseRelIso_DEN_TightIDandIPCut'
        },
        'tight' : {
            'medium' : 'NUM_TightRelIso_DEN_MediumID',
            'tight' : 'NUM_TightRelIso_DEN_TightIDandIPCut'
        },
    }

    triggersfnames = {
        'IsoMu24' : 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight'
    }

    leadmu = muons[:,0]
    submu = muons[:,1]

    leadpt = leadmu.pt
    leadpt = ak.fill_none(leadpt, 0)
    leadeta = np.abs(leadmu.eta)
    leadeta = ak.fill_none(leadeta, 0)

    badleadpt = leadpt <= config.muonSelection.leadpt
    leadpt = np.where(badleadpt, config.muonSelection.leadpt, leadpt)

    badleadeta = leadeta > config.muonSelection.leadeta-EPS
    leadeta = np.where(badleadeta, config.muonSelection.leadeta-EPS, leadeta)

    subpt = submu.pt
    subpt = ak.fill_none(subpt, 0)
    subeta = np.abs(submu.eta)
    subeta = ak.fill_none(subeta, 0)

    badsubpt = subpt <= config.muonSelection.subpt
    subpt = np.where(badsubpt, config.muonSelection.subpt, subpt)

    badsubeta = subeta > config.muonSelection.subeta-EPS
    subeta = np.where(badsubeta, config.muonSelection.subeta-EPS, subeta)

    badlead = badleadpt | badleadeta
    badsub = badsubpt | badsubeta

    idsfname = idsfnames[config.muonSelection.ID]
    isosfname = isosfnames[config.muonSelection.iso][config.muonSelection.ID]
    triggersfname = triggersfnames[config.eventSelection.trigger]

    idsf_lead, idsf_lead_up, idsf_lead_dn = getMuonSF(cset, 
                                                      idsfname, 
                                                      '2018_UL', 
                                                      leadeta, 
                                                      leadpt,
                                                      badlead)

    idsf_sub, idsf_sub_up, idsf_sub_dn = getMuonSF(cset, 
                                                   idsfname, 
                                                   '2018_UL', 
                                                   subeta, 
                                                   subpt,
                                                   badsub)

    isosf_lead, isosf_lead_up, isosf_lead_dn = getMuonSF(cset, 
                                                         isosfname, 
                                                         '2018_UL', 
                                                         leadeta,
                                                         leadpt,
                                                         badlead)

    isosf_sub, isosf_sub_up, isosf_sub_dn = getMuonSF(cset,
                                                      isosfname,
                                                      '2018_UL', 
                                                      subeta, 
                                                      subpt,
                                                      badsub)

    triggersf, triggersf_up, triggersf_dn = getMuonSF(cset, 
                                                      triggersfname, 
                                                      '2018_UL', 
                                                      leadeta, 
                                                      leadpt,
                                                      badlead)

    weights.add("wt_idsf", 
                idsf_lead*idsf_sub, 
                idsf_lead_up*idsf_sub_up, 
                idsf_lead_dn*idsf_sub_dn)
    weights.add("wt_isosf", 
                isosf_lead*isosf_sub, 
                isosf_lead_up*isosf_sub_up, 
                isosf_lead_dn*isosf_sub_dn)
    weights.add("wt_triggersf", triggersf, triggersf_up, triggersf_dn)

def getCtagSF(rRecoJet, config, ans, isMC):
    if not isMC:
        return

    raise NotImplementedError

def getBtagSF(rRecoJet, config, ans, isMC):
    if not isMC:
        return

    raise NotImplementedError

    jets = rRecoJet.CHSjets 
    fakejets = jets.pt == 0

    jets = jets[~fakejets]

    cset = CorrectionSet.from_file(config.btag_sfpath)

    flav = ak.flatten(jets.hadronFlavour, axis=-1)
    abseta = np.abs(ak.flatten(jets.eta, axis=-1))
    pt = ak.flatten(jets.pt, axis=-1)

    num = ak.num(flav)
    flav = ak.flatten(flav, axis=None)
    abseta = ak.flatten(abseta, axis=None)
    pt = ak.flatten(pt, axis=None)

    wp = config.btagWP

    jet_sf = cset['deepJet_comb'].evaluate(
        'central',
        flav,
        abseta,
        pt
    )

    jet_sf_up = cset['deepJet_comb'].evaluate(
        'up',
        flav,
        abseta,
        pt
    )

    jet_sf_dn = cset['deepJet_comb'].evaluate(
        'down',
        flav,
        abseta,
        pt
    )
    
    jet_sf = ak.unflatten(jet_sf, num, axis=0)
    jet_sf = ak.prod(jet_sf, axis=-1)

    jet_sf_up = ak.unflatten(jet_sf_up, num, axis=0)
    jet_sf_up = ak.prod(jet_sf_up, axis=-1)

    jet_sf_dn = ak.unflatten(jet_sf_dn, num, axis=0)
    jet_sf_dn = ak.prod(jet_sf_dn, axis=-1)

    ans.add("btagSF", jet_sf, jet_sf_up, jet_sf_dn)

def getCdiscWeight(rRecoJet, config, ans, isMC):
    if not isMC:
        return
    jets = rRecoJet.CHSjets

    fakejets = jets.pt == 0

    jets = jets[~fakejets]

    cset = CorrectionSet.from_file(config.ctag_sfpath)

    flav = ak.flatten(jets.hadronFlavour, axis=-1)
    CvL = ak.flatten(jets.btagDeepFlavCvL, axis=-1)
    CvB = ak.flatten(jets.btagDeepFlavCvB, axis=-1)

    num = ak.num(flav)
    flav = ak.flatten(flav, axis=None)
    CvL = ak.flatten(CvL, axis=None)
    CvB = ak.flatten(CvB, axis=None)
    
    jet_sf = cset['deepJet_shape'].evaluate(
        'central',
        flav, 
        CvL, 
        CvB,
    )
    
    jet_sf = ak.unflatten(jet_sf, num, axis=0)

    jet_sf = ak.prod(jet_sf, axis=-1)

    sumbefore = np.sum(ans.weight())
    sumafter = np.sum(ans.weight() * jet_sf)

    jet_sf = jet_sf * sumbefore / sumafter
    ans.add("ctagSF", jet_sf)

def getScaleWts7pt(weights, x):
    '''
    LHE scale variation weights (w_var / w_nominal); 
    [0] is MUF="0.5" MUR="0.5"; 
    [1] is MUF="1.0" MUR="0.5"; 
    [2] is MUF="2.0" MUR="0.5"; 
    [3] is MUF="0.5" MUR="1.0"; 
    [4] is MUF="1.0" MUR="1.0"; 
    [5] is MUF="2.0" MUR="1.0"; 
    [6] is MUF="0.5" MUR="2.0"; 
    [7] is MUF="1.0" MUR="2.0"; 
    [8] is MUF="2.0" MUR="2.0"
    '''
    if not hasattr(x, 'LHEScaleWeight'):
        return

    var_weights = x.LHEScaleWeight
    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)
 
    up = np.maximum.reduce([var_weights[:,0],
                            var_weights[:,1],
                            var_weights[:,3],
                            var_weights[:,5],
                            var_weights[:,7],
                            var_weights[:,8]])

    down = np.minimum.reduce([var_weights[:,0],
                              var_weights[:,1],
                              var_weights[:,3],
                              var_weights[:,5],
                              var_weights[:,7],
                              var_weights[:,8]])

    weights.add('wt_scale', nom, up, down)

def getScaleWts3pt(weights, x):
    if not hasattr(x, 'LHEScaleWeight'):
        return

    var_weights = x.LHEScaleWeight
    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)

    up = np.maximum(var_weights[:,0], var_weights[:,8])
    down = np.minimum(var_weights[:,0], var_weights[:,8])

    weights.add('wt_scale_3pt', nom, up, down)

def getPSWts(weights, x):
    ps_weights = x.PSWeight
    if ak.num(ps_weights)[0] < 4:
        return

    nweights = len(weights.weight())

    nom  = np.ones(nweights)

    up_isr   = np.ones(nweights)
    down_isr = np.ones(nweights)

    up_fsr   = np.ones(nweights)
    down_fsr = np.ones(nweights)

    up_isr = ps_weights[:,0]
    down_isr = ps_weights[:,2]

    up_fsr = ps_weights[:,1]
    down_fsr = ps_weights[:,3]
        
    weights.add('wt_ISR', nom, up_isr, down_isr)
    weights.add('wt_FSR', nom, up_fsr, down_fsr)

def getPDFweights(weights, x):
    if not hasattr(x, 'LHEPdfWeight'):
        return

    nweights = len(weights.weight())
    pdf_weights = x.LHEPdfWeight

    nom   = np.ones(nweights)

    # Hessian PDF weights
    # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
    arg = pdf_weights[:,1:-2]-np.ones((nweights,100))
    summed = ak.sum(np.square(arg),axis=1)
    pdf_unc = np.sqrt( (1./99.) * summed )
    weights.add('wt_PDF', nom, pdf_unc + nom, nom - pdf_unc)

    # alpha_S weights
    # Eq. 27 of same ref
    as_unc = 0.5*(pdf_weights[:,102] - pdf_weights[:,101])
    weights.add('wt_aS', nom, as_unc + nom, nom - as_unc)

    # PDF + alpha_S weights
    # Eq. 28 of same ref
    pdfas_unc = np.sqrt( np.square(pdf_unc) + np.square(as_unc) )
    weights.add('wt_PDFaS', nom, pdfas_unc + nom, nom - pdfas_unc) 

def getEventWeight(x, muons, rRecoJet, config, isMC):
    ans = Weights(len(x), storeIndividual=True)

    if isMC:
        #generator weight
        ans.add('generator', x.genWeight)

        #theory uncertainties
        getScaleWts7pt(ans, x)
        #getScaleWts3pt(ans, x)
        getPSWts(ans, x)
        getPDFweights(ans, x)

        #muon scale factors
        getAllMuonSFs(ans, x.L1PreFiringWeight, muons, config, isMC)

        #ctag reshaping SFs
        if config.tagging.mode == 'regions':
            getCdiscWeight(rRecoJet, config, ans, isMC)
        elif config.tagging.mode == 'ctag':
            #getCtagSF(rRecoJet, config, ans, isMC)
            print("WARNING: skipping ctag SFs for now")
        elif config.tagging.mode == 'btag':
            #getBtagSF(rRecoJet, config, ans, isMC)
            print("WARNING: skipping btag SFs for now")
        else:
            raise NotImplementedError

    return ans
