import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet
from .util import findBinEdges

def getMuonSF(cset, name, eta, pt):
    
    badnone = ak.is_none(eta) | ak.is_none(pt)
    pt = ak.fill_none(pt, 0)
    eta = ak.fill_none(eta, 0)

    minpt, maxpt = findBinEdges(cset, name, 'pt')
    mineta, maxeta = findBinEdges(cset, name, 'eta')
    ptthresh = minpt
    etathresh = maxeta - 1e-8

    badpt = (pt < ptthresh)
    badeta = (np.abs(eta) > etathresh)
    bad = badpt | badeta | badnone

    etaEval = ak.fill_none(eta, 0)
    ptEval = ak.fill_none(pt, 0)

    etaEval = ak.to_numpy(ak.where(bad, etathresh, etaEval))
    ptEval = ak.to_numpy(ak.where(bad, ptthresh, ptEval))

    nom = cset[name].evaluate(
        etaEval,
        ptEval,
        'nominal'
    )
    up = cset[name].evaluate(
        etaEval,
        ptEval,
        'systup'
    )
    dn = cset[name].evaluate(
        etaEval,
        ptEval,
        'systdown'
    )
    
    nom = np.where(bad, 1, nom)
    up = np.where(bad, 1, up)
    dn = np.where(bad, 1, dn)

    return nom, up, dn

def getAllMuonSFs(weights,
                  readers,
                  config, 
                  noPrefireSF, 
                  noIDsfs,
                  noIsosfs, 
                  noTriggersfs):

    if config.eventSelection.PreFireWeight and not noPrefireSF:
        prefire = readers.prefirewt
        weights.add("wt_prefire", 
                           prefire.Nom, 
                           prefire.Up, 
                           prefire.Dn)

    muons = readers.rMu.muons

    sfconfig = config.muonSFs

    cset = CorrectionSet.from_file(sfconfig.path)

    leadmu = muons[:,0]
    submu = muons[:,1]

    if sfconfig.useRoccoR:
        leadpt = leadmu.pt
        subpt = submu.pt
    else:
        leadpt = leadmu.rawPt
        subpt = submu.rawPt

    leadeta = np.abs(leadmu.eta)
    subeta = np.abs(submu.eta)


    mask = leadpt > subpt

    tmppt = leadpt[:]
    tmpeta = leadeta[:]
    leadpt = np.where(mask, leadpt, subpt)
    leadeta = np.where(mask, leadeta, subeta)
    subpt = np.where(mask, subpt, tmppt)
    subeta = np.where(mask, subeta, np.abs(tmpeta))

    whichID = config.muonSelection.ID
    whichIso = config.muonSelection.iso
    #this is the only time it's awkward to not be using a regular dict
    idsfname = vars(sfconfig.idsfnames)[whichID]
    isosfname = vars(vars(sfconfig.isosfnames)[whichIso])[whichID]
    triggersfname = vars(sfconfig.triggersfnames)[config.eventSelection.trigger]

    clipPt = config.muonSelection.subpt
    clipEta = config.muonSelection.subeta

    if not noIDsfs:
        idsf_lead, idsf_lead_up, idsf_lead_dn = getMuonSF(cset, 
                                                          idsfname, 
                                                          leadeta, 
                                                          leadpt)

        idsf_sub, idsf_sub_up, idsf_sub_dn = getMuonSF(cset, 
                                                       idsfname, 
                                                       subeta, 
                                                       subpt)

 
        weights.add("wt_idsf", 
                    idsf_lead*idsf_sub, 
                    idsf_lead_up*idsf_sub_up, 
                    idsf_lead_dn*idsf_sub_dn)
    if not noIsosfs:
        isosf_lead, isosf_lead_up, isosf_lead_dn = getMuonSF(cset, 
                                                             isosfname, 
                                                             leadeta,
                                                             leadpt)

        isosf_sub, isosf_sub_up, isosf_sub_dn = getMuonSF(cset,
                                                          isosfname,
                                                          subeta, 
                                                          subpt)

        weights.add("wt_isosf", 
                    isosf_lead*isosf_sub, 
                    isosf_lead_up*isosf_sub_up, 
                    isosf_lead_dn*isosf_sub_dn)
    if not noTriggersfs:
        triggersf, triggersf_up, triggersf_dn = getMuonSF(cset, 
                                                      triggersfname, 
                                                      leadeta, 
                                                      leadpt)

        weights.add("wt_triggersf", triggersf, triggersf_up, triggersf_dn)

