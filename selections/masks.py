import numpy as np
import awkward as ak
import warnings

from coffea.analysis_tools import PackedSelection

def getSelectedMuons(readers, config):
    rmu = readers.rMu

    if config.checkPdgId:
        theMuons = rmu.muons[np.abs(rmu.muons.pdgId) == 13]
    else:
        theMuons = rmu.muons

    if config.ID == 'loose':
        theMuons = theMuons[theMuons.looseId]
    elif config.ID == 'medium':
        theMuons = theMuons[theMuons.mediumId]
    elif config.ID == 'tight':
        theMuons = theMuons[theMuons.tightId]
    elif config.ID == 'none':
        pass

    if config.iso == 'loose':
        theMuons = theMuons[theMuons.pfIsoId >= 2]
    elif config.iso == 'tight':
        theMuons = theMuons[theMuons.pfIsoId >= 4]
    elif config.iso == 'none':
        pass

    theMuons = ak.pad_none(theMuons, 2, axis=1)

    if config.onlyCheckLeading:
        return theMuons[:,0], theMuons[:,1]
    else:
        raise NotImplementedError("need to set up cartesian product for muons")

def addMuonSelections(selection, readers, config):
    rmu = readers.rMu

    mu0, mu1 = getSelectedMuons(readers, config)

    leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
    submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

    selection.add('twomu', ~ak.is_none(mu0) & ~ak.is_none(mu1))

    if config.leadpt >= 0:
        selection.add("leadpt", leadmu.pt >= config.leadpt)
    if config.subpt >= 0:
        selection.add("subpt", submu.pt >= config.subpt)
    if config.leadeta >= 0:
        selection.add("leadeta", np.abs(leadmu.eta) < config.leadeta)
    if config.subeta >= 0:
        selection.add("subeta", np.abs(submu.eta) < config.subeta)

    if config.ID == "loose":
        selection.add("leadID", leadmu.looseId)
        selection.add("subID", submu.looseId)
    elif config.ID == "medium":
        selection.add("leadID", leadmu.mediumId)
        selection.add("subID", submu.mediumId)
    elif config.ID == "tight":
        selection.add("leadID", leadmu.tightId)
        selection.add("subID", submu.tightId)
    elif config.ID == 'none':
        pass
    else:
        raise ValueError("Invalid muon ID: {}".format(config.ID))

    if config.iso == 'loose':
        selection.add("leadiso", leadmu.pfIsoId >= 2)
        selection.add("subiso", submu.pfIsoId >= 2)
    elif config.iso == 'tight':
        selection.add("leadiso", leadmu.pfIsoId >= 4)
        selection.add("subiso", submu.pfIsoId >= 4)
    elif config.iso == 'none':
        pass
    else:
        raise ValueError("Invalid muon iso: {}".format(config.iso))

    if config.oppsign:
        selection.add("oppsign", (leadmu.charge * submu.charge) < 0)

    mask = selection.all(*selection.names)

    return selection

def addEventSelections(selection, readers, config, noBkgVeto):
    if config.trigger != '':
        selection.add("trigger", readers.HLT[config.trigger])
    njet = ak.num(readers.rRecoJet.jets.pt)
    if config.MinNumJets >= 0:
        selection.add("minNumJet", njet >= config.MinNumJets)
    if config.MaxNumJets >= 0:
        selection.add("maxNumJet", njet <= config.MaxNumJets)

    if not noBkgVeto:
        if config.maxMETpt >= 0:
            selection.add("METpt", readers.METpt < config.maxMETpt)

        if config.maxNumBtag >= 0:
            if config.maxNumBtag_level == 'loose':
                nPassB = ak.sum(readers.rRecoJet.jets.passLooseB, axis=-1)
            elif config.maxNumBtag_level == 'medium':
                nPassB = ak.sum(readers.rRecoJet.jets.passMediumB, axis=-1)
            elif config.maxNumBtag_level == 'tight':
                nPassB = ak.sum(readers.rRecoJet.jets.passTightB, axis=-1)
            else:
                raise ValueError("Invalid btag level: %s"%(config.maxNumBtag_level))
            selection.add("nbtag", nPassB <= config.maxNumBtag)
    
    if len(config.noiseFilters) > 0:
        filtermask = np.ones(len(readers.eventIdx), dtype=bool)
        for flag in config.noiseFilters:
            filtermask = filtermask & readers.Flag[flag]

        selection.add("noiseFilters", filtermask)

    return selection

def addZSelections(selection, readers, config):
    Z = readers.rMu.Zs
    if config.ZmassWindow >= 0:
        upper = config.Zmass + config.ZmassWindow
        lower = config.Zmass - config.ZmassWindow
        selection.add("Zmass", (Z.mass <= upper) & (Z.mass >= lower))
    if config.minZPt >= 0:
        selection.add("Zpt", Z.pt >= config.minZPt)
    if config.maxZY >= 0:
        selection.add("Zy", np.abs(Z.y) < config.maxZY)
    return selection

def getEventSelection(readers, config, isMC, flags, noBkgVeto):
    selection = PackedSelection()
    selection = addMuonSelections(selection, readers, config.muons)
    selection = addZSelections(selection, readers, config.eventSelection)
    selection = addEventSelections(selection, readers, config.eventSelection,
                                   noBkgVeto)

    if flags is not None:
        for flag in flags:
            if flag.startswith("HTcut"):
                #this is allowd to be a hack
                selection.add("genHT", readers.LHE.HT < int(flag[5:]))

    return selection
