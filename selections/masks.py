import numpy as np
import awkward as ak
import warnings

from coffea.analysis_tools import PackedSelection

def addMuonSelections(selection, readers, config):
    rmu = readers.rMu

    mu0 = rmu.muons[:,0]
    mu1 = rmu.muons[:,1]
    leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
    submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

    nMu = ak.count(rmu.muons.pt, axis=-1)
    selection.add("nummu", (nMu >= config.minNumMu) & (nMu <= config.maxNumMu))
    selection.add("leadpt", leadmu.pt >= config.leadpt)
    selection.add("subpt", submu.pt >= config.subpt)
    selection.add("leadeta", np.abs(leadmu.eta) < config.leadeta)
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
    selection.add("trigger", readers.HLT[config.trigger])
    njet = ak.num(readers.rRecoJet.jets.pt)
    mask = (njet >= config.MinNumJets) & (njet <= config.MaxNumJets)
    selection.add("numjet", mask)

    if not noBkgVeto:
        selection.add("METpt", readers.METpt < config.maxMETpt)

        if config.maxNumBtag_level == 'loose':
            nPassB = ak.sum(readers.rRecoJet.jets.passLooseB, axis=-1)
        elif config.maxNumBtag_level == 'medium':
            nPassB = ak.sum(readers.rRecoJet.jets.passMediumB, axis=-1)
        elif config.maxNumBtag_level == 'tight':
            nPassB = ak.sum(readers.rRecoJet.jets.passTightB, axis=-1)
        else:
            raise ValueError("Invalid btag level: %s"%(config.maxNumBtag_level))
        selection.add("nbtag", nPassB <= config.maxNumBtag)

    filtermask = np.ones(len(readers.eventIdx), dtype=bool)
    for flag in config.noiseFilters:
        filtermask = filtermask & readers.Flag[flag]

    selection.add("noiseFilters", filtermask)

    return selection

def addZSelections(selection, readers, config):
    Z = readers.rMu.Zs
    upper = config.Zmass + config.ZmassWindow
    lower = config.Zmass - config.ZmassWindow
    selection.add("Zmass", (Z.mass <= upper) & (Z.mass >= lower))
    selection.add("Zpt", Z.pt >= config.minPt)
    selection.add("Zy", np.abs(Z.y) < config.maxY)
    return selection

def getEventSelection(readers, config, isMC, flags, noBkgVeto):
    selection = PackedSelection()
    selection = addMuonSelections(selection, readers, config.muonSelection)
    selection = addZSelections(selection, readers, config.Zselection)
    selection = addEventSelections(selection, readers, config.eventSelection,
                                   noBkgVeto)

    if flags is not None:
        for flag in flags:
            if flag.startswith("HTcut"):
                #this is allowd to be a hack
                selection.add("genHT", readers.LHE.HT < int(flag[5:]))

    return selection
