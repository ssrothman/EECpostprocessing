from typing import Any
import numpy as np
import awkward as ak
import warnings

from coffea.analysis_tools import PackedSelection

def getSelectedMuons(readers, config, verbose):
    #print("ROCCOR TEST")
    #print(readers.rMu.muons.pt[:,0])
    #print(readers.rMu.muons.rawPt[:,0])
    #print(readers.rMu._x.ZMuMuMuons.pt[:,0])
    #print()
    #print(readers.rMu.muons.pt[:,1])
    #print(readers.rMu.muons.rawPt[:,1])
    #print(readers.rMu._x.ZMuMuMuons.pt[:,1])
    #print()

    if verbose:
        print("Getting top two passing muons:")
    rmu = readers.rMu

    if config.checkPdgId:
        if verbose:
            print("\tRequiring muon pdgId == 13")
        theMuons = rmu.muons[np.abs(rmu.muons.pdgId) == 13]
    else:
        theMuons = rmu.muons

    if config.ID == 'loose':
        if verbose:
            print("\tRequiring muon loose ID")
        theMuons = theMuons[theMuons.looseId]
    elif config.ID == 'medium':
        if verbose:
            print("\tRequiring muon medium ID")
        theMuons = theMuons[theMuons.mediumId]
    elif config.ID == 'tight':
        if verbose:
            print("\tRequiring muon tight ID")
        theMuons = theMuons[theMuons.tightId]
    elif config.ID == 'none':
        pass

    if config.iso == 'loose':
        if verbose:
            print("\tRequiring muon loose iso")
        theMuons = theMuons[theMuons.pfIsoId >= 2]
    elif config.iso == 'tight':
        if verbose:
            print("\tRequiring muon tight iso")
        theMuons = theMuons[theMuons.pfIsoId >= 4]
    elif config.iso == 'none':
        pass

    theMuons = ak.pad_none(theMuons, 2, axis=1)

    if config.onlyCheckLeading:
        return theMuons[:,0], theMuons[:,1]
    else:
        raise NotImplementedError("need to set up cartesian product for muons")

def addMuonSelections(selection, readers, config, verbose):
    if verbose:
        print("Applying muon selections:")
    rmu = readers.rMu

    mu0, mu1 = getSelectedMuons(readers, config, verbose)

    leadmu : Any = ak.where(mu0.pt > mu1.pt, mu0, mu1)
    submu : Any = ak.where(mu0.pt > mu1.pt, mu1, mu0)

    if verbose:
        print("\tRequiring two muons")
    selection.add('twomu', ~ak.is_none(mu0) & ~ak.is_none(mu1))

    if config.leadpt >= 0:
        if verbose:
            print("\tRequiring lead muon pt > %g"%config.leadpt)
        selection.add("leadpt", leadmu.pt >= config.leadpt)
    if config.subpt >= 0:
        if verbose:
            print("\tRequiring sublead muon pt > %g"%config.subpt)
        selection.add("subpt", submu.pt >= config.subpt)
    if config.leadeta >= 0:
        if verbose:
            print("\tRequiring lead muon |eta| < %g"%config.leadeta)
        selection.add("leadeta", np.abs(leadmu.eta) < config.leadeta)
    if config.subeta >= 0:
        if verbose:
            print("\tRequiring sublead muon |eta| < %g"%config.subeta)
        selection.add("subeta", np.abs(submu.eta) < config.subeta)

    if config.ID == "loose":
        if verbose:
            print("\tRequiring both muon loose ID")
        selection.add("leadID", leadmu.looseId)
        selection.add("subID", submu.looseId)
    elif config.ID == "medium":
        if verbose:
            print("\tRequiring both muon medium ID")
        selection.add("leadID", leadmu.mediumId)
        selection.add("subID", submu.mediumId)
    elif config.ID == "tight":
        if verbose:
            print("\tRequiring both muon tight ID")
        selection.add("leadID", leadmu.tightId)
        selection.add("subID", submu.tightId)
    elif config.ID == 'none':
        pass
    else:
        raise ValueError("Invalid muon ID: {}".format(config.ID))

    if config.iso == 'loose':
        if verbose:
            print("\tRequiring both muon loose iso")
        selection.add("leadiso", leadmu.pfIsoId >= 2)
        selection.add("subiso", submu.pfIsoId >= 2)
    elif config.iso == 'tight':
        if verbose:
            print("\tRequiring both muon tight iso")
        selection.add("leadiso", leadmu.pfIsoId >= 4)
        selection.add("subiso", submu.pfIsoId >= 4)
    elif config.iso == 'none':
        pass
    else:
        raise ValueError("Invalid muon iso: {}".format(config.iso))

    if config.oppsign:
        if verbose:
            print("\tRequiring muons have opposite charge")
        selection.add("oppsign", (leadmu.charge * submu.charge) < 0)

    mask = selection.all(*selection.names)

    return selection

def addEventSelections(selection, readers, config, noBkgVeto, verbose):
    if verbose:
        print("Setting up event selections:")
    if config.trigger != '':
        if verbose:
            print("\tRequiring %s trigger"%config.trigger)
        selection.add("trigger", readers.HLT[config.trigger])
    njet = ak.num(readers.rRecoJet.jets.pt)
    if config.MinNumJets >= 0:
        if verbose:
            print("\tRequiring at least %d jets"%config.MinNumJets)
        selection.add("minNumJet", njet >= config.MinNumJets)
    if config.MaxNumJets >= 0:
        if verbose:
            print("\tRequiring at most %d jets"%config.MaxNumJets)
        selection.add("maxNumJet", njet <= config.MaxNumJets)

    if not noBkgVeto:
        if config.maxMETpt >= 0:
            if verbose:
                print("\tRequiring MET pt < %g"%config.maxMETpt)
            selection.add("METpt", readers.METpt < config.maxMETpt)

        if config.maxNumBtag >= 0:
            if verbose:
                print("\tRequiring at most %d %s b tags"%(config.maxNumBtag, config.maxNumBtag_level))
            
            checkjets = readers.rRecoJet.CHSjets
            checkjets = checkjets[checkjets.pt > 30]
            checkjets = checkjets[checkjets.jetId == 6]

            if config.maxNumBtag_level == 'loose':
                nPassB = ak.sum(checkjets.passLooseB, axis=-1)
            elif config.maxNumBtag_level == 'medium':
                nPassB = ak.sum(checkjets.passMediumB, axis=-1)
            elif config.maxNumBtag_level == 'tight':
                nPassB = ak.sum(checkjets.passTightB, axis=-1)
            else:
                raise ValueError("Invalid btag level: %s"%(config.maxNumBtag_level))
            selection.add("nbtag", nPassB <= config.maxNumBtag)
    
    if len(config.noiseFilters) > 0:
        if verbose:
            print("\tApplying noise filters")
        filtermask = np.ones(len(readers.eventIdx), dtype=bool)
        for flag in config.noiseFilters:
            filtermask = filtermask & readers.Flag[flag]

        selection.add("noiseFilters", filtermask)

    return selection

def addZSelections(selection, readers, config, verbose):
    if verbose:
        print("Setting up Z selections:")
    Z = readers.rMu.Zs
    if config.ZmassWindow >= 0:
        upper = config.Zmass + config.ZmassWindow
        lower = config.Zmass - config.ZmassWindow
        if verbose:
            print("\tRequiring dimuon mass between %g - %g"%(lower, upper))
        selection.add("Zmass", (Z.mass <= upper) & (Z.mass >= lower))
    if config.minZPt >= 0:
        if verbose:
            print("\tRequiring Z pt > %g"%config.minZPt)
        selection.add("Zpt", Z.pt >= config.minZPt)
    if config.maxZY >= 0:
        if verbose:
            print("\tRequiring Z y > %g"%config.maxZY)
        selection.add("Zy", np.abs(Z.rapidity) < config.maxZY)
    return selection

def getEventSelection(readers, config, isMC, flags, noBkgVeto, verbose):
    selection = PackedSelection()
    selection = addMuonSelections(selection, readers, config.muons, 
                                  verbose)
    selection = addZSelections(selection, readers, config.eventSelection, 
                               verbose)
    selection = addEventSelections(selection, readers, config.eventSelection,
                                   noBkgVeto, verbose)

    if flags is not None:
        for flag in flags:
            if flag.startswith("HTcut"):
                #this is allowd to be a hack
                if verbose:
                    print("APPLYING GEN HT CUT < %g"%(int(flag[5:])))
                selection.add("genHT", readers.LHE.HT < int(flag[5:]))

    return selection
