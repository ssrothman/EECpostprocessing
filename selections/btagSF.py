import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def getBtagSF(weights, rRecoJet, config):
    cset_sf = CorrectionSet.from_file(config.btagSF.sfpath)
    cset_eff = CorrectionSet.from_file(config.btagSF.effpath)

    CHSjets = rRecoJet.CHSjets
    fakejets = CHSjets.pt == 0

    CHSjets = CHSjets[~fakejets]
    CHSjets = ak.flatten(CHSjets, axis=2)

    pt = CHSjets.pt
    abseta = np.abs(CHSjets.eta)
    flav = CHSjets.hadronFlavour

    num = ak.num(pt)

    pt = ak.to_numpy(ak.flatten(pt, axis=None))
    abseta = ak.flatten(abseta, axis=None)
    flav = ak.flatten(flav, axis=None)

    taggingwp = config.tagging.wp
    vetowp = config.eventSelection.maxNumBtag_level

    if taggingwp == vetowp:
        if taggingwp == 'loose':
            passWP = ak.flatten(CHSjets.passLooseB, axis=None)
        elif taggingwp == 'medium':
            passWP = ak.flatten(CHSjets.passMediumB, axis=None)
        elif taggingwp == 'tight':
            passWP = ak.flatten(CHSjets.passTightB, axis=None)

        single_wp_btagSF(weights, pt, abseta, flav, passWP, num, 
                         taggingwp, cset_sf, cset_eff)
    else:
        sortvals = {'loose' : 0, 'medium' : 1, 'tight' : 2}
        if sortvals[taggingwp] > sortvals[vetowp]:
            larger = taggingwp
            smaller = vetowp
        else:
            larger = vetowp
            smaller = taggingwp


        if smaller == 'loose':
            passWP_smaller = CHSjets.passLooseB
        elif smaller == 'medium':
            passWP_smaller = CHSjets.passMediumB
        elif smaller == 'tight':
            passWP_smaller = CHSjets.passTightB

        if larger == 'loose':
            passWP_larger = CHSjets.passLooseB
        elif larger == 'medium':
            passWP_larger = CHSjets.passMediumB
        elif larger == 'tight':
            passWP_larger = CHSjets.passTightB

        passWP_smaller = ak.flatten(passWP_smaller, axis=None)
        passWP_larger = ak.flatten(passWP_larger, axis=None)

        double_wp_btagSF(weights, pt, abseta, flav, 
                         passWP_smaller, passWP_larger, num,
                         smaller, larger, cset_sf, cset_eff)
