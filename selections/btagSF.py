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

    wp = config.tagging.wp

    eff = cset_eff[wp].evaluate(pt, abseta, flav)

    light = flav == 0

    WPstrs = {'loose' : 'L', 'medium' : 'M', 'tight' : 'T'}
    
    sf_nom_light = cset_sf['deepJet_incl'].evaluate(
        'central',
        WPstrs[wp],
        flav[light],
        abseta[light],
        pt[light]
    )
    sf_nom_heavy = cset_sf['deepJet_comb'].evaluate(
        'central',
        WPstrs[wp],
        flav[~light],
        abseta[~light],
        pt[~light]
    )
    sf_nom = np.ones_like(pt)
    sf_nom[light] = sf_nom_light
    sf_nom[~light] = sf_nom_heavy

    sf_up_light = cset_sf['deepJet_incl'].evaluate(
        'up',
        WPstrs[wp],
        flav[light],
        abseta[light],
        pt[light]
    )
    sf_up_heavy = cset_sf['deepJet_comb'].evaluate(
        'up',
        WPstrs[wp],
        flav[~light],
        abseta[~light],
        pt[~light]
    )
    sf_up = np.ones_like(pt)
    sf_up[light] = sf_up_light
    sf_up[~light] = sf_up_heavy

    sf_dn_light = cset_sf['deepJet_incl'].evaluate(
        'down',
        WPstrs[wp],
        flav[light],
        abseta[light],
        pt[light]
    )
    sf_dn_heavy = cset_sf['deepJet_comb'].evaluate(
        'down',
        WPstrs[wp],
        flav[~light],
        abseta[~light],
        pt[~light]
    )
    sf_dn = np.ones_like(pt)
    sf_dn[light] = sf_dn_light
    sf_dn[~light] = sf_dn_heavy

    effXsf = eff * sf_nom
    effXsf_up = eff * sf_up
    effXsf_dn = eff * sf_dn

    eff = ak.unflatten(eff, num, axis=0)
    effXsf = ak.unflatten(effXsf, num, axis=0)
    effXsf_up = ak.unflatten(effXsf_up, num, axis=0)
    effXsf_dn = ak.unflatten(effXsf_dn, num, axis=0)

    wp_val = vars(config.tagging.bwps)[wp]
    pass_wp = CHSjets.btagDeepFlavB > wp_val

    probMC = ak.prod(eff[pass_wp], axis=1) * \
            ak.prod(1-eff[~pass_wp], axis=1)
    probData_nom = ak.prod(effXsf[pass_wp], axis=1) * \
            ak.prod(1-effXsf[~pass_wp], axis=1)
    probData_up = ak.prod(effXsf_up[pass_wp], axis=1) * \
            ak.prod(1-effXsf_up[~pass_wp], axis=1)
    probData_dn = ak.prod(effXsf_dn[pass_wp], axis=1) * \
            ak.prod(1-effXsf_dn[~pass_wp], axis=1)

    wt_nom = probData_nom / probMC
    wt_up = probData_up / probMC
    wt_dn = probData_dn / probMC

    wt_nom = ak.to_numpy(wt_nom)
    wt_nom[probMC==0] = 1
    wt_up = ak.to_numpy(wt_up)    
    wt_up[probMC==0] = 1
    wt_dn = ak.to_numpy(wt_dn)
    wt_dn[probMC==0] = 1

    weights.add('wt_btagSF', wt_nom, wt_up, wt_dn)


