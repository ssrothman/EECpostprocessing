import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def get_effs(wp, pt, abseta, flav, cset_eff, num):
    eff_nom = cset_eff['Beff'].evaluate(
        wp, 
        'nominal',
        pt,
        abseta,
        flav
    )
    eff_up = cset_eff['Beff'].evaluate(
        wp,
        'up',
        pt,
        abseta,
        flav
    )
    eff_dn = cset_eff['Beff'].evaluate(
        wp,
        'down',
        pt,
        abseta,
        flav
    )

    eff_nom = ak.unflatten(eff_nom, num, axis=0)
    eff_up = ak.unflatten(eff_up, num, axis=0)
    eff_dn = ak.unflatten(eff_dn, num, axis=0)

    return eff_nom, eff_up, eff_dn

def get_sf(wp, pt, abseta, flav, cset_sf, variation, num):
    wpstrs = {'loose' : 'L', 'medium' : 'M', 'tight' : 'T'}

    light = flav == 0

    sf_light = cset_sf['deepJet_incl'].evaluate(
        variation,
        wpstrs[wp],
        flav[light],
        abseta[light],
        pt[light]
    )

    sf_heavy = cset_sf['deepJet_comb'].evaluate(
        variation,
        wpstrs[wp],
        flav[~light],
        abseta[~light],
        pt[~light]
    )
    
    sf = np.ones_like(pt)
    sf[light] = sf_light
    sf[~light] = sf_heavy

    sf = ak.unflatten(sf, num, axis=0)

    return sf

def get_sfs(wp, pt, abseta, flav, cset_sf, num):
    sf_nom = get_sf(wp, pt, abseta, flav, cset_sf, 'central', num)
    sf_up = get_sf(wp, pt, abseta, flav, cset_sf, 'up', num)
    sf_dn = get_sf(wp, pt, abseta, flav, cset_sf, 'down', num)

    return sf_nom, sf_up, sf_dn

def the_double_wp_sf(eff_smaller, eff_larger,
                     sf_smaller, sf_larger,
                     passWP_smaller, passWP_larger):
    cat1 = passWP_larger
    cat2 = passWP_smaller & ~passWP_larger
    cat3 = ~passWP_smaller

    PMC_cat1 = ak.prod(eff_larger[cat1], axis=1)
    PMC_cat2 = ak.prod((1-eff_larger[cat2]) *\
                       (eff_larger[cat2] - eff_smaller[cat2]),
                       axis=1)
    PMC_cat3 = ak.prod(1-eff_smaller[cat3], axis=1)

    PMC = PMC_cat1 * PMC_cat2 * PMC_cat3
    PMC = ak.where(PMC == 0, 1, PMC)

    SFeff_smaller = sf_smaller * eff_smaller
    SFeff_larger = sf_larger * eff_larger

    PData_cat1 = ak.prod(SFeff_larger[cat1], axis=1)
    PData_cat2 = ak.prod((1-SFeff_larger[cat2]) *\
                        (SFeff_larger[cat2] - SFeff_smaller[cat2]),
                        axis=1)
    PData_cat3 = ak.prod(1-SFeff_smaller[cat3], axis=1)

    PData = PData_cat1 * PData_cat2 * PData_cat3
    PData = ak.where(PData == 0, 1, PData)

    w = PData / PMC

    return w

def double_wp_btagSF(weights,
                     pt, abseta, flav, 
                     passWP_smaller, passWP_larger,
                     num,
                     smaller, larger, 
                     cset_sf, cset_eff):
    eff_smaller = get_effs(smaller, pt, abseta, flav, cset_eff, num)
    eff_larger  = get_effs(larger,  pt, abseta, flav, cset_eff, num)
    sf_smaller  = get_sfs(smaller,  pt, abseta, flav, cset_sf , num)
    sf_larger   = get_sfs(larger,   pt, abseta, flav, cset_sf , num)

    passWP_smaller = ak.unflatten(passWP_smaller, num, axis=0)
    passWP_larger = ak.unflatten(passWP_larger, num, axis=0)

    NOM = 0
    UP = 1
    DN = 2
    w_nom = the_double_wp_sf(eff_smaller[NOM], eff_larger[NOM],
                             sf_smaller[NOM],  sf_larger[NOM],
                             passWP_smaller, passWP_larger)

    w_upEff = the_double_wp_sf(eff_smaller[UP], eff_larger[UP],
                               sf_smaller[NOM],  sf_larger[NOM],
                               passWP_smaller, passWP_larger)
    w_dnEff = the_double_wp_sf(eff_smaller[DN], eff_larger[DN],
                               sf_smaller[NOM],  sf_larger[NOM],
                               passWP_smaller, passWP_larger)

    w_upTightEff = the_double_wp_sf(eff_smaller[NOM], eff_larger[UP],
                                    sf_smaller[NOM],  sf_larger[NOM],
                                    passWP_smaller, passWP_larger)
    w_dnTightEff = the_double_wp_sf(eff_smaller[NOM], eff_larger[DN],
                                    sf_smaller[NOM],  sf_larger[NOM],
                                    passWP_smaller, passWP_larger)

    w_upSF = the_double_wp_sf(eff_smaller[NOM], eff_larger[NOM],
                              sf_smaller[UP],  sf_larger[UP],
                              passWP_smaller, passWP_larger)
    w_dnSF = the_double_wp_sf(eff_smaller[NOM], eff_larger[NOM],
                              sf_smaller[DN],  sf_larger[DN],
                              passWP_smaller, passWP_larger)

    w_upTightSF = the_double_wp_sf(eff_smaller[NOM], eff_larger[NOM],
                                   sf_smaller[NOM],  sf_larger[UP],
                                   passWP_smaller, passWP_larger)
    w_dnTightSF = the_double_wp_sf(eff_smaller[NOM], eff_larger[NOM],
                                   sf_smaller[NOM],  sf_larger[DN],
                                   passWP_smaller, passWP_larger)

    weights.add_multivariation(
        'wt_btagSF', w_nom,
        modifierNames = ['eff', 'tighteff', 'sf', 'tightsf'],
        weightsUp=[w_upEff, w_upTightEff, w_upSF, w_upTightSF],
        weightsDown=[w_dnEff, w_dnTightEff, w_dnSF, w_dnTightSF]
    )

def the_single_wp_sf(eff, sf, passWP):
    passMC = ak.prod(eff[passWP], axis=1)
    failMC = ak.prod(1-eff[~passWP], axis=1)

    PMC = passMC * failMC
    PMC = ak.where(PMC == 0, 1, PMC)

    SFeff = sf * eff

    passData = ak.prod(SFeff[passWP], axis=1)
    failData = ak.prod(1-SFeff[~passWP], axis=1)

    PData = passData * failData
    PData = ak.where(PData == 0, 1, PData)

    w = PData / PMC

    return w

def single_wp_btagSF(weights,
                     pt, abseta, flav, passWP, num,
                     wp, cset_sf, cset_eff):

    eff_nom, eff_up, eff_dn = get_effs(wp, pt, abseta, flav, cset_eff)
    sf_nom, sf_up, sf_dn = get_sfs(wp, pt, abseta, flav, cset_sf)

    passWP = ak.unflatten(passWP, num, axis=0)

    NOM = 0
    UP = 1
    DN = 2

    w_nom = the_single_wp_sf(eff_nom[NOM], sf_nom[NOM], passWP)
    w_upEff = the_single_wp_sf(eff_up[UP], sf_nom[NOM], passWP)
    w_dnEff = the_single_wp_sf(eff_dn[DN], sf_nom[NOM], passWP)
    w_upSF = the_single_wp_sf(eff_nom[NOM], sf_up[UP], passWP)
    w_dnSF = the_single_wp_sf(eff_nom[NOM], sf_dn[DN], passWP)

    weights.add_multivariation(
        'wt_btagSF', w_nom,
        modifierNames = ['eff', 'sf'],
        weightsUp=[w_upEff, w_upSF],
        weightsDown=[w_dnEff, w_dnSF]
    )

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
