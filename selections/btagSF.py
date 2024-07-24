import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def get_effs(wp, pt, abseta, flav, cset_eff):
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

    return eff_nom, eff_up, eff_dn

def get_sf(wp, pt, abseta, flav, cset_sf, variation):
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

    return sf

def get_sfs(wp, pt, abseta, flav, cset_sf):
    sf_nom = get_sf(wp, pt, abseta, flav, cset_sf, 'central')
    sf_up = get_sf(wp, pt, abseta, flav, cset_sf, 'up')
    sf_dn = get_sf(wp, pt, abseta, flav, cset_sf, 'down')

    return sf_nom, sf_up, sf_dn

def double_wp_btagSF(weights,
                     pt, abseta, flav, 
                     passWP_smaller, passWP_larger,
                     num,
                     smaller, larger, 
                     cset_sf, cset_eff):
    eff_nom_smaller, eff_up_smaller, eff_dn_smaller = get_effs(smaller, pt, abseta, flav, cset_eff)
    eff_nom_larger, eff_up_larger, eff_dn_larger = get_effs(larger, pt, abseta, flav, cset_eff)
    sf_nom_smaller, sf_up_smaller, sf_dn_smaller = get_sfs(smaller, pt, abseta, flav, cset_sf)
    sf_nom_larger, sf_up_larger, sf_dn_larger = get_sfs(larger, pt, abseta, flav, cset_sf)

    eff_nom_smaller = ak.unflatten(eff_nom_smaller, num, axis=0)
    eff_up_smaller = ak.unflatten(eff_up_smaller, num, axis=0)
    eff_dn_smaller = ak.unflatten(eff_dn_smaller, num, axis=0)
    eff_nom_larger = ak.unflatten(eff_nom_larger, num, axis=0)
    eff_up_larger = ak.unflatten(eff_up_larger, num, axis=0)
    eff_dn_larger = ak.unflatten(eff_dn_larger, num, axis=0)
    sf_nom_smaller = ak.unflatten(sf_nom_smaller, num, axis=0)
    sf_up_smaller = ak.unflatten(sf_up_smaller, num, axis=0)
    sf_dn_smaller = ak.unflatten(sf_dn_smaller, num, axis=0)
    sf_nom_larger = ak.unflatten(sf_nom_larger, num, axis=0)
    sf_up_larger = ak.unflatten(sf_up_larger, num, axis=0)
    sf_dn_larger = ak.unflatten(sf_dn_larger, num, axis=0)

    passWP_smaller = ak.unflatten(passWP_smaller, num, axis=0)
    passWP_larger = ak.unflatten(passWP_larger, num, axis=0)

    cat1 = passWP_larger
    cat2 = passWP_smaller & ~passWP_larger
    cat3 = ~passWP_smaller

    PMC_nom_cat1 = ak.prod(eff_nom_larger[cat1], axis=1) 
    PMC_nom_cat2 = ak.prod((1-eff_nom_larger[cat2]) *\
                           (eff_nom_larger[cat2] - eff_nom_smaller[cat2]),
                           axis=1)
    PMC_nom_cat3 = ak.prod(1-eff_nom_smaller[cat3], axis=1)

    PMC_upEff_cat1 = ak.prod(eff_up_larger[cat1], axis=1)
    PMC_upEff_cat2 = ak.prod((1-eff_up_larger[cat2]) *\
                            (eff_up_larger[cat2] - eff_up_smaller[cat2]),
                            axis=1)
    PMC_upEff_cat3 = ak.prod(1-eff_up_smaller[cat3], axis=1)

    PMC_dnEff_cat1 = ak.prod(eff_dn_larger[cat1], axis=1)
    PMC_dnEff_cat2 = ak.prod((1-eff_dn_larger[cat2]) *\
                            (eff_dn_larger[cat2] - eff_dn_smaller[cat2]),
                            axis=1)
    PMC_dnEff_cat3 = ak.prod(1-eff_dn_smaller[cat3], axis=1)

    PMC_nom = PMC_nom_cat1 * PMC_nom_cat2 * PMC_nom_cat3
    PMC_upEff = PMC_upEff_cat1 * PMC_upEff_cat2 * PMC_upEff_cat3
    PMC_dnEff = PMC_dnEff_cat1 * PMC_dnEff_cat2 * PMC_dnEff_cat3

    SFeff_nom_larger = sf_nom_larger * eff_nom_larger
    SFeff_nom_smaller = sf_nom_smaller * eff_nom_smaller
    SFeff_upEff_larger = sf_nom_larger * eff_up_larger
    SFeff_upEff_smaller = sf_nom_smaller * eff_up_smaller
    SFeff_dnEff_larger = sf_nom_larger * eff_dn_larger
    SFeff_dnEff_smaller = sf_nom_smaller * eff_dn_smaller
    SFeff_upSF_larger = sf_up_larger * eff_nom_larger
    SFeff_upSF_smaller = sf_up_smaller * eff_nom_smaller
    SFeff_dnSF_larger = sf_dn_larger * eff_nom_larger
    SFeff_dnSF_smaller = sf_dn_smaller * eff_nom_smaller

    PData_nom_cat1 = ak.prod(SFeff_nom_larger[cat1], axis=1)
    PData_nom_cat2 = ak.prod((1-SFeff_nom_larger[cat2]) *\
                            (SFeff_nom_larger[cat2] - SFeff_nom_smaller[cat2]),
                            axis=1)
    PData_nom_cat3 = ak.prod(1-SFeff_nom_smaller[cat3], axis=1)

    PData_upEff_cat1 = ak.prod(SFeff_upEff_larger[cat1], axis=1)
    PData_upEff_cat2 = ak.prod((1-SFeff_upEff_larger[cat2]) *\
                            (SFeff_upEff_larger[cat2] - SFeff_upEff_smaller[cat2]),
                            axis=1)
    PData_upEff_cat3 = ak.prod(1-SFeff_upEff_smaller[cat3], axis=1)

    PData_dnEff_cat1 = ak.prod(SFeff_dnEff_larger[cat1], axis=1)
    PData_dnEff_cat2 = ak.prod((1-SFeff_dnEff_larger[cat2]) *\
            (SFeff_dnEff_larger[cat2] - SFeff_dnEff_smaller[cat2]),
            axis=1)
    PData_dnEff_cat3 = ak.prod(1-SFeff_dnEff_smaller[cat3], axis=1)

    PData_upSF_cat1 = ak.prod(SFeff_upSF_larger[cat1], axis=1)
    PData_upSF_cat2 = ak.prod((1-SFeff_upSF_larger[cat2]) *\
            (SFeff_upSF_larger[cat2] - SFeff_upSF_smaller[cat2]),
            axis=1)
    PData_upSF_cat3 = ak.prod(1-SFeff_upSF_smaller[cat3], axis=1)

    PData_dnSF_cat1 = ak.prod(SFeff_dnSF_larger[cat1], axis=1)
    PData_dnSF_cat2 = ak.prod((1-SFeff_dnSF_larger[cat2]) *\
            (SFeff_dnSF_larger[cat2] - SFeff_dnSF_smaller[cat2]),
            axis=1)
    PData_dnSF_cat3 = ak.prod(1-SFeff_dnSF_smaller[cat3], axis=1)

    PData_nom = PData_nom_cat1 * PData_nom_cat2 * PData_nom_cat3
    PData_upEff = PData_upEff_cat1 * PData_upEff_cat2 * PData_upEff_cat3
    PData_dnEff = PData_dnEff_cat1 * PData_dnEff_cat2 * PData_dnEff_cat3
    PData_upSF = PData_upSF_cat1 * PData_upSF_cat2 * PData_upSF_cat3
    PData_dnSF = PData_dnSF_cat1 * PData_dnSF_cat2 * PData_dnSF_cat3

    w_nom = PData_nom / PMC_nom
    w_upEff = PData_upEff / PMC_upEff
    w_dnEff = PData_dnEff / PMC_dnEff
    w_upSF = PData_upSF / PMC_nom
    w_dnSF = PData_dnSF / PMC_nom

    w_nom = np.nan_to_num(w_nom)
    w_upEff = np.nan_to_num(w_upEff)
    w_dnEff = np.nan_to_num(w_dnEff)
    w_upSF = np.nan_to_num(w_upSF)
    w_dnSF = np.nan_to_num(w_dnSF)

    weights.add_multivariation(
        'wt_btagSF', w_nom,
        modifierNames = ['eff', 'sf'],
        weightsUp=[w_upEff, w_upSF],
        weightsDown=[w_dnEff, w_dnSF]
    )

def single_wp_btagSF(weights,
                     pt, abseta, flav, passWP, num,
                     wp, cset_sf, cset_eff):

    eff_nom, eff_up, eff_dn = get_effs(wp, pt, abseta, flav, cset_eff)
    sf_nom, sf_up, sf_dn = get_sfs(wp, pt, abseta, flav, cset_sf)

    eff_nom = ak.unflatten(eff, num, axis=0)
    eff_up = ak.unflatten(eff_up, num, axis=0)
    eff_dn = ak.unflatten(eff_dn, num, axis=0)

    sf_nom = ak.unflatten(sf_nom, num, axis=0)
    sf_up = ak.unflatten(sf_up, num, axis=0)
    sf_dn = ak.unflatten(sf_dn, num, axis=0)

    passWP = ak.unflatten(passWP, num, axis=0)

    passMC_nom = ak.prod(eff_nom[passWP], axis=1)
    failMC_nom = ak.prod(1-eff_nom[~passWP], axis=1)
    passMC_upEff = ak.prod(eff_up[passWP], axis=1)
    failMC_upEff = ak.prod(1-eff_up[~passWP], axis=1)
    passMC_dnEff = ak.prod(eff_dn[passWP], axis=1)
    failMC_dnEff = ak.prod(1-eff_dn[~passWP], axis=1)

    PMC_nom = passMC_nom * failMC_nom
    PMC_upEff = passMC_upEff * failMC_upEff
    PMC_dnEff = passMC_dnEff * failMC_dnEff

    SFeff = sf_nom * eff_nom
    SFeff_upEff = sf_nom * eff_up
    SFeff_dnEff = sf_nom * eff_dn
    SFeff_upSF = sf_up * eff_nom
    SFeff_dnSF = sf_dn * eff_nom

    passData_nom = ak.prod(SFeff[passWP], axis=1)
    failData_nom = ak.prod(1-SFeff[~passWP], axis=1)
    passData_upEff = ak.prod(SFeff_upEff[passWP], axis=1)
    failData_upEff = ak.prod(1-SFeff_upEff[~passWP], axis=1)
    passData_dnEff = ak.prod(SFeff_dnEff[passWP], axis=1)
    failData_dnEff = ak.prod(1-SFeff_dnEff[~passWP], axis=1)
    passData_upSF = ak.prod(SFeff_upSF[passWP], axis=1)
    failData_upSF = ak.prod(1-SFeff_upSF[~passWP], axis=1)
    passData_dnSF = ak.prod(SFeff_dnSF[passWP], axis=1)
    failData_dnSF = ak.prod(1-SFeff_dnSF[~passWP], axis=1)

    PData_nom = passData_nom * failData_nom
    PData_upEff = passData_upEff * failData_upEff
    PData_dnEff = passData_dnEff * failData_dnEff
    PData_upSF = passData_upSF * failData_upSF
    PData_dnSF = passData_dnSF * failData_dnSF

    w_nom = PData_nom / PMC_nom
    w_upEff = PData_upEff / PMC_upEff
    w_dnEff = PData_dnEff / PMC_dnEff
    w_upSF = PData_upSF / PMC_nom
    w_dnSF = PData_dnSF / PMC_nom

    w_nom = np.nan_to_num(w_nom)
    w_upEff = np.nan_to_num(w_upEff)
    w_dnEff = np.nan_to_num(w_dnEff)
    w_upSF = np.nan_to_num(w_upSF)
    w_dnSF = np.nan_to_num(w_dnSF)

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
