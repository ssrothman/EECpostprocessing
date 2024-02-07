import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def getEventWeight(x, muons, config):
    ans = Weights(len(x))
    ans.add('generator', x.genWeight)
    if config.eventSelection.PreFireWeight:
        ans.add("prefire", x.L1PreFiringWeight.Nom)
    
    #sfs are from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/POG/MUO/2018_UL/muon_Z_v2.json.gz?ref_type=heads
    cset = CorrectionSet.from_file(config.sfpath)

    ilead = ak.argmax(muons.pt, axis=1)
    leadmu = muons[ilead]

    leadpt = ak.firsts(leadmu.pt)
    leadpt = ak.fill_none(leadpt, 0)

    leadeta = np.abs(ak.firsts(leadmu.eta))
    leadeta = ak.fill_none(leadeta, 0)
    
    badpt = leadpt <= 26
    leadpt = np.where(badpt, 26, leadpt)

    badeta = leadeta >= 2.4
    leadeta = np.where(badeta, 2.3999, leadeta)

    if config.muonSelection.ID == 'loose':
        id_sf = cset['NUM_LooseID_DEN_genTracks'].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )
        if config.muonSelection.iso == 'loose':
            iso_sf = cset['NUM_LooseRelIso_DEN_LooseID'].evaluate(
                '2018_UL',
                leadeta,
                leadpt,
                'sf'
            )
        else:
            raise NotImplementedError("SF for %s iso on loose ID not available"%config.muonSelection.iso)

    elif config.muonSelection.ID == 'medium':
        id_sf = cset['NUM_MediumID_DEN_genTracks'].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )

        if config.muonSelection.iso == 'loose':
            iso_sf = cset['NUM_LooseRelIso_DEN_MediumID'].evaluate(
                '2018_UL',
                leadeta,
                leadpt,
                'sf'
            )
        elif config.muonSelection.iso == 'tight':
            iso_sf = cset['NUM_TightRelIso_DEN_MediumID'].evaluate(
                '2018_UL',
                leadeta,
                leadpt,
                'sf'
            )
        else:
            raise NotImplementedError("SF for %s iso on medium ID not available"%config.muonSelection.iso)
    elif config.muonSelection.ID == 'tight':
        id_sf = cset['NUM_TightID_DEN_genTracks'].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )

        if config.muonSelection.iso == 'loose':
            iso_sf = cset['NUM_LooseRelIso_DEN_TightIDandIPCut'].evaluate(
                '2018_UL',
                leadeta,
                leadpt,
                'sf'
            )
        elif config.muonSelection.iso == 'tight':
    #q = cset['NUM_LooseID_DEN_genTracks']
    #for a in q.inputs:
    #    print(a.name)
    #    print("\t", a.type)
    #    print("\t", a.description)
            iso_sf = cset['NUM_TightRelIso_DEN_TightIDandIPCut'].evaluate(
                '2018_UL',
                leadeta,
                leadpt,
                'sf'
            )
        else:
            raise NotImplementedError("SF for %s iso on tight ID not available"%config.muonSelection.iso)
    else:
        raise NotImplementedError("SF for %d ID is not available"%config.muonSelection.ID)

    if config.eventSelection.trigger == 'IsoMu24':
        # Trigger SF
        trigger_sf = cset["NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )
    else:
        raise NotImplementedError(f"SF for trigger {config.eventSelection.trigger} not implemented")

    id_sf = np.where(badpt | badeta, 1, id_sf)
    iso_sf = np.where(badpt | badeta, 1, iso_sf)
    trigger_sf = np.where(badpt | badeta, 1, trigger_sf)

    ans.add("idsf", id_sf)
    ans.add("isosf", iso_sf)
    ans.add("triggersf", trigger_sf)

    return ans
