import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def getEventWeight(x, muons, config):
    ans = Weights(len(x), storeIndividual=True)

    if hasattr(x, "genWeight"):
        ans.add('generator', x.genWeight)
        if config.eventSelection.PreFireWeight:
            ans.add("prefire", x.L1PreFiringWeight.Nom)
        
        #sfs are from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/POG/MUO/2018_UL/muon_Z_v2.json.gz?ref_type=heads
        cset = CorrectionSet.from_file(config.sfpath)

        leadmu = muons[:,0]
        submu = muons[:,1]

        leadpt = leadmu.pt/leadmu.RoccoR
        leadpt = ak.fill_none(leadpt, 0)

        leadeta = np.abs(leadmu.eta)
        leadeta = ak.fill_none(leadeta, 0)
        
        badleadpt = leadpt <= config.muonSelection.leadpt
        leadpt = np.where(badleadpt, config.muonSelection.leadpt, leadpt)

        badleadeta = leadeta > config.muonSelection.leadeta-0.01
        leadeta = np.where(badleadeta, config.muonSelection.leadeta-0.01, leadeta)

        subpt = submu.pt/submu.RoccoR
        subpt = ak.fill_none(subpt, 0)

        subeta = np.abs(submu.eta)
        subeta = ak.fill_none(subeta, 0)

        badsubpt = subpt <= config.muonSelection.subpt
        subpt = np.where(badsubpt, config.muonSelection.subpt, subpt)
        
        badsubeta = subeta > config.muonSelection.subeta-0.01
        subeta = np.where(badsubeta, config.muonSelection.subeta-0.01, subeta)

        if config.muonSelection.ID == 'loose':
            idsfname = 'NUM_LooseID_DEN_genTracks'

            if config.muonSelection.iso == 'loose':
                isosfname = 'NUM_LooseRelIso_DEN_LooseID'
            else:
                raise NotImplementedError("SF for %s iso on loose ID not available"%config.muonSelection.iso)

        elif config.muonSelection.ID == 'medium':
            idsfname = 'NUM_MediumID_DEN_genTracks'

            if config.muonSelection.iso == 'loose':
                isosfname = 'NUM_LooseRelIso_DEN_MediumID'
            elif config.muonSelection.iso == 'tight':
                isosfname = 'NUM_TightRelIso_DEN_MediumID'
            else:
                raise NotImplementedError("SF for %s iso on medium ID not available"%config.muonSelection.iso)
        elif config.muonSelection.ID == 'tight':
            idsfname = 'NUM_TightID_DEN_genTracks'

            if config.muonSelection.iso == 'loose':
                isosfname = 'NUM_LooseRelIso_DEN_TightIDandIPCut'
            elif config.muonSelection.iso == 'tight':
                isosfname = 'NUM_TightRelIso_DEN_TightIDandIPCut'
            else:
                raise NotImplementedError("SF for %s iso on tight ID not available"%config.muonSelection.iso)
        else:
            raise NotImplementedError("SF for %d ID is not available"%config.muonSelection.ID)

        if config.eventSelection.trigger == 'IsoMu24':
            # Trigger SF
            triggersfname = 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight'
        else:
            raise NotImplementedError(f"SF for trigger {config.eventSelection.trigger} not implemented")


        id_sf_lead = cset[idsfname].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )
        id_sf_sub = cset[idsfname].evaluate(
            '2018_UL',
            subeta,
            subpt,
            'sf'
        )
        iso_sf_lead = cset[isosfname].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )
        iso_sf_sub = cset[isosfname].evaluate(
            '2018_UL',
            subeta,
            subpt,
            'sf'
        )
        trigger_sf = cset[triggersfname].evaluate(
            '2018_UL',
            leadeta,
            leadpt,
            'sf'
        )
        
        trigger_sf = np.where(badleadpt | badleadeta, 0, trigger_sf)

        id_sf_lead = np.where(badleadpt | badleadeta, 0, id_sf_lead)
        iso_sf_lead = np.where(badleadpt | badleadeta, 0, iso_sf_lead)

        id_sf_sub  = np.where(badsubpt | badsubeta, 0, id_sf_sub)
        iso_sf_sub = np.where(badsubpt | badsubeta, 0, iso_sf_sub)

        ans.add("idsf_lead", id_sf_lead)
        ans.add("isosf_lead", iso_sf_lead)
        ans.add("idsf_sub", id_sf_sub)
        ans.add("isosf_sub", iso_sf_sub)
        ans.add("triggersf", trigger_sf)

    return ans
