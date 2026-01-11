import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects import EEC
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

from typing import Sequence

class EECgenericTable:
    def __init__(self):
        pass

    def _table_observed(self,
                       objs : AllObjects, 
                       evtsel : PackedSelection, 
                       jetsel : PackedJetSelection, 
                       weights : Weights,
                       gen : bool,
                       whichEEC : str,
                       binning_coords : Sequence[str],
                       order : int):
        
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)

        if gen:
            theEECs = objs.GenEEC
            thejets = objs.GenJets
            iReco = objs.EECtransfer.jetidx_reco
        else:
            theEECs = objs.RecoEEC
            thejets = objs.RecoJets
            iReco = theEECs.jetidx_reco

        iReco = ak.values_astype(iReco, np.int32)
        EECmask = jetmask[iReco]

        #jet properties
        iJet = ak.values_astype(theEECs.jetidx, np.int32)
        thevals['Jpt'] = thejets.jets.pt[iJet][EECmask][evtmask]
        thevals['Jeta'] = thejets.jets.eta[iJet][EECmask][evtmask]
        
        if not gen:
            thevals['passLooseB'] = thejets.simonjets.passLooseB[iJet][EECmask][evtmask]
            thevals['passMediumB'] = thejets.simonjets.passMediumB[iJet][EECmask][evtmask]
            thevals['passTightB'] = thejets.simonjets.passTightB[iJet][EECmask][evtmask]
        
        if hasattr(thejets.jets, 'hadronFlavour'):
            thevals['flav'] = thejets.jets.hadronFlavour[iJet][EECmask][evtmask]
        
        thevals['ptdenom'] = theEECs.ptdenom[EECmask][evtmask]

        correction = np.power(
            thevals['ptdenom']/thevals['Jpt'], 
            order
        )

        # EEC values
        EECvals = getattr(theEECs, whichEEC)
        for coord in binning_coords:
            thevals[coord] = getattr(EECvals, coord)[EECmask][evtmask]

        thevals['wt'] = EECvals.wt[EECmask][evtmask] * correction

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals['wt']

        broadcast_all(thevals, shape_target)

        return to_pa_table(thevals)

    def _table_transfer(self,
                       objs : AllObjects, 
                       evtsel : PackedSelection, 
                       jetsel : PackedJetSelection, 
                       weights : Weights,
                       whichEEC : str,
                       binning_coords : Sequence[str],
                       order : int):
        
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)

        iReco = objs.EECtransfer.jetidx_reco
        iGen = objs.EECtransfer.jetidx_gen
        iReco = ak.values_astype(iReco, np.int32)
        iGen = ak.values_astype(iGen, np.int32)

        EECmask = jetmask[iReco]

        theEECs = objs.EECtransfer

        #jet properties
        recojets = objs.RecoJets
        genjets = objs.GenJets

        thevals['Jpt_gen'] = genjets.jets.pt[iGen][EECmask][evtmask]
        thevals['Jeta_gen'] = genjets.jets.eta[iGen][EECmask][evtmask]

        thevals['Jpt_reco'] = recojets.jets.pt[iReco][EECmask][evtmask]
        thevals['Jeta_reco'] = recojets.jets.eta[iReco][EECmask][evtmask]

        thevals['passLooseB'] = recojets.simonjets.passLooseB[iReco][EECmask][evtmask]
        thevals['passMediumB'] = recojets.simonjets.passMediumB[iReco][EECmask][evtmask]
        thevals['passTightB'] = recojets.simonjets.passTightB[iReco][EECmask][evtmask]

        if hasattr(recojets.jets, 'hadronFlavour'):
            thevals['flav_reco'] = recojets.jets.hadronFlavour[iReco][EECmask][evtmask]
        if hasattr(genjets.jets, 'hadronFlavour'):
            thevals['flav_gen'] = genjets.jets.hadronFlavour[iGen][EECmask][evtmask]
        
        thevals['ptdenom_reco'] = theEECs.ptdenom_reco[EECmask][evtmask]
        thevals['ptdenom_gen'] = theEECs.ptdenom_gen[EECmask][evtmask]

        correction_gen = np.power(
            thevals['ptdenom_gen']/thevals['Jpt_gen'], 
            order
        )
        correction_reco = np.power(
            thevals['ptdenom_reco']/thevals['Jpt_reco'], 
            order
        )

        EECvals = getattr(theEECs, whichEEC)
        thevals['wt_gen'] = EECvals.wt_gen[EECmask][evtmask] * correction_gen
        thevals['wt_reco'] = EECvals.wt_reco[EECmask][evtmask] * correction_reco
        for coord in binning_coords:
            thevals[coord+'_gen'] = getattr(EECvals, coord + '_gen')[EECmask][evtmask]
            thevals[coord+'_reco'] = getattr(EECvals, coord + '_reco')[EECmask][evtmask]

        add_weight_variations(thevals, weights, evtmask)

        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals['wt_reco']
        broadcast_all(thevals, shape_target)

        return to_pa_table(thevals)
    
class EECres4ObsTable(EECgenericTable):
    def __init__(self, gen : bool, whichEEC : str):
        self._gen = gen
        self._whichEEC = whichEEC
        print("Initialized EECres4ObsTable:", self.name)

    @property
    def name(self) -> str:
        thename = 'res4%s' % self._whichEEC
        if self._gen:
            thename += '_gen'
        else:
            thename += '_reco'
        return thename

    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        binning_coords = ['R', 'r', 'c']
        order = 4
        return self._table_observed(
            objs,
            evtsel,
            jetsel,
            weights,
            self._gen,
            self._whichEEC,
            binning_coords,
            order
        )

class EECres4TransferTable(EECgenericTable):
    def __init__(self, whichEEC : str):
        self._whichEEC = whichEEC

    @property
    def name(self) -> str:
        thename = 'res4%s_transfer' % self._whichEEC
        return thename

    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        binning_coords = ['R', 'r', 'c']
        order = 4
        return self._table_transfer(
            objs,
            evtsel,
            jetsel,
            weights,
            self._whichEEC,
            binning_coords,
            order
        )