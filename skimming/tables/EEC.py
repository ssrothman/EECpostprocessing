import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects import EEC
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any, Literal, assert_never
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

from typing import Sequence

_EECobjs = Literal['total', 'unmatched', 'untransfered']

_PROJ_R_BINS = np.array([
    0.001, 0.001193, 0.001423, 0.001697, 0.002024, 0.002414, 0.00288, 0.003435,
    0.004097, 0.004886, 0.005828, 0.006952, 0.008292, 0.00989, 0.011797, 0.014071,
    0.016783, 0.020019, 0.023878, 0.02848, 0.033971, 0.040519, 0.04833, 0.057646,
    0.068758, 0.082012, 0.097821, 0.116678, 0.139169, 0.165997, 0.197995, 0.236162,
    0.281685, 0.335985, 0.400751, 0.478002, 0.570145, 0.680049, 0.811139, 0.967499,
    1.153999, 1.376451, 1.641783, 1.958262, 2.335748, 2.786, 3.323046, 3.963615,
    4.727664, 5.638995, 6.726
])
_PROJ_R_CENTERS = 0.5 * (_PROJ_R_BINS[:-1] + _PROJ_R_BINS[1:])

class EECgenericTable:
    def __init__(self):
        pass

    def _table_observed(self,
                       objs : AllObjects, 
                       evtsel : PackedSelection, 
                       jetsel : PackedJetSelection, 
                       weights : Weights,
                       gen : bool,
                       whichEECdistr : str,
                       whichEECobj : _EECobjs,
                       binning_coords : Sequence[str],
                       order : int):
        '''
        whichEECdistr: str is the name of the EEC distribution attribute in the EEC object
            eg 'tee', 'triangle', 'dipole', 'proj', or 'res3'
        whichEECobj: str is one of ['total', 'unmatched', 'untransfered']
        '''
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)

        if gen:
            if whichEECobj == 'total':
                theEECs = objs.GenEEC
            elif whichEECobj == 'unmatched':
                theEECs = objs.UnmatchedGenEEC
            elif whichEECobj == 'untransfered':
                theEECs = objs.UntransferedGenEEC
            else:
                assert_never(whichEECobj)

            thejets = objs.GenJets
            iReco = theEECs.jetidx_reco
        else:
            if whichEECobj == 'total':
                theEECs = objs.RecoEEC
            elif whichEECobj == 'unmatched':
                theEECs = objs.UnmatchedRecoEEC
            elif whichEECobj == 'untransfered':
                theEECs = objs.UntransferedRecoEEC
            else:
                assert_never(whichEECobj)

            thejets = objs.RecoJets
            iReco = theEECs.jetidx_reco

        iReco = ak.values_astype(iReco, np.int32)
        EECmask = jetmask[iReco]

        #jet properties
        iJet = ak.values_astype(theEECs.jetidx, np.int32)
        thevals['Jpt'] = thejets.jets.pt[iJet][EECmask][evtmask]
        thevals['Jeta'] = thejets.jets.eta[iJet][EECmask][evtmask]
        
        if not gen:
            if hasattr(thejets.simonjets, 'passLooseB'):
                thevals['passLooseB'] = thejets.simonjets.passLooseB[iJet][EECmask][evtmask]
            if hasattr(thejets.simonjets, 'passMediumB'):
                thevals['passMediumB'] = thejets.simonjets.passMediumB[iJet][EECmask][evtmask]
            if hasattr(thejets.simonjets, 'passTightB'):
                thevals['passTightB'] = thejets.simonjets.passTightB[iJet][EECmask][evtmask]
        
        if hasattr(thejets.jets, 'hadronFlavour'):
            thevals['flav'] = thejets.jets.hadronFlavour[iJet][EECmask][evtmask]
        
        thevals['ptdenom'] = theEECs.ptdenom[EECmask][evtmask]

        correction = np.power(
            thevals['ptdenom']/thevals['Jpt'], 
            order
        )

        # EEC values
        EECvals = getattr(theEECs, whichEECdistr)
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

        if hasattr(recojets.simonjets, 'passLooseB'):
            thevals['passLooseB'] = recojets.simonjets.passLooseB[iReco][EECmask][evtmask]
        if hasattr(recojets.simonjets, 'passMediumB'):  
            thevals['passMediumB'] = recojets.simonjets.passMediumB[iReco][EECmask][evtmask]
        if hasattr(recojets.simonjets, 'passTightB'):
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
    
class EECprojObsTable(EECgenericTable):
    def __init__(self, gen: bool, whichEECobj: _EECobjs):
        self._gen = gen
        self._whichEECobj = whichEECobj

    @property
    def name(self) -> str:
        level = 'Gen' if self._gen else 'Reco'
        if self._whichEECobj == 'total':
            return 'proj_%s' % level
        return 'proj_%s%s' % (self._whichEECobj, level)

    def run_table(self,
                  objs: AllObjects,
                  evtsel: PackedSelection,
                  jetsel: PackedJetSelection,
                  weights: Weights):
        table = self._table_observed(
            objs, evtsel, jetsel, weights,
            self._gen,
            'proj',
            self._whichEECobj,
            ['R'],
            2
        )
        R_idx = table['R'].to_pylist()
        R_float = pa.array([float(_PROJ_R_CENTERS[i]) for i in R_idx], type=pa.float32())
        col_idx = table.schema.get_field_index('R')
        return table.set_column(col_idx, 'R', R_float)


class EECres4ObsTable(EECgenericTable):
    def __init__(self, gen : bool, whichEECdistr : str, whichEECobj : _EECobjs):
        self._gen = gen
        self._whichEECdistr = whichEECdistr
        self._whichEECobj : _EECobjs = whichEECobj

    @property
    def name(self) -> str:
        thename = 'res4%s' % self._whichEECdistr
        if self._gen:
            thename += '_%sGen' % self._whichEECobj
        else:
            thename += '_%sReco' % self._whichEECobj

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
            self._whichEECdistr,
            self._whichEECobj,
            binning_coords,
            order
        )

class EECres4TransferTable(EECgenericTable):
    def __init__(self, whichEECdistr : str):
        self._whichEECdistr = whichEECdistr

    @property
    def name(self) -> str:
        thename = 'res4%s_transfer' % self._whichEECdistr
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
            self._whichEECdistr,
            binning_coords,
            order
        )