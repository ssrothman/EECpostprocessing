import awkward as ak
from .common import unflatVector
import numpy as np

class EECgeneric:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _setup_generic(self, kind: str):
        result = ak.materialize(self._events[self._name + kind])

        #due to an oversight, there are two different spellings for nEntry
        if hasattr(self._events[self._name + 'BK'], 'nEntry_' + kind):
            nEntry = ak.materialize(self._events[self._name + 'BK']['nEntry_' + kind])
        else:
            nEntry = ak.materialize(self._events[self._name + 'BK']['nEntries_' + kind])

        #due to an overwight, nEntry may have the wrong dtype
        nEntry = ak.values_astype(nEntry, np.int32)

        result = unflatVector(result, nEntry)

        # obj EECs have `wt`, transfer EECs have `wt_gen`
        if hasattr(result, 'wt'):
            result = result[result.wt > 0] 
        else:
            result = result[result.wt_gen > 0]

        return result
    
    '''
    I've provided definitions for ptdenom and jetidx for obs EECs
    and their gen/reco counterparts for transfer EECs.

    Calling the wrong one for the wrong type of EEC will likely
    result in an error.
    '''

    @property
    def ptdenom(self):
        return ak.materialize(
            self._events[self._name + 'BK'].pt_denom
        )
    
    @property
    def jetidx(self):
        return ak.materialize(
            self._events[self._name + 'BK'].iJet
        )

    @property
    def ptdenom_gen(self):
        return ak.materialize(
            self._events[self._name + 'BK'].pt_denom_gen
        )
    
    @property
    def jetidx_gen(self):
        return ak.materialize(
            self._events[self._name + 'BK'].iJet_gen
        )
    
    @property
    def ptdenom_reco(self):
        return ak.materialize(
            self._events[self._name + 'BK'].pt_denom_reco
        )
    
    @property
    def jetidx_reco(self):
        return ak.materialize(
            self._events[self._name + 'BK'].iJet_reco
        )

class EECproj(EECgeneric):
    def __init__(self, events, name):
        super().__init__(events, name)

        self._proj = None

    @property
    def proj(self):
        if self._proj is None:
            self._setup_proj()

        return self._proj
    
    def _setup_proj(self):
        self._proj = self._setup_generic('proj')

class EECres3(EECgeneric):
    def __init__(self, events, name):
        super().__init__(events, name)

        self._res3 = None

    @property
    def res3(self):
        if self._res3 is None:
            self._setup_res3()

        return self._res3
    
    def _setup_res3(self):
        self._res3 = self._setup_generic('res3')

class EECres4(EECgeneric):
    def __init__(self, events, name):
        super().__init__(events, name)

        self._tee = None
        self._dipole = None
        self._triangle = None
    
    @property
    def tee(self):
        if self._tee is None:
            self._setup_tee()

        return self._tee
    
    @property
    def dipole(self):
        if self._dipole is None:
            self._setup_dipole()

        return self._dipole
    
    @property
    def triangle(self):
        if self._triangle is None:
            self._setup_triangle()

        return self._triangle
    
    def _setup_tee(self):
        self._tee = self._setup_generic('tee')

    def _setup_dipole(self):
        self._dipole = self._setup_generic('dipole')

    def _setup_triangle(self):
        self._triangle = self._setup_generic('triangle')