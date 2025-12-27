import awkward as ak
import numpy as np

from util.util import unflatVector

class Jets:
    def __init__(self, 
                 events : ak.Array, 
                 jetsname : str,
                 simonjetsname : str,
                 CHSjetsname : str):

        self._jetsname = jetsname
        self._simonjetsname = simonjetsname
        self._CHSjetsname = CHSjetsname

        self._parts : ak.Array = ak.materialize(
            events[simonjetsname]
        )

        self._simonjets : ak.Array = ak.materialize(
            events[simonjetsname+"BK"]
        )

        self._parts : ak.Array = unflatVector(
            self._parts,
            self._simonjets.nPart
        )

        #we sometimes pad with zero-pt parts, 
        #so nParts is wrong and needs to be recalculated
        self._parts : ak.Array= self._parts[self._parts.pt > 0]
        self._simonjets['nPart'] = ak.num(self._parts.pt)

        self._jets : ak.Array = ak.materialize(
            events[jetsname]
        )
        self._jets = self._jets[self._simonjets.iJet]

        if CHSjetsname:
            self._matchedCHS : ak.Array = ak.materialize(
                events[CHSjetsname]
            )
            nCHS = ak.flatten(
                self._simonjets.nCHS, 
                axis=1
            )
            iCHS = ak.materialize(
                events[simonjetsname+'CHS'].idx
            )

            bad = iCHS == 99999999
            iCHS = ak.where(bad, 0, iCHS)

            self._matchedCHS = ak.unflatten(
                self._matchedCHS[iCHS],
                nCHS
            )
            bad = ak.unflatten(
                bad, 
                nCHS
            )
            
            self._matchedCHS['pt'] = ak.where(
                bad, 
                0, 
                self._matchedCHS.pt
            )
            self._matchedCHS = self._matchedCHS[
                self._matchedCHS.pt > 0
            ]

    @property
    def parts(self) -> ak.Array:
        return self._parts

    @property  
    def simonjets(self) -> ak.Array:
        return self._simonjets

    @property  
    def jets(self) -> ak.Array:
        return self._jets 

    @property
    def matchedCHSjets(self) -> ak.Array:
        if hasattr(self, '_matchedCHS'):
            return self._matchedCHS     
        else:
            raise AttributeError("No CHS jets were loaded")     
    
    @property
    def jetsname(self) -> str:
        return self._jetsname   
    
    @property
    def simonjetsname(self) -> str:
        return self._simonjetsname
    
    @property
    def CHSjetsname(self) -> str:
        return self._CHSjetsname