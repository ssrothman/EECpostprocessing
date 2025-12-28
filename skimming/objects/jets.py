from typing import Any
import warnings
import awkward as ak
import numpy as np

from util.util import unflatVector

class SimpleJets:
    '''
    Basic jet object with kinematic and jetID/PUID selections

    Supports check_btags()
    '''
    def __init__(self,
                 events : ak.Array,
                 name : str,
                 minpt : float,
                 maxeta : float,
                 minJetID : int,
                 minPUID : int,
                 PUID_ptthreshold : float,
                 index_by : ak.Array | None = None,
                 reshape_by : ak.Array | None = None):
        
        self._name = name
        self._jets : ak.Array = ak.materialize(
            events[self._name]
        )

        if index_by is not None:
            bad : Any = index_by == 99999999 # Any type to appease pyright
            index = ak.where(bad, 0, index_by)
            self._jets = self._jets[index]
            #set bad jets to pt=0
            #this way they will be removed by pt cuts later
            self._jets['pt'] = ak.where(
                bad,
                0,
                self._jets.pt
            )

        if reshape_by is not None:
            self._jets = ak.unflatten(
                self._jets,
                ak.flatten(reshape_by, axis=None), # pyright: ignore[reportArgumentType]
                axis=1
            )

        self._jets = self._jets[
            (self._jets.pt > minpt)
            & (np.abs(self._jets.eta) < maxeta)
        ]
        if minJetID >= 0:
            self._jets = self._jets[
                self._jets.jetId >= minJetID
            ]

        if minPUID >= 0:
            self._jets = self._jets[
                (self._jets.puId >= minPUID)
                | (self._jets.pt >= PUID_ptthreshold)
            ]
    

    def check_btags(self, btagger : str, wps : dict[str, float]) -> None:
        #check btag scores :D
        for wp_name, wp_value in wps.items():
            btag_score = self._jets[btagger]
            self._jets['pass%sB' % (wp_name)] = (
                btag_score > wp_value
            )
        
        if hasattr(self._jets, 'partonFlavour'):
            self._jets['hadronFlavour'] = ak.where(
                (self._jets.hadronFlavour == 0) & (self._jets.partonFlavour == 21),
                21,
                self._jets.hadronFlavour
            )
        
    @property  
    def jets(self) -> ak.Array:
        return self._jets

    @property
    def name(self) -> str:
        return self._name

class SimonJets:
    def __init__(self, 
                 events : ak.Array, 
                 jetsname : str,
                 simonjetsname : str,
                 CHSjetsname : str,
                 skipJEC : bool, #skipJEC implies skipJER AND skipJES and skipJUNC
                 skipJER : bool,
                 skipJES : bool,
                 skipJUNC : bool,
                 CHS_ptthreshold : float,
                 CHS_etathreshold : float,
                 CHS_minJetID : int,
                 CHS_minPUID : int,
                 CHS_PUID_ptthreshold : float):

        self._jetsname = jetsname
        self._simonjetsname = simonjetsname
        self._CHSjetsname = CHSjetsname

        self._skipJEC = skipJEC
        self._skipJER = skipJER
        self._skipJES = skipJES
        self._skipJUNC = skipJUNC

        self._CHS_ptthreshold = CHS_ptthreshold
        self._CHS_etathreshold = CHS_etathreshold
        self._CHS_minJetID = CHS_minJetID
        self._CHS_minPUID = CHS_minPUID 
        self._CHS_PUID_ptthreshold = CHS_PUID_ptthreshold

        if skipJEC:
            if not skipJER:
                warnings.warn("skipJEC is True, but skipJER is False. Setting skipJER to True.")
                skipJER = True

            if not skipJES:
                warnings.warn("skipJEC is True, but skipJES is False. Setting skipJES to True.")
                skipJES = True

            if not skipJUNC:
                warnings.warn("skipJEC is True, but skipJUNC is False. Setting skipJUNC to True.")
                skipJUNC = True

        self._setup_SimonJets(events)
        self._setup_Jets(events)
        
        if self._CHSjetsname:
            self._setup_CHS(events)

    def check_btags(self, btagger : str, wps : dict[str, float]) -> None:
        if hasattr(self, '_matchedCHS'):
            self._matchedCHS.check_btags(btagger, wps)
            
            #check btag scores :D
            for wp_name, wp_value in wps.items():
                matchedBtag = self.matchedCHSjets['pass%sB' % (wp_name)]
                
                #for jets without matched partners,
                #btag is False
                matchedBtag = ak.pad_none(
                    matchedBtag,
                    1,
                    axis=-1,
                )
                matchedBtag = ak.fill_none(
                    matchedBtag,
                    False
                )
                self.simonjets['pass%sB' % (wp_name)] = ak.max(
                    self.matchedCHSjets['pass%sB' % (wp_name)],
                    axis=-1
                )
                
        
        #add gluon flavour to light hadron-flavour jets
        if hasattr(self.jets, 'partonFlavour'):
            self.jets['hadronFlavour'] = ak.where(
                (self.jets.hadronFlavour == 0) & (self.jets.partonFlavour == 21),
                21,
                self.jets.hadronFlavour
            )
        
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
            return self._matchedCHS.jets     
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
        
    @property
    def skipJEC(self) -> bool:
        return self._skipJEC
    
    @property
    def skipJER(self) -> bool:
        return self._skipJER    
    
    @property
    def skipJES(self) -> bool:
        return self._skipJES
    
    @property
    def skipJUNC(self) -> bool:
        return self._skipJUNC

    def _setup_SimonJets(self, events : ak.Array):
        self._parts : ak.Array = ak.materialize(
            events[self._simonjetsname]
        )

        self._simonjets : ak.Array = ak.materialize(
            events[self._simonjetsname+"BK"]
        )

        self._parts : ak.Array = unflatVector(
            self._parts,
            self._simonjets.nPart
        )

        #we sometimes pad with zero-pt parts, 
        #so nParts is wrong and needs to be recalculated
        self._parts : ak.Array= self._parts[self._parts.pt > 0]
        self._simonjets['nPart'] = ak.num(self._parts.pt)

    def _setup_Jets(self, events : ak.Array):
        self._jets : ak.Array = ak.materialize(
            events[self._jetsname]
        )
        self._jets = self._jets[self._simonjets.iJet]
        
    def _setup_CHS(self, events : ak.Array):
        self._matchedCHS = SimpleJets(
            events,
            self._CHSjetsname,
            self._CHS_ptthreshold,
            self._CHS_etathreshold,
            self._CHS_minJetID,
            self._CHS_minPUID,
            self._CHS_PUID_ptthreshold,
            index_by = ak.materialize(
                events[self._simonjetsname+'CHS'].idx
            ),
            reshape_by = self._simonjets.nCHS
        )

        self._simonjets['nCHS'] = ak.num(
            self._matchedCHS.jets.pt,
            axis=-1
        )
        '''
        self._matchedCHS : ak.Array = ak.materialize(
            events[self._CHSjetsname]
        )
        nCHS = ak.flatten(
            self._simonjets.nCHS, 
            axis=None # pyright: ignore[reportArgumentType]
        )
        iCHS = ak.materialize(
            events[self._simonjetsname+'CHS'].idx
        )

        bad = iCHS == 99999999
        iCHS = ak.where(bad, 0, iCHS)

        self._matchedCHS = ak.unflatten(
            self._matchedCHS[iCHS],
            nCHS,
            axis=1
        )
        bad = ak.unflatten(
            bad, 
            nCHS,
            axis=1
        )
        
        self._matchedCHS['pt'] = ak.where(
            bad, 
            0, 
            self._matchedCHS.pt
        )
        self._matchedCHS = self._matchedCHS[
            self._matchedCHS.pt > 0
        ]
        self._matchedCHS['puID'] = ak.where(
            self._matchedCHS.pt < self._CHS_PUID_ptthreshold,
            self._matchedCHS.puID,
            self._CHS_minPUID
        )
        self._matchedCHS = self._matchedCHS[
            (np.abs(self._matchedCHS.eta) < self._CHS_etathreshold)
            & (self._matchedCHS.pt > self._CHS_ptthreshold)
        ]
        #override nCHS 
        self._simonjets['nCHS'] = ak.num(self._matchedCHS.pt, axis=-1)
        '''