import warnings
import awkward as ak
import numpy as np

from util.util import unflatVector

class Jets:
    def __init__(self, 
                 events : ak.Array, 
                 jetsname : str,
                 simonjetsname : str,
                 CHSjetsname : str,
                 skipJEC : bool, #skipJEC implies skipJER AND skipJES and skipJUNC
                 skipJER : bool,
                 skipJES : bool,
                 skipJUNC : bool):

        self._jetsname = jetsname
        self._simonjetsname = simonjetsname
        self._CHSjetsname = CHSjetsname

        self._skipJEC = skipJEC
        self._skipJER = skipJER
        self._skipJES = skipJES
        self._skipJUNC = skipJUNC

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
        #check btag scores :D
        for wp_name, wp_value in wps.items():
            btag_score = self.matchedCHSjets[btagger]
            self.matchedCHSjets['pass%sB' % (wp_name)] = (
                btag_score > wp_value
            )

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
        if hasattr(self.matchedCHSjets, 'partonFlavour'):
            self.matchedCHSjets['hadronFlavour'] = ak.where(
                (self.matchedCHSjets.hadronFlavour == 0) & (self.matchedCHSjets.partonFlavour == 21),
                21, 
                self.matchedCHSjets.hadronFlavour
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
