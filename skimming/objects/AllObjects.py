import awkward as ak
from vector import obj
from .jets import Jets
from .muons import Muons
from .met import MET
from .generics import GenericObjectContainer
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JECStack, CorrectedJetsFactory
import cachetools
import numpy as np
from typing import Any, Mapping

objclasses = {
    "Jets" : Jets,
    "Muons" : Muons,
    "MET" : MET,
    "GenericObjectContainer" : GenericObjectContainer
}

class AllObjects:
    def __init__(self, 
                events : ak.Array, 
                JECera : str,
                objcfg : dict,
                btagcfg : dict,
                JECcfg : dict,
                objsyst : str):
       
        self._setup_objects(events, objcfg, objsyst)
        self._check_btags(btagcfg)
        self._run_JEC(JECera, JECcfg)

    def _setup_objects(self, events : ak.Array,
                       objcfg : dict, 
                       objsyst : str) -> None:
        
        self._objects = {}

        #special attributes
        self._objects['objsyst'] =  objsyst
        self._objects['isMC'] = hasattr(events, 'Generator') 

        for objname, objcfg in objcfg.items():
            if objname == 'JECTARGET': # not actually an object
                self._JECtarget = objcfg
                continue

            clsname = objcfg['class']
            cls = objclasses[clsname]

            if 'params-%s' in objcfg:
                paramskey = 'params-%s'%objsyst
            else:
                paramskey = 'params'

            nextobj = cls(
                events,
                **objcfg[paramskey]
            )

            if clsname == 'GenericObjectContainer':
                for subname in objcfg['params']['mandatory_names'].keys():
                    self._objects[subname] = getattr(nextobj, subname)
                for subname in objcfg['params']['optional_names'].keys():
                    objobj = getattr(nextobj, subname)
                    if objobj is not None:
                        self._objects[subname] = objobj
            else:
                self._objects[objname] = nextobj

    def __getattr__(self, name : str) -> Any:
        if name in self._objects:
            return self._objects[name]
        else:
            raise AttributeError("Object %s not loaded!"%name)
        
    @property
    def objlist(self) -> list[str]:
        return list(self._objects.keys())

    #autocomplete support :)    
    def __dir__(self) -> list[str]:
        return self.objlist
    
    def _check_btags(self, btagcfg : dict) -> None:
        for obj in self._objects.values():
            if hasattr(obj, 'check_btags'):
                obj.check_btags(
                    btagger = btagcfg['btagger'],
                    wps = btagcfg['wps']
                )
    
    def _run_JEC(self, era : str, JECcfg : dict) -> None:
        if not hasattr(self, '_JECtarget'):
            raise RuntimeError("No JECTARGET specified in object config, cannot run JEC!")
        
        if era == 'skip':
            return #skip!

        targetobj = self._objects[self._JECtarget]
       
        if targetobj.skipJEC:
            return #skip!
        
        # save cmssw pT
        # for checking :)
        targetobj.jets['pt_cmssw'] = targetobj.jets.pt
        targetobj.jets['mass_cmssw'] = targetobj.jets.mass

        #first, setup inputs
        targetobj.jets['pt_raw'] = targetobj.jets.pt * targetobj.jets.jecFactor
        targetobj.jets['mass_raw'] = targetobj.jets.mass * targetobj.jets.jecFactor
        targetobj.jets['event_rho'] = self.rho

        if self.isMC:
            targetobj.jets['pt_gen'] = targetobj.simonjets.jetMatchPt
            targetobj.jets['eta_gen'] = targetobj.simonjets.jetMatchEta
            targetobj.jets['phi_gen'] = targetobj.simonjets.jetMatchPhi

        #then, build the JEC stack
        stacknames = []
        files = JECcfg['files'][era]
        if not targetobj.skipJES:
            stacknames += JECcfg['JESstacks'][era]
        if self.isMC and not targetobj.skipJER:
            stacknames += JECcfg['JERstacks'][era]
        if self.isMC and not targetobj.skipJUNC:
            stacknames += JECcfg['JECuncertainties']
        
        ex = extractor()
        ex.add_weight_sets(
            ['* * %s'%f for f in files]
        )
        ex.finalize()
        ev = ex.make_evaluator()

        stack = JECStack({key : ev[key] for key in stacknames})

        #then setup input name map
        name_map = stack.blank_name_map
        name_map['JetPt'] = 'pt'                # pyright: ignore[reportArgumentType]
        name_map['JetMass'] = 'mass'            # pyright: ignore[reportArgumentType]
        name_map['JetEta'] = 'eta'              # pyright: ignore[reportArgumentType]
        name_map['Rho'] = 'event_rho'           # pyright: ignore[reportArgumentType]
        name_map['JetA'] = 'area'               # pyright: ignore[reportArgumentType]
        name_map['massRaw'] = 'mass_raw'        # pyright: ignore[reportArgumentType]
        name_map['ptRaw'] = 'pt_raw'            # pyright: ignore[reportArgumentType]
        if 'MC' in era:
            name_map['ptGenJet'] = 'pt_gen'     # pyright: ignore[reportArgumentType]

        #jec_cache = cachetools.Cache(np.inf)
        jet_factory = CorrectedJetsFactory(name_map, stack)
        corrected_jets : ak.Array = jet_factory.build(targetobj.jets)      # pyright: ignore[reportAssignmentType]
                                           #lazy_cache=jec_cache)

        if self.objsyst == 'JER_UP':
            corrjets = corrected_jets.JER['up']
            factor = 1.0
        elif self.objsyst == 'JER_DN':
            corrjets = corrected_jets.JER['down']
            factor = 1.0
        elif 'JES' in self.objsyst:
            corrjets = corrected_jets
            uncs = []
            for field in corrjets.fields:
                if field.startswith("jet_energy_uncertainty"):
                    uncs.append(np.square(corrjets[field][:,:,0] - 1))
            
            total_unc = np.sqrt(ak.sum(uncs, axis=0))
            if self.objsyst.endswith('_UP'):
                factor = 1.0 + total_unc
            else:
                factor = 1.0 - total_unc

        else:
            corrjets = corrected_jets
            factor = 1.0
        
        targetobj.jets['pt'] = corrjets.pt * factor
        targetobj.jets['mass'] = corrjets.mass * factor