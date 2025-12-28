from skimming.objects.AllObjects import AllObjects
from skimming.typing.Protocols import EventSelectorProtocol, JetSelectorProtocol
from .StandardJetSelector import StandardJetSelector
from .ZMuMuEventSelector import ZMuMuEventSelector
from .PackedJetSelection import PackedJetSelection
from coffea.analysis_tools import PackedSelection
import awkward as ak

eventselectors = {
    'ZMuMuEventSelector' : ZMuMuEventSelector,
}
jetselectors = {
    'StandardJetSelector' : StandardJetSelector,
}

def runEventSelection(cfg : dict,
                      allobjects : AllObjects,
                      flags : dict) -> PackedSelection:
    
    selclass = eventselectors[cfg['class']]
    evtsel : EventSelectorProtocol = selclass(cfg['params'])
    selection = evtsel.select_events(allobjects, flags)
    return selection

def runJetSelection(cfg : dict,
                    allobjects : AllObjects,
                    evtsel : PackedSelection,
                    flags : dict) -> PackedJetSelection:
    
    selclass = jetselectors[cfg['class']]
    jetsel : JetSelectorProtocol = selclass(cfg['params'])
    jetselection  = jetsel.select_jets(allobjects, evtsel, flags)
    return jetselection