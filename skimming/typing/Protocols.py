from typing import Any, Protocol, runtime_checkable

from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection

@runtime_checkable
class EventSelectorProtocol(Protocol):
    def select_events(self, allobjects : AllObjects, flags : dict) -> PackedSelection:
        ...

@runtime_checkable
class JetSelectorProtocol(Protocol):
    def select_jets(self, allobjects : AllObjects, evtsel : PackedSelection, flags : dict) -> PackedJetSelection:
        ...