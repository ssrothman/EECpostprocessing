import awkward as ak
from coffea.analysis_tools import PackedSelection
from .PackedJetSelection import PackedJetSelection
from skimming.objects.AllObjects import AllObjects


class NullJetSelector:

    def __init__(self, cfg: dict):
        pass

    def select_jets(self,
                    allobjects: AllObjects,
                    evtsel: PackedSelection,
                    flags: dict) -> PackedJetSelection:
        selection = PackedJetSelection(evtsel)
        jets = allobjects.RecoJets.jets
        selection.add("all", ak.ones_like(jets.pt, dtype=bool))
        return selection
