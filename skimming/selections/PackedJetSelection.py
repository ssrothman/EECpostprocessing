import awkward as ak
from coffea.analysis_tools import PackedSelection
import numpy as np

class PackedJetSelection:
    def __init__(self, evtSel=None):
        self.selection = PackedSelection()
        self.nums = None
        self.evtSel = evtSel

    def add(self, name, mask):
        if self.nums is None:
            self.nums = ak.num(mask)
            if self.evtSel is not None:
                for evtName in self.evtSel.names:
                    evtMask = self.evtSel.all(evtName)
                    newMask, _ = ak.broadcast_arrays(evtMask, mask)
                    self.selection.add(evtName, ak.flatten(newMask))
        else:
            assert ak.all(self.nums == ak.num(mask))
        self.selection.add(name, ak.flatten(mask))

    def all(self, *names):
        mask = self.selection.all(*names)
        return ak.unflatten(mask, self.nums)

    def any(self, *names):
        mask = self.selection.any(*names)
        return ak.unflatten(mask, self.nums)

    def require(self, **names):
        mask = self.selection.require(**names)
        return ak.unflatten(mask, self.nums)

    @property
    def names(self):
        return self.selection.names
