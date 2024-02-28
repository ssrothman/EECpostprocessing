import awkward as ak
import numpy as np
from coffea import processor

from reading.allreader import AllReaders

import selections.masks as masks
import selections.weights as weights

from Locker import Locker
import pickle

import os

from binning.binEEC import EECbinner
from binning.binEvt import EventBinner

class EECProcessor(processor.ProcessorABC):
    def __init__(self, config, statsplit=False, what='EEC'):
        self.config = config
        self.statsplit = statsplit
        self.what = what

        if what == 'EEC':
            self.binner = EECbinner(config.binning, config.btag, config.ctag)
        elif what == 'Event':
            self.binner = EventBinner(config.binning, config.btag, config.ctag)
        else:
            raise ValueError("invalid 'what' %s" % what)


    def postprocess(self, accumulator):
        pass
    
    def process_from_fname(self, fname):
        from coffea.nanoevents import NanoEventsFactory
        events = NanoEventsFactory.from_root(fname).events()
        return self.process(events)

    def process(self, events):
        #setup inputs
        readers = []
        names = []
        if self.what == 'EEC':
            for i in range(len(self.config.EECnames)):
                readers.append(AllReaders(events, self.config, i))
                names.append(self.config.EECnames[i])
        else:
            readers = [AllReaders(events, self.config, 0)]
            names = ['Events']

        evtSel = masks.getEventSelection(
                readers[0].rMu, readers[0].HLT, self.config)
        jetSel = masks.getJetSelection(
                readers[0].rRecoJet, readers[0].rMu, 
                evtSel, self.config.jetSelection)

        jetMask = jetSel.all(*jetSel.names)
        evtMask = evtSel.all(*evtSel.names)

        evtWeight = weights.getEventWeight(events, readers[0].rMu.muons, 
                                           self.config)
        weight = evtWeight.weight()

        #return outputs
        result = {}
        for reader,name in zip(readers, names):
            if self.statsplit:
                result[name+"1"] = self.binner.binAll(
                        reader, jetMask & (events.event%2==0), 
                        evtMask & (events.event%2==0), weight)
                result[name+"2"] = self.binner.binAll(
                        reader, jetMask & (events.event%2==1), 
                        evtMask & (events.event%2==1), weight)
            else:
                result[name] = self.binner.binAll(
                        reader, jetMask, evtMask, weight)

        return result
