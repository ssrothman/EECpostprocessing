import awkward as ak
import pandas as pd
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
from binning.TrainingData import TrainingData
from binning.binMultiplicity import MultiplicityBinner
from binning.binBeff import BeffBinner
from binning.binKin import KinBinner
from binning.binBtag import BtagBinner

class EECProcessor(processor.ProcessorABC):
    def __init__(self, config, statsplit=False, what='EEC', 
                 syst='nom', syst_updn=None,
                 era='MC'):
        self.config = config
        self.statsplit = statsplit
        self.what = what
        self.syst = syst
        self.syst_updn = syst_updn
        self.era = era

        if what == 'EEC':
            self.binner = EECbinner(config.binning, config.tagging, config.controlJetSelection)
        elif what == 'Event':
            self.binner = EventBinner(config.binning, config.tagging)
        elif what == 'Multiplicity':
            self.binner = MultiplicityBinner(config.binning)
        elif what == 'TrainingData':
            self.binner = TrainingData()
        elif what == 'Beff':
            self.binner = BeffBinner(config.binning, config.tagging)
        elif what == 'Kin':
            self.binner = KinBinner(config.binning, config.tagging)
        elif what == "Btag":
            self.binner = BtagBinner(config.binning, config.tagging)
        else:
            raise ValueError("invalid 'what' %s" % what)

    def postprocess(self, accumulator):
        pass
    
    def process_from_fname(self, fname):
        from coffea.nanoevents import NanoEventsFactory
        events = NanoEventsFactory.from_root(fname).events()
        return self.process(events)

    def syst_weight(self, evtWeight):
        if 'wt' in self.syst:
            print("Varying event weight %s %s"%(self.syst, self.syst_updn))
            if self.syst_updn == 'UP':
                return evtWeight.weight(self.syst + 'Up')
            elif self.syst_updn == 'DN':
                return evtWeight.weight(self.syst + 'Down')
            else:
                raise ValueError("Invalid syst_updn %s" % self.syst_updn)
        else:
            return evtWeight.weight()

    def process(self, events):
        #setup inputs
        isMC = hasattr(events, 'genWeight')
        self.binner.isMC = isMC

        readers = AllReaders(events, self.config)
        readers.runJEC(self.era, self.syst, self.syst_updn)

        evtSel = masks.getEventSelection(
                readers.rMu, readers.rRecoJet,
                readers.HLT, self.config,
                isMC)
        jetSel = masks.getJetSelection(
                readers.rRecoJet, readers.rMu, 
                evtSel, self.config.jetSelection,
                isMC)

        jetMask = jetSel.all(*jetSel.names)
        evtMask = evtSel.all(*evtSel.names)

        evtWeight = weights.getEventWeight(events, 
                                           readers.rMu.rawmuons, 
                                           readers.rRecoJet,
                                           self.config, isMC)
        weight = self.syst_weight(evtWeight)

        #return outputs
        result = {}
        if self.statsplit:
            result["split"+"1"] = self.binner.binAll(
                    readers, jetMask & (events.event%2==0), 
                    evtMask & (events.event%2==0), weight)
            result["split"+"2"] = self.binner.binAll(
                    readers, jetMask & (events.event%2==1), 
                    evtMask & (events.event%2==1), weight)
        else:
            ans = self.binner.binAll(
                    readers, jetMask, evtMask, weight)
            if type(ans) is pd.DataFrame:
                fname = events.behavior["__events_factory__"]
                fname = fname._partition_key.replace("/", "_")
                fname = fname + '.parquet'
                fname = os.path.join("trainingdata/", fname)
                ans.to_parquet(fname)
            else:
                result = ans

        return result
