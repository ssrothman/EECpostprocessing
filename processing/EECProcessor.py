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

from time import time

from binning.binMatch2 import MatchBinner
from binning.binEEC import EECbinner
from binning.TrainingData import TrainingData
from binning.binMultiplicity import MultiplicityBinner
from binning.binBeff import BeffBinner
from binning.binKin import KinBinner
from binning.binBtag import BtagBinner
from binning.binHT import HTBinner
from binning.dummy import DummyBinner
from binning.EECproj import EECprojBinner

class EECProcessor(processor.ProcessorABC):
    def __init__(self, config, statsplit=False, what='EEC', 
                 syst='nom', syst_updn=None,
                 era='MC', flags=None,
                 noRoccoR=False,
                 noJER=False, noJEC=False,
                 noPUweight=False,
                 noPrefireSF=False,
                 noIDsfs=False,
                 noIsosfs=False,
                 noTriggersfs=False,
                 noBtagSF=False,
                 Zreweight=False,
                 treatAsData=False):
        self.config = config
        self.statsplit = statsplit
        self.what = what
        self.syst = syst
        self.syst_updn = syst_updn
        self.era = era
        self.flags = flags

        self.treatAsData = treatAsData

        self.noRoccoR = noRoccoR
        self.noJER = noJER
        self.noJEC = noJEC
        self.noPUweight = noPUweight
        self.noPrefireSF = noPrefireSF
        self.noIDsfs = noIDsfs
        self.noIsosfs = noIsosfs
        self.noTriggersfs = noTriggersfs
        self.noBtagSF = noBtagSF

        self.Zreweight = Zreweight
    
        what= what.strip().upper()

        if what == 'DUMMY':
            self.binner = DummyBinner()
        elif what == 'EECPROJ':
            self.binner = EECprojBinner(config.binning, config.tagging)
        elif what == 'MATCH':
            self.binner = MatchBinner(config.binning, config.tagging)
        elif what == 'EEC':
            self.binner = EECbinner(config.binning, config.tagging, config.controlJetSelection)
        elif what == 'EVENT':
            self.binner = EventBinner(config.binning, config.tagging)
        elif what == 'MULTIPLICITY':
            self.binner = MultiplicityBinner(config.binning)
        elif what == 'BEFF':
            self.binner = BeffBinner(config.binning, config.tagging)
        elif what == 'KIN':
            self.binner = KinBinner(config.binning, config.tagging)
        elif what == "BTAG":
            self.binner = BtagBinner(config.binning, config.tagging)
        elif what == 'HT':
            self.binner = HTBinner(config.binning)
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
        t0 = time()
        isMC = hasattr(events, 'genWeight')
        self.binner.isMC = False if self.treatAsData else isMC

        readers = AllReaders(events, self.config, 
                             self.noRoccoR,
                             self.noJER, self.noJEC)
        t1 = time()
        readers.runJEC(self.era, self.syst, self.syst_updn)
        readers.checkBtags(self.config)
        t2 = time()

        evtSel = masks.getEventSelection(
                readers.rMu, readers.rRecoJet,
                readers.HLT, self.config,
                isMC, self.flags)
        t3 = time()
        jetSel = masks.getJetSelection(
                readers.rRecoJet, readers.rMu, 
                evtSel, self.config.jetSelection,
                isMC)
        t4 = time()

        jetMask = jetSel.all(*jetSel.names)
        evtMask = evtSel.all(*evtSel.names)
        t5 = time()
        #print(evtSel.names)

        evtWeight = weights.getEventWeight(events, 
                                           readers.rMu.rawmuons, 
                                           readers.rMu.Zs,
                                           readers.rRecoJet,
                                           self.config, isMC,
                                           self.noPUweight,
                                           self.noPrefireSF,
                                           self.noIDsfs,
                                           self.noIsosfs,
                                           self.noTriggersfs,
                                           self.noBtagSF,
                                           self.Zreweight)
        t6 = time()
        #for wt in evtWeight.weightStatistics.keys():
        #    print("\t", wt, evtWeight.weightStatistics[wt])
        weight = self.syst_weight(evtWeight)
        t7 = time()
        #print("weight from", ak.max(weight), "to", ak.min(weight))
        #print("weight sources:")

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

        t8 = time()
        result['sumwt'] = ak.sum(weight, axis=None)
        result['sumwt_pass'] = ak.sum(weight[evtMask], axis=None)
        result['numjet'] = ak.sum(jetMask * weight, axis=None)
        t9 = time()

        print("runtime summary:")
        print("\tinitial setup: %0.2g" % (t1-t0))
        print("\tJEC: %0.2g" % (t2-t1))
        print("\tevent selection: %0.2g" % (t3-t2))
        print("\tjet selection: %0.2g" % (t4-t3))
        print("\tmask building: %0.2g" % (t5-t4))
        print("\tweight computation: %0.2g" % (t6-t5))
        print("\tweighting: %0.2g" % (t7-t6))
        print("\tbinning: %0.2g" % (t8-t7))
        print("\tsummary weights: %0.2g" % (t9-t8))

        return result
