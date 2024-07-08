import awkward as ak
import pandas as pd
import numpy as np
from coffea import processor

from reading.allreader import AllReaders

from selections.masks import getEventSelection
from selections.jetMask import getJetSelection
from selections.weights import getEventWeight

import pickle
import os
from time import time

from binning.dummy import DummyBinner
from binning.EECproj import EECprojBinner
from binning.Kinematics import KinematicsBinner
from binning.Match import MatchBinner
from binning.Beff import BeffBinner
from binning.Btag import BtagBinner

BINNERS = {
    'DUMMY' : DummyBinner,
    'EECPROJ' : EECprojBinner,
    'KINEMATICS' : KinematicsBinner,
    'MATCH' : MatchBinner,
    'BEFF' : BeffBinner,
    'BTAG' : BtagBinner
}

class EECProcessor(processor.ProcessorABC):
    def __init__(self, config, statsplit=False, what='EEC', 
                 sepPt=False,
                 scanSyst = False,
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
                 treatAsData=False,
                 manualcov=False,
                 poissonbootstrap=0):
        self.config = config
        self.statsplit = statsplit
        self.what = what
        self.era = era
        self.flags = flags
        self.scanSyst = scanSyst

        self.treatAsData = treatAsData
        self.manualcov = manualcov
        self.poissonbootstrap = poissonbootstrap

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

        self.binner = BINNERS[what](config,
                                    manualcov=manualcov,
                                    poissonbootstrap=poissonbootstrap,
                                    statsplit=statsplit,
                                    sepPt=sepPt)

    def postprocess(self, accumulator):
        pass
    
    def process_from_fname(self, fname):
        from coffea.nanoevents import NanoEventsFactory
        events = NanoEventsFactory.from_root(fname).events()
        return self.process(events)

    def process(self, events):
        #setup inputs
        t0 = time()
        isMC = hasattr(events, 'genWeight')
        self.binner.isMC = False if self.treatAsData else isMC

        readers = AllReaders(events, self.config, 
                             self.noRoccoR,
                             self.noJER, self.noJEC)
        t1 = time()
        readers.runJEC(self.era, '', '')
        readers.checkBtags(self.config)
        t2 = time()

        evtSel = getEventSelection(
                readers, self.config,
                isMC, self.flags)
        t3 = time()
        jetSel = getJetSelection(
                readers.rRecoJet, readers.rMu, 
                evtSel, self.config.jetSelection,
                isMC)
        t4 = time()

        jetMask = jetSel.all(*jetSel.names)
        evtMask = evtSel.all(*evtSel.names)
        t5 = time()
        #print(evtSel.names)

        evtWeight = getEventWeight(events, 
                                   readers,
                                  self.config, isMC,
                                  self.noPUweight,
                                  self.noPrefireSF,
                                  self.noIDsfs,
                                  self.noIsosfs,
                                  self.noTriggersfs,
                                  self.noBtagSF,
                                  self.Zreweight)
        t6 = time()
        for wt in evtWeight.weightStatistics.keys():
            print("\t", wt, evtWeight.weightStatistics[wt])
        t7 = time()

        nomweight = evtWeight.weight()
        print("CUTFLOW")
        for name in evtSel.names:
            print("\t", name, ak.sum(evtSel.all(name) * nomweight, axis=None))

        #return outputs
        result = {}

        print("doing nominal")
        readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['corrpt']
        nominal = self.binner.binAll(readers, 
                                     jetMask, evtMask,
                                     nomweight)

        nominal['sumwt'] = ak.sum(nomweight, axis=None)
        nominal['sumwt_pass'] = ak.sum(nomweight[evtMask], axis=None)
        nominal['numjet'] = ak.sum(jetMask * nomweight, axis=None)

        result['nominal'] = nominal

        if self.scanSyst:
            for variation in evtWeight.variations:
                print("doing", variation)
                theweight = evtWeight.weight(variation)

                result[variation] = self.binner.binAll(
                        readers, jetMask, evtMask,
                        theweight)
                result[variation]['sumwt'] = ak.sum(theweight, axis=None)
                result[variation]['sumwt_pass'] = ak.sum(theweight[evtMask],
                                                         axis=None)
                result[variation]['numjet'] = ak.sum(jetMask * theweight, 
                                                     axis=None)

            for jvar in ['JER_UP', 'JER_DN', 'JES_UP', 'JES_DN']:
                print('doing', jvar)
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets[jvar]

                result[jvar] = self.binner.binAll(
                        readers, jetMask, evtMask,
                        nomweight)
                result[jvar]['sumwt'] = result['nominal']['sumwt']
                result[jvar]['sumwt_pass'] = result['nominal']['sumwt_pass']
                result[jvar]['numjet'] = result['nominal']['numjet']

        t8 = time()
        t9 = time()

        print("SUMWT", result['nominal']['sumwt'])
        print("SUMWT_PASS", result['nominal']['sumwt_pass'])
        print("NUMJET", result['nominal']['numjet'])

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

        result['config'] = self.config

        return result
