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
from binning.Kinematics import KinematicsBinner
from binning.Match import MatchBinner
from binning.Beff import BeffBinner
from binning.Btag import BtagBinner
from binning.EECproj import EECprojBinner
from binning.EECres3 import EECres3Binner
from binning.EECres4dipole import EECres4dipoleBinner
from binning.EECres4tee import EECres4teeBinner
#from binning.EECgeneric import EECgenericBinner

allowedvariations = {
    'scanTheory' : ['wt_scaleUp', 'wt_scaleDown', 
                    'wt_PDFaSUp', 'wt_PDFaSDown'],
    'scanMuon' : ['wt_idsfUp', 'wt_idsfDown',
                  'wt_isosfUp', 'wt_isosfDown'],
    'scanTrigger' : ['wt_triggersfUp', 'wt_triggersfDown',
                     'wt_prefireUp', 'wt_prefireDown'],
    'scanPS' : ['wt_ISRUp', 'wt_ISRDown',
                'wt_FSRUp', 'wt_FSRDown'],
    'scanBtag' : ['wt_btagSF_effUp', 'wt_btagSF_effDown',
                  'wt_btagSF_sfUp', 'wt_btagSF_sfDown'],
    'scanPileup' : ['wt_PUUp', 'wt_PUDown'],
    'scanJetMET' : ['JER_UP', 'JER_DN', 'JES_UP', 'JES_DN'],
    'noSyst' : []
}

BINNERS = {
    'DUMMY' : DummyBinner,
    'KINEMATICS' : KinematicsBinner,
    'MATCH' : MatchBinner,
    'BEFF' : BeffBinner,
    'BTAG' : BtagBinner,
    'EECPROJ' : EECprojBinner,
    "EECRES3" : EECres3Binner,
    "EECRES4DIPOLE" : EECres4dipoleBinner,
    "EECRES4TEE" : EECres4teeBinner,
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
                 poissonbootstrap=0,
                 noBkgVeto=False,
                 skipNominal=False):
        self.config = config
        self.statsplit = statsplit
        self.what = what
        self.era = era
        self.flags = flags
        self.scanSyst = scanSyst
        self.skipNominal = skipNominal

        self.noBkgVeto = noBkgVeto

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

    def actually_process(self, events, readers,
                         resultdict,
                         object_systematic=None):
        if object_systematic is not None:
            if object_systematic == 'JER_UP':
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['JER_UP']
                readers.METpt = readers.MET.ptJERUp
            elif object_systematic == 'JER_DN':
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['JER_DN']
                readers.METpt = readers.MET.ptJERDown
            elif object_systematic == 'JES_UP':
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['JES_UP']
                readers.METpt = readers.MET.ptJESUp
            elif object_systematic == 'JES_DN':
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['JES_DN']
                readers.METpt = readers.MET.ptJESDown 
            elif object_systematic == "UNCLUSTERED_UP":
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['corrpt']
                readers.METpt = readers.MET.ptUnclusteredUp
            elif object_systematic == "UNCLUSTERED_DN":
                readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['corrpt']
                readers.METpt = readers.MET.ptUnclusteredDown
            else:
                raise ValueError("Unknown systematic: %s" % object_systematic)
        else:
            readers.rRecoJet.jets['pt'] = readers.rRecoJet.jets['corrpt']
            readers.METpt = readers.MET.pt


        if (object_systematic is not None) or (not self.skipNominal):
            evtSel = getEventSelection(
                    readers, self.config,
                    self.isMC, self.flags,
                    self.noBkgVeto)

            jetSel = getJetSelection(
                    readers.rRecoJet, readers.rMu, 
                    evtSel, self.config.jetSelection,
                    self.config.jetvetomap,
                    self.isMC)

            evtWeight = getEventWeight(events, 
                                       readers,
                                       self.config, 
                                       self.isMC,
                                       self.noPUweight,
                                       self.noPrefireSF,
                                       self.noIDsfs,
                                       self.noIsosfs,
                                       self.noTriggersfs,
                                       self.noBtagSF,
                                       self.Zreweight)

            jetMask = jetSel.all(*jetSel.names)
            evtMask = evtSel.all(*evtSel.names)
            nomweight = evtWeight.weight()
            if np.any(nomweight > 1e2):
                print("WARNING: large weights found")
                print("setting to 1")
                nomweight[nomweight > 1e2] = 1
            if np.any(nomweight < 1e-2):
                print("WARNING: small weights found")
                print("setting to 1")
                nomweight[nomweight < 1e-2] = 1

            if object_systematic is None:
                print("CUTFLOW")
                cuts_so_far = []
                for name in evtSel.names:
                    cuts_so_far.append(name)
                    print("\t%s:%g"%(name, ak.sum(evtSel.all(*cuts_so_far) * nomweight, axis=None)))

                print("JET CUTFLOW")
                cuts_so_far = []
                for name in jetSel.names:
                    cuts_so_far.append(name)
                    print("\t%s:%g"%(name, ak.sum(jetSel.all(*cuts_so_far) * nomweight, axis=None)))

                print("WEIGHTS")
                for wt in evtWeight.weightStatistics.keys():
                    print("\t", wt, evtWeight.weightStatistics[wt])
                print("minwt = ", np.min(nomweight))
                print("maxwt = ", np.max(nomweight))

            nominal = self.binner.binAll(readers, 
                                         jetMask, evtMask,
                                         nomweight)
            nominal['sumwt'] = ak.sum(nomweight, axis=None)
            nominal['sumwt_pass'] = ak.sum(nomweight[evtMask], axis=None)
            nominal['numjet'] = ak.sum(jetMask * nomweight, axis=None)

            if object_systematic is None:
                nominalname = 'nominal'
            else:
                nominalname = object_systematic

            resultdict[nominalname] = nominal

        if self.scanSyst != 'noSyst'\
                and object_systematic is None \
                and self.isMC:

            for variation in evtWeight.variations:
                if variation not in allowedvariations[self.scanSyst]:
                    continue

                print("doing", variation)
                theweight = evtWeight.weight(variation)
                print("minwt = ", np.min(theweight))
                print("maxwt = ", np.max(theweight))

                if np.any(theweight > 1e2):
                    print("WARNING: large weights found")
                    print("setting to 1")
                    theweight[theweight > 1e2] = 1
                if np.any(theweight < 1e-2):
                    print("WARNING: small weights found")
                    print("setting to 1")
                    theweight[theweight < 1e-2] = 1

                resultdict[variation] = self.binner.binAll(
                        readers, jetMask, evtMask,
                        theweight)
                resultdict[variation]['sumwt'] = ak.sum(
                        theweight, 
                        axis=None)
                resultdict[variation]['sumwt_pass'] = ak.sum(
                        theweight[evtMask],
                        axis=None)
                resultdict[variation]['numjet'] = ak.sum(
                        jetMask * theweight, 
                        axis=None)

    def process(self, events):
        #setup inputs
        t0 = time()
        self.isMC = hasattr(events, 'genWeight')
        self.binner.isMC = False if self.treatAsData else self.isMC

        readers = AllReaders(events, self.config, 
                             self.noRoccoR,
                             self.noJER, self.noJEC)

        readers.runJEC(self.era, '', '')
        readers.checkBtags(self.config)

        #for wt in evtWeight.weightStatistics.keys():
        #    print("\t", wt, evtWeight.weightStatistics[wt])

        #print("CUTFLOW")
        #for name in evtSel.names:
        #    print("\t", name, ak.sum(evtSel.all(name) * nomweight, axis=None))

        #return outputs
        result = {}

        if self.scanSyst in ['scanJetMET', 'scanAll'] and self.isMC:
            objsys_l = [None, 
                        'JER_UP', 'JER_DN', 'JES_UP', 'JES_DN',
                        'UNCLUSTERED_UP', 'UNCLUSTERED_DN']
        else:
            objsys_l = [None]

        for objsys in objsys_l:
            self.actually_process(events, readers, 
                                  result, 
                                  object_systematic=objsys)

        print("SUMWT", result['nominal']['sumwt'])
        print("SUMWT_PASS", result['nominal']['sumwt_pass'])
        print("NUMJET", result['nominal']['numjet'])

        #print("runtime summary:")
        #print("\tinitial setup: %0.2g" % (t1-t0))
        #print("\tJEC: %0.2g" % (t2-t1))
        #print("\tevent selection: %0.2g" % (t3-t2))
        #print("\tjet selection: %0.2g" % (t4-t3))
        #print("\tmask building: %0.2g" % (t5-t4))
        #print("\tweight computation: %0.2g" % (t6-t5))
        #print("\tweighting: %0.2g" % (t7-t6))
        #print("\tbinning: %0.2g" % (t8-t7))
        #print("\tsummary weights: %0.2g" % (t9-t8))

        result['config'] = self.config

        return result
