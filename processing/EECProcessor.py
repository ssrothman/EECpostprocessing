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
import os.path
from time import time

from binning.dummy import DummyBinner
from binning.Kinematics import KinematicsBinner
from binning.Match import MatchBinner
from binning.Res import ResBinner
from binning.Beff import BeffBinner
from binning.Btag import BtagBinner
from binning.EECproj import EECprojBinner
from binning.EECres4dipole import EECres4dipoleBinner
from binning.EECres4tee import EECres4teeBinner
from binning.EECres4triangle import EECres4triangleBinner

#from binning.EECgeneric import EECgenericBinner

BINNERS = {
    'DUMMY' : DummyBinner,
    'KINEMATICS' : KinematicsBinner,
    'MATCH' : MatchBinner,
    "RES" : ResBinner,
    'BEFF' : BeffBinner,
    'BTAG' : BtagBinner,
    'EECPROJ' : EECprojBinner,
    "EECRES4DIPOLE" : EECres4dipoleBinner,
    "EECRES4TEE" : EECres4teeBinner,
    'EECRES4TRIANGLE' : EECres4triangleBinner,
}

class EECProcessor(processor.ProcessorABC):
    def __init__(self, 
                 config,
                 basepath, 
                 binningtype,
                 era,
                 flags,
                 noRoccoR,
                 noJER,
                 noJEC,
                 noJUNC,
                 noPUweight,
                 noPrefireSF,
                 noIDsfs,
                 noIsosfs,
                 noTriggersfs,
                 noBtagSF,
                 Zreweight,
                 noBkgVeto,
                 syst,
                 verbose):
        
        self.syst = syst
        self.basepath = basepath

        self.verbose = verbose

        self.config = config
        self.binningtype = binningtype
        self.era = era
        self.flags = flags

        self.noBkgVeto = noBkgVeto

        self.isMC = config.isMC

        self.noRoccoR = noRoccoR
        self.noJER = noJER
        self.noJEC = noJEC
        self.noJUNC = noJUNC
        self.noPUweight = noPUweight
        self.noPrefireSF = noPrefireSF
        self.noIDsfs = noIDsfs
        self.noIsosfs = noIsosfs
        self.noTriggersfs = noTriggersfs
        self.noBtagSF = noBtagSF

        self.Zreweight = Zreweight
    
        binningtype= binningtype.strip().upper()

        if binningtype == 'COUNT':
            self.binner = 'COUNT'
        elif binningtype == 'CUTFLOW':
            self.binner = 'CUTFLOW'
        else:
            self.binner = BINNERS[binningtype](config)

    def postprocess(self, accumulator):
        pass
    
    def process_from_fname(self, fname):
        from coffea.nanoevents import NanoEventsFactory
        events = NanoEventsFactory.from_root(fname).events()
        return self.process(events)

    def actually_process(self, events, readers,
                         resultdict):

        evtSel = getEventSelection(
                readers, self.config,
                self.isMC, self.flags,
                self.noBkgVeto, 
                self.verbose)

        jetSel = getJetSelection(
                readers.rRecoJet, readers.rMu, 
                evtSel, self.config,
                self.isMC,
                self.verbose)

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
        if np.any(nomweight > 1e3):
            print("WARNING: large weights found")
            print("setting to 1")
            nomweight[nomweight > 1e3] = 1
        if np.any(nomweight < 1e-3):
            print("WARNING: small weights found")
            print("setting to 1")
            nomweight[nomweight < 1e-3] = 1

        weightvariations = {'evtwt_nominal' : nomweight}
        if self.config.isMC and self.syst == 'nominal':
            for wt in evtWeight.variations:
                weightvariations["evt"+wt] = evtWeight.weight(wt)

        if self.verbose:
            print("CUTFLOW")
            cuts_so_far = []
            print("\tnone:%g"%(ak.sum(evtSel.all()*nomweight, axis=None)))
            for name in evtSel.names:
                cuts_so_far.append(name)
                print("\t%s:%g"%(name, ak.sum(evtSel.all(*cuts_so_far) * nomweight, axis=None)))

            print("JET CUTFLOW")
            cuts_so_far = []
            print("\tnone:%g"%(ak.sum(jetSel.all()*nomweight,axis=None)))
            for name in jetSel.names:
                cuts_so_far.append(name)
                print("\t%s:%g"%(name, ak.sum(jetSel.all(*cuts_so_far) * nomweight, axis=None)))

            print("WEIGHTS")
            for wt in evtWeight.weightStatistics.keys():
                print("\t", wt, evtWeight.weightStatistics[wt])
            print("minwt = ", np.min(nomweight))
            print("maxwt = ", np.max(nomweight))

            if len(evtWeight.variations) > 0:
                print("Available weight variations")
                for wt in evtWeight.variations:
                    print("\t", wt)

        if type(self.binner) is str and self.binner == 'CUTFLOW':
            flow_evt = {}

            flow_evt['none'] = ak.sum(nomweight, axis=None)
            cuts_so_far = []
            evtsel_cuts = list(evtSel.names)
            evtsel_cuts.remove('METpt')
            evtsel_cuts.append('METpt')
            evtsel_cuts.remove('nbtag')
            evtsel_cuts.append('nbtag')
            for name in evtsel_cuts:
                cuts_so_far.append(name)
                flow_evt[name] = ak.sum(evtSel.all(*cuts_so_far) * nomweight, axis=None)

            flow_jet = {}

            flow_jet['none'] = ak.sum(jetSel.all() * nomweight, axis=None)
            cuts_so_far = []
            jetsel_cuts = list(jetSel.names)
            jetsel_cuts.remove('METpt')
            jetsel_cuts.append('METpt')
            jetsel_cuts.remove('nbtag')
            jetsel_cuts.append('nbtag')
            for name in jetsel_cuts:
                cuts_so_far.append(name)
                flow_jet[name] = ak.sum(jetSel.all(*cuts_so_far) * nomweight, axis=None)

            resultdict[self.syst] = {
                'evt' : flow_evt,
                'jet' : flow_jet
            }
            return

        thepath = os.path.join(self.basepath, self.syst)
        nominal = self.binner.binAll(readers, 
                                     jetMask, evtMask,
                                        weightvariations,
                                     thepath)
        nominal['sumwt'] = ak.sum(nomweight, axis=None)
        nominal['sumwt_pass'] = ak.sum(nomweight[evtMask], axis=None)
        nominal['numjet'] = ak.sum(jetMask * nomweight, axis=None)

        resultdict[self.syst] = nominal

    def process(self, events):
        #setup inputs
        t0 = time()

        #if(len(events) == 0):
        #    print("SKIPPING FILE")
        #    return {}

        if type(self.binner) is str and self.binner == 'COUNT':
            if self.isMC:
                return {'num_evt': np.sum(events.genWeight)}
            else:
                return {'num_evt': len(events)}

        if type(self.binner) is not str:
            self.binner.isMC = self.isMC

        readers = AllReaders(events, self.config, 
                             self.noRoccoR,
                             self.noJER, self.noJEC, self.noJUNC,
                             self.syst)

        readers.runJEC(self.era, self.verbose)
        readers.checkBtags(self.config)

        result = {}

        self.actually_process(events, readers, 
                              result)

        if self.verbose and type(self.binner) is not str:
            for key in result.keys():
                print("SUMWT", result[key]['sumwt'])
                print("SUMWT_PASS", result[key]['sumwt_pass'])
                print("NUMJET", result[key]['numjet'])

        result['config'] = self.config

        return result
