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
from binning.Res import ResBinner
from binning.Beff import BeffBinner
from binning.Btag import BtagBinner
from binning.EECproj import EECprojBinner
from binning.EECres3 import EECres3Binner
from binning.EECres4dipole import EECres4dipoleBinner
from binning.EECres4tee import EECres4teeBinner
from binning.EECres4triangle import EECres4triangleBinner
from binning.EECres4minR import EECres4minRBinner

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
    'scanBtagEff' : ['wt_btagSF_effUp', 'wt_btagSF_effDown',
                    'wt_btagSF_tighteffUp', 'wt_btagSF_tighteffDown'],
    'scanBtagSF' : ['wt_btagSF_sfUp', 'wt_btagSF_sfDown',
                    'wt_btagSF_tightsfUp', 'wt_btagSF_tightsfDown'],
    'scanPileup' : ['wt_PUUp', 'wt_PUDown'],
    'scanCBxsec' : ['wt_c_xsecUp', 'wt_c_xsecDown',
                    'wt_b_xsecUp', 'wt_b_xsecDown'],
    'scanLxsec' : ['wt_uds_xsecUp', 'wt_uds_xsecDown'
                   'wt_g_xsecUp', 'wt_g_xsecDown'],
    'scanJetMET' : ['JER_UP', 'JER_DN', 'JES_UP', 'JES_DN'],
    'noSyst' : []
}

BINNERS = {
    'DUMMY' : DummyBinner,
    'KINEMATICS' : KinematicsBinner,
    'MATCH' : MatchBinner,
    "RES" : ResBinner,
    'BEFF' : BeffBinner,
    'BTAG' : BtagBinner,
    'EECPROJ' : EECprojBinner,
    "EECRES3" : EECres3Binner,
    "EECRES4DIPOLE" : EECres4dipoleBinner,
    "EECRES4TEE" : EECres4teeBinner,
    'EECRES4TRIANGLE' : EECres4triangleBinner,
    'EECRES4MINR' : EECres4minRBinner
}

class EECProcessor(processor.ProcessorABC):
    def __init__(self, config, statsplit=False, binningtype='EEC', 
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
                 isMC=False,
                 manualcov=False,
                 poissonbootstrap=0,
                 skipBtag = False,
                 noBkgVeto=False,
                 skipNominal=False,
                 verbose=False):
        self.verbose = verbose

        self.config = config
        self.statsplit = statsplit
        self.binningtype = binningtype
        self.era = era
        self.flags = flags
        self.scanSyst = scanSyst
        self.skipNominal = skipNominal

        self.noBkgVeto = noBkgVeto

        self.isMC = isMC
        self.manualcov = manualcov
        self.poissonbootstrap = poissonbootstrap
        self.skipBtag = skipBtag

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
    
        binningtype= binningtype.strip().upper()

        self.binner = BINNERS[binningtype](config,
                                    manualcov=manualcov,
                                    poissonbootstrap=poissonbootstrap,
                                    skipBtag=skipBtag,
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
            if hasattr(readers, '_MET'):
                readers.METpt = readers.MET.pt
            else:
                import warnings
                warnings.warn("No MET in events")

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
        if np.any(nomweight > 1e2):
            print("WARNING: large weights found")
            print("setting to 1")
            nomweight[nomweight > 1e2] = 1
        if np.any(nomweight < 1e-2):
            print("WARNING: small weights found")
            print("setting to 1")
            nomweight[nomweight < 1e-2] = 1

        if (object_systematic is not None) or (not self.skipNominal):
            if self.verbose and object_systematic is None:
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

            nominal = self.binner.binAll(readers, 
                                         jetMask, evtMask,
                                         nomweight)
            nominal['sumwt'] = ak.sum(nomweight, axis=None)
            nominal['sumwt_pass'] = ak.sum(nomweight[evtMask], axis=None)
            nominal['numjet'] = ak.sum(jetMask * nomweight, axis=None)
            if 'reco' in nominal:
                nominal['sumwt_reco'] = nominal['reco'][:,0].sum()

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
                if 'reco' in resultdict[variation]:
                    resultdict[variation]['sumwt_reco'] = resultdict[variation]['reco'][:,0].sum()

    def process(self, events):
        #setup inputs
        t0 = time()
        self.binner.isMC = self.isMC

        readers = AllReaders(events, self.config, 
                             self.noRoccoR,
                             self.noJER, self.noJEC)

        readers.runJEC(self.era, self.verbose)
        readers.checkBtags(self.config)

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

        if self.verbose:
            print("SUMWT", result['nominal']['sumwt'])
            print("SUMWT_PASS", result['nominal']['sumwt_pass'])
            print("NUMJET", result['nominal']['numjet'])
            if 'reco' in result['nominal']:
                print("SUMWT_RECO", result['nominal']['sumwt_reco'])

        result['config'] = self.config

        return result
