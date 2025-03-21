from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JECStack, CorrectedJetsFactory
import cachetools
import numpy as np
import awkward as ak

class JERC_handler:
    def __init__(self, config,
                 noJER=False, noJEC=False,
                 verbose=False):
        self.config = config

        self.noJER = noJER
        self.noJEC = noJEC

        self.evaluators = {}

        self.verbose=verbose

    def setup_JEC_inputs(self, allreaders, isMC):
        if hasattr(allreaders.rRecoJet.jets, 'jecFactor'):
            allreaders.rRecoJet.jets['pt_raw'] = \
                    allreaders.rRecoJet.jets.pt * \
                    (allreaders.rRecoJet.jets.jecFactor)
            allreaders.rRecoJet.jets['mass_raw'] = \
                    allreaders.rRecoJet.jets.mass * \
                    (allreaders.rRecoJet.jets.jecFactor)
        else:
            allreaders.rRecoJet.jets['pt_raw'] = \
                    allreaders.rRecoJet.jets.pt * \
                    (1- allreaders.rRecoJet.jets.rawFactor)
            allreaders.rRecoJet.jets['mass_raw'] = \
                    allreaders.rRecoJet.jets.mass * \
                    (1- allreaders.rRecoJet.jets.rawFactor)

        allreaders.rRecoJet.jets['event_rho'] = allreaders.rho

        if isMC:
            allreaders.rRecoJet.jets['pt_gen'] = allreaders.rRecoJet.simonjets['jetMatchPt']
            allreaders.rRecoJet.jets['eta_gen'] = allreaders.rRecoJet.simonjets['jetMatchEta']
            allreaders.rRecoJet.jets['phi_gen'] = allreaders.rRecoJet.simonjets['jetMatchPhi']


    def setup_JEC_stack(self, era):
        stacknames = []
        if era == 'MC':
            files = self.config.files.MC

            if not self.noJEC:
                stacknames += self.config.JECstack.MC
                stacknames += self.config.JECuncertainties
            if not self.noJER:
                stacknames += [self.config.JER.resolution]
                stacknames += [self.config.JER.sf]
        elif era == '2018A':
            files = self.config.files.DATA_2018A

            if not self.noJEC:
                stacknames += self.config.JECstack.DATA_2018A
        elif era == '2018B':
            files = self.config.files.DATA_2018B

            if not self.noJEC:
                stacknames = self.config.JECstack.DATA_2018B
        elif era == '2018C':
            files = self.config.files.DATA_2018C

            if not self.noJEC:
                stacknames = self.config.JECstack.DATA_2018C
        elif era == '2018D':
            files = self.config.files.DATA_2018D

            if not self.noJEC:
                stacknames = self.config.JECstack.DATA_2018D

        if era in self.evaluators:
            ev = self.evaluators[era]
        else:
            ex = extractor()

            ex.add_weight_sets(
                ["* * %s"%f for f in files]
            )
            ex.finalize()
            ev = ex.make_evaluator()
            self.evaluators[era] = ev

        stack = JECStack({key: ev[key] for key in stacknames})
        
        return stack

    def setup_factory(self, allreaders, era):
        self.setup_JEC_inputs(allreaders, 'MC' in era)
        stack = self.setup_JEC_stack(era)

        name_map = stack.blank_name_map
        name_map['JetPt'] = 'pt'
        name_map['JetMass'] = 'mass'
        name_map['JetEta'] = 'eta'
        name_map['Rho'] = 'event_rho'
        name_map['JetA'] = 'area'
        name_map['massRaw'] = 'mass_raw'
        name_map['ptRaw'] = 'pt_raw'
        if 'MC' in era:
            name_map['ptGenJet'] = 'pt_gen'

        jec_cache = cachetools.Cache(np.inf)
        jet_factory = CorrectedJetsFactory(name_map, stack)
        corrected_jets = jet_factory.build(allreaders.rRecoJet.jets, 
                                           lazy_cache=jec_cache)

        if self.verbose:
            print()
            print("RUNNING JERC:")
            print("JEC:\n", stack.jec)
            print("JER:\n", stack.jer)
            print("JUNC:\n", stack.junc)
            print("JERSF:\n", stack.jersf)

        return corrected_jets
