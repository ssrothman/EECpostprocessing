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

    def get_JER_SFs(self, rJet, isMC):
        if isMC:
            SFfunc = self.evaluator[self.config.JER.MC.sf]
            resfunc = self.evalutor[self.config.JER.MC.resolution]
        else:
            SFfunc = self.evaluator[self.config.JER.DATA.sf]
            resfunc = self.evalutor[self.config.JER.DATA.resolution]

        SF = SFfunc(rJet.pt, rJet.eta)
        res = resfunc(rJet.pt, rJet.eta)
        return SF, res

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
            #this is really dumb.....
            match_iGen = allreaders.rMatch.iGen
            match_iReco = allreaders.rMatch.iReco
            evtidx = ak.local_index(match_iReco, axis=0)
            evtidx, _ = ak.broadcast_arrays(evtidx, match_iReco)

            jetidx = ak.local_index(allreaders.rRecoJet.jets.pt, 
                                    axis=1)
            maxlen = ak.max(jetidx)+1
            numevt = len(jetidx)

            iGen = np.ones((numevt, maxlen), dtype=np.int32) * -1
            iGen[ak.flatten(evtidx), ak.flatten(match_iReco)] = \
                    ak.flatten(match_iGen)
    
            iGen = np.ma.MaskedArray(iGen, iGen == -1)
            iGen = ak.Array(iGen)
            iGen = iGen[jetidx]

            genpt = allreaders.rGenJet.jets.pt[iGen]
            genpt = ak.fill_none(genpt, 0)
            genpt = ak.values_astype(genpt, np.float32)

            geneta = allreaders.rGenJet.jets.eta[iGen]
            geneta = ak.fill_none(geneta, 999999)
            geneta = ak.values_astype(geneta, np.float32)

            genphi = allreaders.rGenJet.jets.phi[iGen]
            genphi = ak.fill_none(genphi, 999999)
            genphi = ak.values_astype(genphi, np.float32)

            allreaders.rRecoJet.jets['pt_gen'] = genpt
            allreaders.rRecoJet.jets['eta_gen'] = geneta
            allreaders.rRecoJet.jets['phi_gen'] = genphi


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

    def runJEC(self, rJet, era):
        corrector = FactorizedJetCorrector(**{key: self.evaluator[key] for key in stack})

        factors = corrector.getCorrection(
            JetEta = rJet.jets.eta,
            Rho = rJet.jets.event_rho,
            JetPt = rJet.jets.rawPt,
            JetA = rJet.jets.area,
        )

        return factors

    def getFactory(self, rJet, era):
        self.setup_JEC_inputs(rJet)

