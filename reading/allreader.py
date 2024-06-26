from .reader import *
from .JERC import JERC_handler

class AllReaders:
    def __init__(self, x, config, 
                 noRoccoR=False,
                 noJER=False, noJEC=False):
        self._config = config

        self.noRoccoR = noRoccoR
        self.noJER = noJER
        self.noJEC = noJEC

        self._rRecoJet = jetreader(x, 
            config.names.puppijets, 
            config.names.SimonJets,
            config.names.CHSjets)
        self._rGenJet = jetreader(x,
            config.names.genjets,
            config.names.GenSimonJets,
            None)
        self._rGenEEC = EECreader(x,
            'Gen' + config.names.EECs)
        self._rGenEECUNMATCH = EECreader(x,
            'Gen' + config.names.EECs + 'UNMATCH')
        self._rRecoEEC = EECreader(x,
            'Reco' + config.names.EECs)
        self._rRecoEECUNMATCH = EECreader(x,
            'Reco' + config.names.EECs + 'PU')
        self._rTransfer = transferreader(x,
            config.names.EECs + 'Transfer')
        self._rMu = muonreader(x, config.names.muons, noRoccoR)
        self._rMatch = matchreader(x, config.names.Matches)

        self._rho = x[config.names.rho]
        
        if hasattr(x, "Pileup"):
            self._nPU = x.Pileup.nPU
            self._nTrueInt = x.Pileup.nTrueInt
        else:
            self._nPU = None
        self._HLT = x.HLT
        if hasattr(config.names, "ControlJets"):
            self._rControlJet = jetreader(x,
                None,
                config.names.ControlJets,
                None)
            self._rControlEEC = EECreader(x,
                'Reco' + config.names.ControlEECs)
        else:
            self._rControlJet = None
            self._rControlEEC = None

    def checkBtags(self, config):
        bvals = self.rRecoJet.CHSjets.btagDeepFlavB

        loosewp = config.tagging.bwps.loose
        mediumwp = config.tagging.bwps.medium
        tightwp = config.tagging.bwps.tight

        self.rRecoJet.jets['passLooseB'] = ak.max(bvals > loosewp, axis=-1)
        self.rRecoJet.jets['passMediumB'] = ak.max(bvals > mediumwp, axis=-1)
        self.rRecoJet.jets['passTightB'] = ak.max(bvals > tightwp, axis=-1)

        if hasattr(self.rRecoJet._x, "Pileup"):
            genpass = self.rGenJet.jets.hadronFlavour == 5
            self.rGenJet.jets['passLooseB'] = genpass
            self.rGenJet.jets['passMediumB'] = genpass
            self.rGenJet.jets['passTightB'] = genpass

    def runJEC(self, era, syst, syst_updn):
        #print("TOP OF RUN JEC")
        #print("\tera is", era)
        from time import time
        t0 = time()
        handler = JERC_handler(self._config.JERC,
                               self.noJER, self.noJEC)
        t1 = time()
        corrjets = handler.setup_factory(self, era)
        t2 = time()
        if 'JER' in syst:
            if syst_updn == 'UP':
                #print("JER UP")
                corrpt = corrjets.JER['up'].pt
            else:
                #print("JER DOWN")
                corrpt = corrjets.JER['down'].pt
        elif 'JES' in syst:
            JESs = []
            for field in corrjets.fields:
                if field.startswith("jet_energy_uncertainty"):
                    JESs.append(corrjets[field][:,:,0] - 1)
    
            JEStot = np.sqrt(ak.sum([np.square(JES) for JES in JESs], axis=0))

            if syst_updn == 'UP':
                #print("JES UP")
                JEStot = 1+JEStot
            else:
                #print("JES DOWN")
                JEStot = 1-JEStot

            corrpt = corrjets.pt * JEStot
        else:
            corrpt = corrjets.pt
        t3 = time()

        #print("corr/raw")
        #print(ak.min(corrpt/corrjets.pt_raw))
        #print(ak.max(corrpt/corrjets.pt_raw))
        #print()
        #print("corr/cmssw")
        #print(ak.min(corrpt/self.rRecoJet.jets.pt))
        #print(ak.max(corrpt/self.rRecoJet.jets.pt))

        self.rRecoJet.jets['corrpt'] = corrpt
        if hasattr(self.rRecoJet._x, "Pileup"):
            self.rGenJet.jets['corrpt'] = self.rGenJet.jets.pt #just so the genjets.corrpt is defined
        t4 = time()

        print("JERC timing summary:")
        print("\tsetup: %g"%(t1-t0))
        print("\tsetup_factory: %g"%(t2-t1))
        print("\tsystematics: %g"%(t3-t2))
        print("\tset answer: %g"%(t4-t3))

    @property
    def rMatch(self):
        return self._rMatch

    @property
    def rho(self):
        return self._rho

    @property
    def rControlJet(self):
        return self._rControlJet

    @property
    def rControlEEC(self):
        return self._rControlEEC

    @property
    def rRecoJet(self):
        return self._rRecoJet

    @property
    def rGenJet(self):
        return self._rGenJet

    @property
    def rGenEEC(self):
        return self._rGenEEC

    @property
    def rGenEECUNMATCH(self):
        return self._rGenEECUNMATCH

    @property
    def rRecoEEC(self):
        return self._rRecoEEC

    @property
    def rRecoEECUNMATCH(self):
        return self._rRecoEECUNMATCH

    @property
    def rTransfer(self):
        return self._rTransfer

    @property
    def rMu(self):
        return self._rMu

    @property
    def nPU(self):
        return self._nPU

    @property
    def nTrueInt(self):
        return self._nTrueInt

    @property
    def HLT(self):
        return self._HLT
