from .reader import *
from .JERC import JERC_handler

JETMET_SYSTS = [
    'JER_UP', 'JER_UN',
    'JES_UP', 'JES_DN',
    'UNCLUSTERED_UP', 'UNCLUSTERED_DN'
]

class AllReaders:
    def __init__(self, x, config, 
                 noRoccoR=False,
                 noJER=False, noJEC=False, noJUNC=False,
                 syst='nominal'):

        self.syst = syst

        self.isMC = hasattr(x, 'Generator')

        self._config = config

        self.noRoccoR = noRoccoR
        self.noJER = noJER
        self.noJEC = noJEC
        self.noJUNC = noJUNC

        if hasattr(x, "Flag"):
            self._Flag = x.Flag

        if hasattr(x, "L1PreFiringWeight"):
            self._PrefireWeight = x.L1PreFiringWeight

        self._rRecoJet = jetreader(x, 
            config.names.puppijets, 
            config.names.SimonJets,
            config.names.CHSjets)

        if self.isMC:
            self._rGenJet = jetreader(x,
                config.names.genjets,
                config.names.GenSimonJets,
                None)
            self._rGenEEC = EECreader(x,
                'Gen' + config.names.EECs)
            self._rUnmatchedGenEEC = EECreader(x,
                'UnmatchedGen' + config.names.EECs)
            self._rUntransferedGenEEC = EECreader(x,
                'UntransferedGen' + config.names.EECs)
            self._rTransfer = transferreader(x,
                config.names.EECs + 'Transfer')
            self._rMatch = matchreader(x, config.names.Matches)

            self._rUnmatchedRecoEEC = EECreader(x,
                'UnmatchedReco' + config.names.EECs)
            self._rUntransferedRecoEEC = EECreader(x,
                'UntransferedReco' + config.names.EECs)

        self._rRecoEEC = EECreader(x,
            'Reco' + config.names.EECs)

        self._rMu = muonreader(x, config.names.muons, noRoccoR or not config.muons.applyRoccoR)

        if hasattr(config.names, 'rho'):
            self._rho = x[config.names.rho]

        self._event = x.event

        if hasattr(config.names, 'MET'):
            self._MET = x[config.names.MET]
        
        if hasattr(x, "Pileup"):
            self._nPU = x.Pileup.nPU
            self._nTrueInt = x.Pileup.nTrueInt
            self._genwt = x.genWeight

        if hasattr(x, "LHE"):
            self._LHE = x.LHE
            self._scalewt = x.LHEScaleWeight
            self._psweight = x.PSWeight
            self._pdfwt = x.LHEPdfWeight
        else:
            self._LHE = None
            self._scalewt = None
            self._psweight = None
            self._pdfwt = None

        if hasattr(x, "HLT"):
            self._HLT = x.HLT

        if hasattr(config.names, "ControlJets"):
            self._rControlJet = jetreader(x,
                None,
                config.names.ControlJets,
                None)
            self._rControlEEC = EECreader(x,
                'Reco' + config.names.ControlEECs)

    def checkBtags(self, config):
        if not hasattr(config, 'tagging'):
            self.rRecoJet.jets['passB'] = False
            self.rRecoJet.jets['passLooseB'] = False
            self.rRecoJet.jets['passMediumB'] = False
            self.rRecoJet.jets['passTightB'] = False
            return

        if hasattr(self.rRecoJet.jets, "partonFlavour"):
            self.rRecoJet.jets['hadronFlavour'] = ak.where((self.rRecoJet.jets['hadronFlavour'] == 0) & (self.rRecoJet.jets.partonFlavour==21), 21, self.rRecoJet.jets['hadronFlavour'])

        loosewp = config.tagging.bwps.loose
        mediumwp = config.tagging.bwps.medium
        tightwp = config.tagging.bwps.tight

        bvals = self.rRecoJet.simonjets.CHSbtagDeepFlavB

        self.rRecoJet.jets['passLooseB'] = bvals > loosewp
        self.rRecoJet.jets['passMediumB'] = bvals > mediumwp
        self.rRecoJet.jets['passTightB'] = bvals > tightwp

        if config.tagging.wp == 'tight':
            passname = 'passTightB'
        elif config.tagging.wp == 'medium':
            passnme = 'passMediumB'
        elif config.tagging.wp == 'loose':
            passname = 'passLooseB'
        else:
            raise ValueError("Unknown b-tagging working point")

        self.rRecoJet.jets['passB'] = self.rRecoJet.jets[passname]

        CHSval = self.rRecoJet.CHSjets.btagDeepFlavB

        self.rRecoJet.CHSjets['passLooseB'] = CHSval > loosewp
        self.rRecoJet.CHSjets['passMediumB'] = CHSval > mediumwp
        self.rRecoJet.CHSjets['passTightB'] = CHSval > tightwp
        self.rRecoJet.CHSjets['passB'] = self.rRecoJet.CHSjets[passname]

        CHSval2 = self.rRecoJet.matchedCHSjets.btagDeepFlavB
        self.rRecoJet.matchedCHSjets['passLooseB'] = CHSval2 > loosewp
        self.rRecoJet.matchedCHSjets['passMediumB'] = CHSval2 > mediumwp
        self.rRecoJet.matchedCHSjets['passTightB'] = CHSval2 > tightwp
        self.rRecoJet.matchedCHSjets['passB'] = self.rRecoJet.matchedCHSjets[passname]

        if self.isMC:
            genpass = self.rGenJet.jets.hadronFlavour == 5

            self.rGenJet.jets['passLooseB'] = genpass
            self.rGenJet.jets['passMediumB'] = genpass
            self.rGenJet.jets['passTightB'] = genpass
            self.rGenJet.jets['passB'] = genpass

    def runJEC(self, era, verbose):
        if era != 'skip':
            handler = JERC_handler(self._config.JERC,
                                   self.noJER, self.noJEC,
                                   self.noJUNC,
                                   verbose)
            corrjets = handler.setup_factory(self, era)

            nominal = corrjets.pt

            if era == 'MC':
                if not self.noJER:
                    JER_UP = corrjets.JER['up'].pt
                    JER_DN = corrjets.JER['down'].pt

                    self.rRecoJet.jets['JER_UP'] = JER_UP
                    self.rRecoJet.jets['JER_DN'] = JER_DN

                if not self.noJUNC:
                    JESs = []
                    for field in corrjets.fields:
                        if field.startswith("jet_energy_uncertainty"):
                            JESs.append(corrjets[field][:,:,0] - 1)

                    JEStot = np.sqrt(ak.sum([np.square(JES) for JES in JESs], axis=0))

                    JES_UP = corrjets.pt * (1+JEStot)
                    JES_DN = corrjets.pt * (1-JEStot)

                    self.rRecoJet.jets['JES_UP'] = JES_UP
                    self.rRecoJet.jets['JES_DN'] = JES_DN

                self.rRecoJet.jets['CMSSWpt'] = self.rRecoJet.jets.pt

                #null so that we can't accidentally use uncorrected pT
                self.rRecoJet.jets['pt'] = None
                self.rRecoJet.simonjets['jetPt'] = None

                #just so the genjets.corrpt is defined
                self.rGenJet.jets['corrpt'] = self.rGenJet.jets.pt

                if self.syst == 'JER_UP':
                    self.rRecoJet.jets['corrpt'] = JER_UP
                    self.METpt = self.MET.ptJERUp
                elif self.syst == 'JER_DN':
                    self.rRecoJet.jets['corrpt'] = JER_DN
                    self.METpt = self.MET.ptJERDown
                elif self.syst == 'JES_UP':
                    self.rRecoJet.jets['corrpt'] = JES_UP
                    self.METpt = self.MET.ptJESUp
                elif self.syst == 'JES_DN':
                    self.rRecoJet.jets['corrpt'] = JES_DN
                    self.METpt = self.MET.ptJESDown
                elif self.syst == 'UNCLUSTERED_UP':
                    self.rRecoJet.jets['corrpt'] = nominal
                    self.METpt = self.MET.ptUnclusteredUp
                elif self.syst == 'UNCLUSTERED_DN':
                    self.rRecoJet.jets['corrpt'] = nominal
                    self.METpt = self.MET.ptUnclusteredDown
                else:   
                    self.rRecoJet.jets['corrpt'] = nominal
                    self.METpt = self.MET.pt

            else: #data doesn't have JER/JES variations
                self.rRecoJet.jets['corrpt'] = nominal
                self.rRecoJet.jets['CMSSWpt'] = self.rRecoJet.jets.pt
                self.METpt = self.MET.pt

                #null so that we can't accidentally use uncorrected pT
                self.rRecoJet.jets['pt'] = None
                self.rRecoJet.simonjets['jetPt'] = None

                if self.syst in JETMET_SYSTS:
                    raise ValueError("JETMET systematics not available in data")

        else: #era == 'skip'
            self.rRecoJet.jets['corrpt'] = self.rRecoJet.jets.pt
            self.rGenJet.jets['corrpt'] = self.rGenJet.jets.pt
            self.METpt = self.MET.pt
            if self.syst in JETMET_SYSTS:
                raise ValueError("JETMET systematics not available in 'skip' mode")

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
    def rUnmatchedGenEEC(self):
        return self._rUnmatchedGenEEC

    @property
    def rUntransferedGenEEC(self):
        return self._rUntransferedGenEEC

    @property
    def rRecoEEC(self):
        return self._rRecoEEC

    @property
    def rUnmatchedRecoEEC(self):
        return self._rUnmatchedRecoEEC

    @property
    def rUntransferedRecoEEC(self):
        return self._rUntransferedRecoEEC

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

    @property
    def eventIdx(self):
        return self._event

    @property
    def MET(self):
        return self._MET
    
    @property
    def LHE(self):
        return self._LHE

    @property
    def scalewt(self):
        return self._scalewt

    @property
    def psweight(self):
        return self._psweight

    @property
    def pdfwt(self):
        return self._pdfwt

    @property
    def genwt(self):
        return self._genwt

    @property
    def prefirewt(self):
        return self._PrefireWeight

    @property
    def Flag(self):
        return self._Flag

