from .reader import *
from .JERC import JERC_handler

class AllReaders:
    def __init__(self, x, config, 
                 noRoccoR=False,
                 noJER=False, noJEC=False):

        self.isMC = hasattr(x, 'Generator')

        self._config = config

        self.noRoccoR = noRoccoR
        self.noJER = noJER
        self.noJEC = noJEC

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
            self._rGenEECUNMATCH = EECreader(x,
                'Gen' + config.names.EECs + 'UNMATCH')
            self._rTransfer = transferreader(x,
                config.names.EECs + 'Transfer')
            self._rMatch = matchreader(x, config.names.Matches)

        self._rRecoEEC = EECreader(x,
            'Reco' + config.names.EECs)
        self._rRecoEECUNMATCH = EECreader(x,
            'Reco' + config.names.EECs + 'PU')

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
        if hasattr(self.rRecoJet.jets, "partonFlavour"):
            self.rRecoJet.jets['hadronFlavour'] = ak.where((self.rRecoJet.jets['hadronFlavour'] == 0) & (self.rRecoJet.jets.partonFlavour==21), 21, self.rRecoJet.jets['hadronFlavour'])

        if self.rRecoJet.checkCHS():
            bvals = self.rRecoJet.CHSjets.btagDeepFlavB

            loosewp = config.tagging.bwps.loose
            mediumwp = config.tagging.bwps.medium
            tightwp = config.tagging.bwps.tight

            self.rRecoJet.CHSjets['passLooseB'] = bvals > loosewp
            self.rRecoJet.CHSjets['passMediumB'] = bvals > mediumwp
            self.rRecoJet.CHSjets['passTightB'] = bvals > tightwp

            self.rRecoJet.jets['passLooseB'] = ak.max(bvals > loosewp, axis=-1)
            self.rRecoJet.jets['passMediumB'] = ak.max(bvals > mediumwp, axis=-1)
            self.rRecoJet.jets['passTightB'] = ak.max(bvals > tightwp, axis=-1)

            if config.tagging.wp == 'tight':
                self.rRecoJet.jets['passB'] = self.rRecoJet.jets['passTightB']
            elif config.tagging.wp == 'medium':
                self.rRecoJet.jets['passB'] = self.rRecoJet.jets['passMediumB']
            elif config.tagging.wp == 'loose':
                self.rRecoJet.jets['passB'] = self.rRecoJet.jets['passLooseB']
            elif config.tagging.wp == 'hadronFlavour':
                self.rRecoJet.jets['passB'] = self.rRecoJet.jets.hadronFlavour == 5
            elif config.tagging.wp == 'partonFlavour':
                self.rRecoJet.jets['passB'] = self.rRecoJet.jets.partonFlavour == 5
            else:
                raise ValueError("Unknown b-tagging working point")
        else:
            import warnings
            #warnings.warn("WARNING: no available CHS jets for b-tagging\nfalling back on hadron flavour")
        
            if config.tagging.wp == 'hadronFlavour':
                isB = self.rRecoJet.jets.hadronFlavour == 5
            elif config.tagging.wp == 'partonFlavour':
                isB = self.rRecoJet.jets.partonFlavour == 5
            else:
                raise ValueError("Unknown b-tagging working point [without CHS jets]")

            self.rRecoJet.jets['passLooseB'] = isB
            self.rRecoJet.jets['passMediumB'] = isB
            self.rRecoJet.jets['passTightB'] = isB
            self.rRecoJet.jets['passB'] = isB

        if self.isMC:
            if config.tagging.wp == 'hadronFlavour':
                genpass = self.rGenJet.jets.hadronFlavour == 5
            else:
                genpass = self.rGen

            self.rGenJet.jets['passLooseB'] = genpass
            self.rGenJet.jets['passMediumB'] = genpass
            self.rGenJet.jets['passTightB'] = genpass
            self.rGenJet.jets['passB'] = genpass

    def runJEC(self, era, verbose):
        if era != 'skip':
            handler = JERC_handler(self._config.JERC,
                                   self.noJER, self.noJEC,
                                   verbose)
            corrjets = handler.setup_factory(self, era)

            nominal = corrjets.pt

            if era == 'MC':
                JER_UP = corrjets.JER['up'].pt
                JER_DN = corrjets.JER['down'].pt

                JESs = []
                for field in corrjets.fields:
                    if field.startswith("jet_energy_uncertainty"):
                        JESs.append(corrjets[field][:,:,0] - 1)

                JEStot = np.sqrt(ak.sum([np.square(JES) for JES in JESs], axis=0))

                JES_UP = corrjets.pt * (1+JEStot)
                JES_DN = corrjets.pt * (1-JEStot)

                self.rRecoJet.jets['corrpt'] = nominal
                self.rRecoJet.jets['JER_UP'] = JER_UP
                self.rRecoJet.jets['JER_DN'] = JER_DN
                self.rRecoJet.jets['JES_UP'] = JES_UP
                self.rRecoJet.jets['JES_DN'] = JES_DN
                self.rRecoJet.jets['pt'] = nominal

                #just so the genjets.corrpt is defined
                self.rGenJet.jets['corrpt'] = self.rGenJet.jets.pt

            else: #data doesn't have JER/JES variations
                #TEST
                #print("JERC TEST")
                #print('raw', ak.flatten(self.rRecoJet.jets.pt_raw))
                #print('cmssw', ak.flatten(self.rRecoJet.jets.pt))
                #print('coffea', ak.flatten(corrjets.pt))
                #print('coffea_orig', ak.flatten(corrjets.pt_orig))
                #print('coffea_JEC', 1/ak.flatten(corrjets.jet_energy_correction))
                #print('coffea_pt_JEC', ak.flatten(corrjets.pt_jec))
                #print("cmssw jec factor",ak.flatten(corrjets.jecFactor))
                #print(corrjets.fields)
                #print()

                self.rRecoJet.jets['corrpt'] = nominal
                self.rRecoJet.jets['pt'] = nominal



        else: #eta == 'skip'
            self.rRecoJet.jets['corrpt'] = self.rRecoJet.jets.pt

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

