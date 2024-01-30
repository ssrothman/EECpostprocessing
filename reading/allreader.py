from .reader import *

class AllReaders:
    def __init__(self, x, config, i=0):
        self._rRecoJet = jetreader(x, 
            config.names.puppijets, 
            config.names.simonjets,
            config.names.CHSjets)
        self._rGenJet = jetreader(x,
            config.names.genjets,
            config.names.gensimonjets,
            None)
        self._rGenEEC = EECreader(x,
            'Gen' + config.EECnames[i])
        self._rGenEECUNMATCH = EECreader(x,
            'Gen' + config.EECnames[i] + 'UNMATCH')
        self._rRecoEEC = EECreader(x,
            'Reco' + config.EECnames[i])
        self._rRecoEECUNMATCH = EECreader(x,
            'Reco' + config.EECnames[i] + 'PU')
        self._rTransfer = transferreader(x,
            config.EECnames[i] + 'Transfer')
        self._rMu = muonreader(x, config.names.muons)
        self._nPU = x.Pileup.nPU
        self._HLT = x.HLT

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
    def HLT(self):
        return self._HLT
