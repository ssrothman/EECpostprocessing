from .readMu import * 
from .readJets import *

import reading.readEEC

class EECreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name
        self._proj = {}

    def proj(self, order):
        if order not in self._proj:
            self._proj[order] = reading.readEEC.getProj(
                    self._x, self._name, 'value%d'%order)
        return self._proj[order]

    @property
    def allproj(self):
        if not hasattr(self, '_allproj'):
            self._allproj = reading.readEEC.getAllProj(self._x, self._name)
        return self._allproj

    @property
    def res4dipole(self):
        if not hasattr(self, '_res4'):
            self._res4dipole = reading.readEEC.getRes4dipole(self._x, self._name)
        return self._res4dipole

    @property
    def res4tee(self):
        if not hasattr(self, '_res4tee'):
            self._res4tee = reading.readEEC.getRes4tee(self._x, self._name)
        return self._res4tee

    @property
    def res4triangle(self):
        if not hasattr(self, '_res4triangle'):
            self._res4triangle = reading.readEEC.getRes4triangle(self._x, self._name)
        return self._res4triangle
    
    @property
    def res4minR(self):
        if not hasattr(self, '_res4minR'):
            self._res4minR = reading.readEEC.getRes4minR(self._x, self._name)
        return self._res4minR

    @property
    def res3(self):
        if not hasattr(self, '_res3'):
            self._res3 = reading.readEEC.getRes3(self._x, self._name)
        return self._res3

    @property
    def iJet(self):
        if not hasattr(self, '_iJet'):
            self._iJet = reading.readEEC.getJetIdx(self._x, self._name)
        return self._iJet

    @property
    def iReco(self):
        if not hasattr(self, '_iReco'):
            self._iReco = reading.readEEC.getRecoIdx(self._x, self._name)
        return self._iReco

    @property
    def nproj(self):
        if not hasattr(self, '_nproj'):
            self._nproj = reading.readEEC.getNProj(self._x, self._name)
        return self._nproj
    
class transferreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name
        self._proj = {}

    '''
    ak.sum(transferP[evt, jet], axis=0) = reco - PU
    '''
    def proj(self, order):
        if order not in self._proj:
            self._proj[order] = reading.readEEC.getTransferP(
                self._x, self._name, 'value%d'%order)
        return self._proj[order]

    @property
    def res3(self):
        if not hasattr(self, '_res3'):
            self._res3 = reading.readEEC.getTransferRes3(self._x, self._name)
        return self._res3

    @property
    def iReco(self):
        if not hasattr(self, '_iReco'):
            self._iReco = reading.readEEC.getRecoIdx(self._x, self._name)
        return self._iReco

    @property
    def iGen(self):
        if not hasattr(self, '_iGen'):
            self._iGen = reading.readEEC.getGenIdx(self._x, self._name)
        return self._iGen

class jetreader:
    def __init__(self, x, jetsname, simonjetsname, CHSjetsname):
        self._x = x
        self._jetsname = jetsname
        self._simonjetsname = simonjetsname
        self._CHSjetsname = CHSjetsname

    def check(self):
        return hasattr(self._x, self._jetsname)

    def checkCHS(self):
        return hasattr(self._x, self._CHSjetsname)

    @property
    def parts(self):
        if not hasattr(self, '_parts'):
            self._parts = getParts(self._x, self._simonjetsname)
        return self._parts

    @property
    def simonjets(self):
        if not hasattr(self, '_simonjets'):
            self._simonjets = getSimonJets(self._x, self._simonjetsname)
        return self._simonjets

    @property
    def jets(self):
        if not hasattr(self, '_jets'):
            self._jets = getJets(self._x, self._jetsname, 
                                          self._simonjetsname)
        return self._jets

    @property
    def CHSjets(self):
        if not hasattr(self, '_CHSjets'):
            self._CHSjets = getCHSjets(self._x, self._CHSjetsname, 
                                              self._simonjetsname)
        return self._CHSjets

    @property
    def alljets(self):
        return self._x[self._jetsname]

class muonreader:
    def __init__(self, x, name, noRoccoR):
        self._x = x
        self._name = name
        self.noRoccoR = noRoccoR

    @property
    def muons(self):
        if not hasattr(self, '_muons'):
            self._muons = getMuons(self._x, self._name, self.noRoccoR)
        return self._muons

    @property
    def Zs(self):
        if not hasattr(self, '_Zs'):
            mu0p4 = ak.zip(
                {
                    'pt': self._muons[:,0].pt,
                    'eta': self._muons[:,0].eta,
                    'phi': self._muons[:,0].phi,
                    'mass': self._muons[:,0].mass,
                },
                with_name='PtEtaPhiMLorentzVector'
            )
            mu1p4 = ak.zip(
                {
                    'pt': self._muons[:,1].pt,
                    'eta': self._muons[:,1].eta,
                    'phi': self._muons[:,1].phi,
                    'mass': self._muons[:,1].mass,
                },
                with_name='PtEtaPhiMLorentzVector'
            )
            self._Zs = mu0p4 + mu1p4 
            M = self._Zs.mass
            E = self._Zs.energy
            cosh = np.cosh(self._Zs.eta)
            sinh = np.sinh(self._Zs.eta)
            PT = self._Zs.pt
            numerator = np.sqrt(M*M + np.square(PT*cosh)) + PT*sinh
            denominator = np.sqrt(M*M + np.square(PT))
            self._Zs['rapidity'] = np.log(numerator/denominator)

        return self._Zs

class matchreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def iGen(self):
        if not hasattr(self, '_iGen'):
            self._iGen = self._x[self._name+'BK'].iGen
        return self._iGen

    @property
    def iReco(self):
        if not hasattr(self, '_iReco'):
            self._iReco = self._x[self._name+'BK'].iReco
        return self._iReco
