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
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def muons(self):
        if not hasattr(self, '_muons'):
            self._muons = getMuons(self._x, self._name)
        return self._muons

    @property
    def Zs(self):
        if not hasattr(self, '_Zs'):
            self._Zs = self._muons[:,0] + self._muons[:,1]
            M = self._Zs.mass
            cosh = np.cosh(self._Zs.eta)
            sinh = np.sinh(self._Zs.eta)
            PT = self._Zs.pt
            numerator = np.sqrt(M*M + np.square(PT*cosh)) + PT*sinh
            denominator = np.sqrt(M*M + np.square(PT))
            self._Zs['y'] = np.log(numerator/denominator)
        return self._Zs
