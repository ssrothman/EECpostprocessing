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
    def res4(self):
        if not hasattr(self, '_res4'):
            self._res4 = reading.readEEC.getRes4shapes(self._x, self._name)
        return self._res4

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
    def rawmuons(self):
        if not hasattr(self, '_rawmuons'):
            self._rawmuons = getRawMuons(self._x, self._name)
        return self._rawmuons

    @property
    def Zs(self):
        if not hasattr(self, '_Zs'):
            self._Zs = self._muons[:,0] + self._muons[:,1]
            M = self._Zs.mass
            E = self._Zs.energy
            cosh = np.cosh(self._Zs.eta)
            sinh = np.sinh(self._Zs.eta)
            PT = self._Zs.pt
            numerator = np.sqrt(M*M + np.square(PT*cosh)) + PT*sinh
            denominator = np.sqrt(M*M + np.square(PT))
            self._Zs['y'] = np.log(numerator/denominator)

            #pz = self._Zs.pz
            #y2 = ak.to_numpy(0.5 * np.log((E + pz)/(E - pz)))
            #import matplotlib.pyplot as plt
            #y1 = ak.to_numpy(self._Zs.y)
            #plt.hist(y1/y2, bins=100, range=(0.9, 1.1))
            #plt.show()
            #assert np.allclose(ak.to_numpy(self._Zs.y),
            #                   ak.to_numpy(y2))
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
