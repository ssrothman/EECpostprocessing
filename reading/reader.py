from .readMu import * 
from .readJets import *

class resolutionreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def res(self):
        if not hasattr(self, '_res'):
            self._res = getResolutionStudy(self._x, self._name)
        return self._res

    @property
    def resIdx(self):
        if not hasattr(self, '_resIdx'):
            self._resIdx = getResolutionStudyIdx(self._x, self._name)
        return self._resIdx

import reading.readEEC

class EECreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def proj(self):
        if not hasattr(self, '_proj'):
            self._proj = reading.readEEC.getProj(self._x, self._name, 'value2')
        return self._proj

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

    '''
    ak.sum(transferP[evt, jet], axis=0) = reco - PU
    '''
    @property
    def proj(self):
        if not hasattr(self, '_transferP'):
            self._transferP = reading.readEEC.getTransferP(self._x, 
                                                           self._name, 
                                                           'value2')
        return self._transferP

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
    def __init__(self, x, jetsname, simonjetsname):
        self._x = x
        self._jetsname = jetsname
        self._simonjetsname = simonjetsname

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
    def alljets(self):
        return self._x[self._jetsname]

class genjetreader:
    def __init__(self, x, jetsname, simonjetsname, transfername):
        self._x = x
        self._jetsname = jetsname
        self._simonjetsname = simonjetsname
        self._transfername = transfername
        self.jetsreader_ = jetreader(x, jetsname, simonjetsname)

    @property
    def transferIdx(self):
        if not hasattr(self, '_transferIdx'):
            self._transferIdx = gettransferGenIdx(
                    self._x, self._transfername)
        return self._transferIdx

    @property
    def parts(self):
        if not hasattr(self, '_parts'):
            self._parts = self.jetsreader_.parts[self.transferIdx]
        return self._parts

    @property
    def jets(self):
        if not hasattr(self, '_jets'):
            self._jets = self.jetsreader_.jets[self.transferIdx]
        return self._jets

    @property
    def alljets(self):
        return self.jetsreader_.alljets

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
