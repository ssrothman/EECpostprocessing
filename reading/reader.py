from .readEEC import *
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

class EECreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def proj(self):
        if not hasattr(self, '_proj'):
            self._proj = getproj(self._x, self._name)
        return self._proj

    @property
    def projdR(self):
        if not hasattr(self, '_projdR'):
            self._projdR = getprojdR(self._x, self._name)
        return self._projdR

    @property
    def projOrder(self):
        if not hasattr(self, '_projOrder'):
            self._projOrder = getprojOrder(self._x, self._name)
        return self._projOrder

    @property
    def projJetIdx(self):
        if not hasattr(self, '_projJetIdx'):
            self._projJetIdx = getprojJetIdx(self._x, self._name)
        return self._projJetIdx

    @property
    def res3(self):
        if not hasattr(self, '_res3'):
            self._res3 = getres3(self._x, self._name)
        return self._res3

    @property
    def res4(self):
        if not hasattr(self, '_res4'):
            self._res4 = getres4(self._x, self._name)
        return self._res4

    @property
    def covPxP(self):
        if not hasattr(self, '_covPxP'):
            self._covPxP = getcovPxP(self._x, self._name)
        return self._covPxP

    @property
    def cov3x3(self):
        if not hasattr(self, '_cov3x3'):
            self._cov3x3 = getcov3x3(self._x, self._name)
        return self._cov3x3

    @property
    def cov3xP(self):
        if not hasattr(self, '_cov3xP'):
            self._cov3xP = getcov3xP(self._x, self._name)
        return self._cov3xP  

    @property
    def cov4x4(self):
        if not hasattr(self, '_cov4x4'):
            self._cov4x4 = getcov4x4(self._x, self._name)
        return self._cov4x4

    @property
    def cov4x3(self):
        if not hasattr(self, '_cov4x3'):
            self._cov4x3 = getcov4x3(self._x, self._name)
        return self._cov4x3

    @property
    def cov4xP(self):
        if not hasattr(self, '_cov4xP'):
            self._cov4xP = getcov4xP(self._x, self._name)
        return self._cov4xP

    @property
    def projdR_forhist(self):
        if not hasattr(self, '_projdR_forhist'):
            
            self._projdR_forhist, _ = ak.broadcast_arrays(
                    self.projdR[:, :, None, :], self.proj)
        return self._projdR_forhist

    @property
    def projdR_forcov(self):
        if not hasattr(self, '_projdR_forcov'):
            self._projdR_forcov = ak.flatten(
                    self.projdR_forhist, axis=-1)
        return self._projdR_forcov

    @property
    def projOrder_forhist(self):
        if not hasattr(self, '_projOrder_forhist'):
            self._projOrder_forhist, _ = ak.broadcast_arrays(
                    self.projOrder, self.proj)
        return self._projOrder_forhist

    @property
    def projOrder_forcov(self):
        if not hasattr(self, '_projOrder_forcov'):
            self._projOrder_forcov = ak.flatten(
                    self.projOrder_forhist, axis=-1)
        return self._projOrder_forcov



class transferreader:
    def __init__(self, x, name):
        self._x = x
        self._name = name

    @property
    def transferP(self):
        if not hasattr(self, '_transferP'):
            self._transferP = gettransferP(self._x, self._name)
        return self._transferP

    @property
    def transferRecoIdx(self):
        if not hasattr(self, '_transferPrecoIdx'):
            self._transferPrecoIdx = gettransferRecoIdx(
                    self._x, self._name)
        return self._transferPrecoIdx

    @property
    def transferGenIdx(self):
        if not hasattr(self, '_transferPgenIdx'):
            self._transferPgenIdx = gettransferGenIdx(
                    self._x, self._name)
        return self._transferPgenIdx

    @property
    def transfer3(self):
        if not hasattr(self, '_transfer3'):
            self._transfer3 = getTransfer3(self._x, self._name)
        return self._transfer3

    @property
    def transfer4(self):
        if not hasattr(self, '_transfer4'):
            self._transfer4 = getTransfer4(self._x, self._name)
        return self._transfer4

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
