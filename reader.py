from read import *

class reader:
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
    def transferP(self):
        if not hasattr(self, '_transferP'):
            self._transferP = gettransferP(self._x, self._name)
        return self._transferP

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
            self._transfer3 = gettransfer3(self._x, self._name)
        return self._transfer3

    @property
    def transfer4(self):
        if not hasattr(self, '_transfer4'):
            self._transfer4 = gettransfer4(self._x, self._name)
        return self._transfer4

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
