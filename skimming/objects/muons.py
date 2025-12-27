import awkward  as ak
import numpy as np
import warnings

class Muons:
    def __init__(self, 
                 events : ak.Array, 
                 name : str, 
                 roccor : bool,
                 roccorThreshold : float,
                 padTo : int):
        
        self._name = name
        self._roccor = roccor
        self._roccorThreshold = roccorThreshold

        self._muons = ak.materialize(events[name])  
        self._muons = ak.pad_none(self._muons, padTo)

        if roccor and not hasattr(self._muons, "RoccoR"):
                warnings.warn("Could not find RoccoR! not applying")
                self._roccor = False

        if roccor:
            self._muons['rawPt'] = self._muons.pt[:]
            self._muons['pt'] = ak.where(self._muons.pt < roccorThreshold, 
                                         self._muons.pt * self._muons.RoccoR,
                                         self._muons.pt)
        else:
            self._muons['rawPt'] = self._muons.pt

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
        cosh = np.cosh(self._Zs.eta)
        sinh = np.sinh(self._Zs.eta)
        PT = self._Zs.pt
        numerator = np.sqrt(M*M + np.square(PT*cosh)) + PT*sinh
        denominator = np.sqrt(M*M + np.square(PT))
        self._Zs['y'] = np.log(numerator/denominator)

    @property
    def muons(self) -> ak.Array:
        return self._muons
    
    @property
    def Zs(self) -> ak.Array:
        return self._Zs

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def roccor(self) -> bool:
        return self._roccor
    
