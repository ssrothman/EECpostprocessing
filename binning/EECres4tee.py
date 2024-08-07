import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

from .EECgeneric import EECgenericBinner

class EECres4teeBinner(EECgenericBinner):
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit,
                 sepPt):
        super(EECres4teeBinner, self).__init__(config,
                                            manualcov, poissonbootstrap,
                                            statsplit, sepPt)

    def binAll(self, readers, mask, evtMask, wt):
        transfer = self.binTransfer(
            readers.rTransfer.res4tee,
            readers.rGenJet,
            readers.rRecoJet,
            readers.rTransfer.iGen,
            readers.rTransfer.iReco,
            readers.eventIdx,
            mask, wt
        )

        reco = self.binObserved(
                readers.rRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        recopure = self.binObserved(
                readers.rRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt, 
                noCov=True,
                subtract = readers.rRecoEECUNMATCH.res4tee)
        gen = self.binObserved(
                readers.rGenEEC.res4tee,
                readers.rGenJet,
                readers.rGenEEC.iJet,
                readers.rGenEEC.iReco,
                readers.eventIdx,
                mask, wt)
        genpure = self.binObserved(
                readers.rGenEEC.res4tee,
                readers.rGenJet,
                readers.rGenEEC.iJet,
                readers.rGenEEC.iReco,
                readers.eventIdx,
                mask, wt, 
                noCov=True,
                subtract = readers.rGenEECUNMATCH.res4tee)

        result = {}
        result['recopure'] = recopure
        result['genpure'] = genpure
        result['transfer'] = transfer
        if self.manualcov:
            result['reco'] = reco[0]
            result['covreco'] = reco[1]
            result['gen'] = gen[0]
            result['covgen'] = gen[1]
        else:
            result['reco'] = reco
            result['gen'] = gen

        return result
    
