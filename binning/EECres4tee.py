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
        result = {}
        if self.isMC:
            result['transfer'] = self.binTransfer(
                readers.rTransfer.res4tee,
                readers.rGenJet,
                readers.rRecoJet,
                readers.rTransfer.iGen,
                readers.rTransfer.iReco,
                readers.eventIdx,
                mask, wt
            )
            result['recopure'] = self.binObserved(
                    readers.rRecoEEC.res4tee,
                    readers.rRecoJet,
                    readers.rRecoEEC.iJet,
                    readers.rRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rRecoEECUNMATCH.res4tee)
            result['gen'] = self.binObserved(
                    readers.rGenEEC.res4tee,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt)
            result['genpure'] = self.binObserved(
                    readers.rGenEEC.res4tee,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rGenEECUNMATCH.res4tee)

        result['reco'] = self.binObserved(
                readers.rRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        
        if self.manualcov:
            result['covreco'] = result['reco'][1]
            result['reco'] = result['reco'][0]
            result['covgen'] = result['gen'][1]
            result['gen'] = result['gen'][0]

        return result
    
