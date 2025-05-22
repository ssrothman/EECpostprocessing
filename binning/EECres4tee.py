import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time
import os.path

from .EECgeneric import EECgenericBinner

class EECres4teeBinner(EECgenericBinner):
    def __init__(self, *args, **kwargs):
        super(EECres4teeBinner, self).__init__(*args, **kwargs)

    def binAll(self, readers, mask, evtMask, wtVars, basepath):
        result = {}
        result['reco'] = self.binObserved(
                readers.rRecoEEC.res4tee,
                readers.rRecoEEC.ptDenom, 4,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wtVars,
                os.path.join(basepath, 'reco'),
                isRes=True)
        
        if self.isMC:
            result['gen'] = self.binObserved(
                    readers.rGenEEC.res4tee,
                    readers.rGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'gen'),
                    isRes=True)

            result['unmatchedReco'] = self.binObserved(
                    readers.rUnmatchedRecoEEC.res4tee,
                    readers.rUnmatchedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUnmatchedRecoEEC.iJet,
                    readers.rUnmatchedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedReco'),
                    isRes=True)

            result['unmatchedGen'] = self.binObserved(
                    readers.rUnmatchedGenEEC.res4tee,
                    readers.rUnmatchedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUnmatchedGenEEC.iJet,
                    readers.rUnmatchedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedGen'),
                    isRes=True)

            result['untransferedReco'] = self.binObserved(
                    readers.rUntransferedRecoEEC.res4tee,
                    readers.rUntransferedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedReco'),
                    isRes=True)

            result['untransferedGen'] = self.binObserved(
                    readers.rUntransferedGenEEC.res4tee,
                    readers.rUntransferedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUntransferedGenEEC.iJet,
                    readers.rUntransferedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedGen'),
                    isRes=True)

            result['transfer'] = self.binTransfer(
                readers.rTransfer.res4tee,
                readers.rTransfer.ptDenomReco,
                readers.rTransfer.ptDenomGen,
                4,
                readers.rGenJet,
                readers.rRecoJet,
                readers.rTransfer.iGen,
                readers.rTransfer.iReco,
                readers.eventIdx,
                mask, wtVars,
                os.path.join(basepath, 'transfer'),
                isRes=True
            )

        return result
    
