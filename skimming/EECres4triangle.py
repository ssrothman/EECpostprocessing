import numpy as np
import awkward as ak
import hist
from skimming.util import *
from time import time
import os.path

from .EECgeneric import EECgenericSkimmer

class EECres4triangleSkimmer(EECgenericSkimmer):
    def __init__(self, *args, **kwargs):
        super(EECres4triangleSkimmer, self).__init__(*args, **kwargs)

    def skimAll(self, readers, mask, evtMask, wtVars, basepath):
        result = {}
        result['reco'] = self.skimObserved(
                readers.rRecoEEC.res4triangle,
                readers.rRecoEEC.ptDenom, 4,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wtVars,
                os.path.join(basepath, 'reco'),
                rMu = readers.rMu,
                isRes=True)
        
        if self.isMC:
            result['gen'] = self.skimObserved(
                    readers.rGenEEC.res4triangle,
                    readers.rGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'gen'),
                    rMu = readers.rMu,
                    isRes=True)

            result['unmatchedReco'] = self.skimObserved(
                    readers.rUnmatchedRecoEEC.res4triangle,
                    readers.rUnmatchedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUnmatchedRecoEEC.iJet,
                    readers.rUnmatchedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedReco'),
                    rMu = readers.rMu,
                    isRes=True)

            result['unmatchedGen'] = self.skimObserved(
                    readers.rUnmatchedGenEEC.res4triangle,
                    readers.rUnmatchedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUnmatchedGenEEC.iJet,
                    readers.rUnmatchedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedGen'),
                    rMu = readers.rMu,
                    isRes=True)

            result['untransferedReco'] = self.skimObserved(
                    readers.rUntransferedRecoEEC.res4triangle,
                    readers.rUntransferedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedReco'),
                    rMu = readers.rMu,
                    isRes=True)

            result['untransferedGen'] = self.skimObserved(
                    readers.rUntransferedGenEEC.res4triangle,
                    readers.rUntransferedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUntransferedGenEEC.iJet,
                    readers.rUntransferedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedGen'),
                    rMu = readers.rMu,
                    isRes=True)

            result['transfer'] = self.skimTransfer(
                readers.rTransfer.res4triangle,
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
                rMu = readers.rMu,
                isRes=True
            )

        return result
    
