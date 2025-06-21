import numpy as np
import awkward as ak
import hist
from skimming.util import *
from time import time
import os.path

from .EECgeneric import EECgenericSkimmer

class EECres4dipoleSkimmer(EECgenericSkimmer):
    def __init__(self, *args, **kwargs):
        super(EECres4dipoleSkimmer, self).__init__(*args, **kwargs)

    def skimAll(self, readers, mask, evtMask, wtVars, basepath):
        result = {}
        result['reco'] = self.skimObserved(
                readers.rRecoEEC.res4dipole,
                readers.rRecoEEC.ptDenom, 4,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wtVars,
                os.path.join(basepath, 'reco'),
                isRes=True)
        
        if self.isMC:
            result['gen'] = self.skimObserved(
                    readers.rGenEEC.res4dipole,
                    readers.rGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'gen'),
                    isRes=True)

            result['unmatchedReco'] = self.skimObserved(
                    readers.rUnmatchedRecoEEC.res4dipole,
                    readers.rUnmatchedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUnmatchedRecoEEC.iJet,
                    readers.rUnmatchedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedReco'),
                    isRes=True)

            result['unmatchedGen'] = self.skimObserved(
                    readers.rUnmatchedGenEEC.res4dipole,
                    readers.rUnmatchedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUnmatchedGenEEC.iJet,
                    readers.rUnmatchedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedGen'),
                    isRes=True)

            result['untransferedReco'] = self.skimObserved(
                    readers.rUntransferedRecoEEC.res4dipole,
                    readers.rUntransferedRecoEEC.ptDenom, 4,
                    readers.rRecoJet,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.rUntransferedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedReco'),
                    isRes=True)

            result['untransferedGen'] = self.skimObserved(
                    readers.rUntransferedGenEEC.res4dipole,
                    readers.rUntransferedGenEEC.ptDenom, 4,
                    readers.rGenJet,
                    readers.rUntransferedGenEEC.iJet,
                    readers.rUntransferedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'untransferedGen'),
                    isRes=True)

            result['transfer'] = self.skimTransfer(
                readers.rTransfer.res4dipole,
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
    
