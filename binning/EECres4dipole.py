import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

from .EECgeneric import EECgenericBinner

class EECres4dipoleBinner(EECgenericBinner):
    def __init__(self, config,
                 manualcov,
                 poissonbootstrap, 
                 skipBtag,
                 statsplit,
                 sepPt):
        super(EECres4dipoleBinner, self).__init__(config,
                                            manualcov, 
                                            poissonbootstrap,
                                            skipBtag,
                                            statsplit, 
                                            sepPt)

    def binAll(self, readers, mask, evtMask, wt):
        result = {}
        result['reco'] = self.binObserved(
                readers.rRecoEEC.res4dipole,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        
        result['gen'] = self.binObserved(
                readers.rGenEEC.res4dipole,
                readers.rGenJet,
                readers.rGenEEC.iJet,
                readers.rGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['unmatchedReco'] = self.binObserved(
                readers.rUnmatchedRecoEEC.res4dipole,
                readers.rRecoJet,
                readers.rUnmatchedRecoEEC.iJet,
                readers.rUnmatchedRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['unmatchedGen'] = self.binObserved(
                readers.rUnmatchedGenEEC.res4dipole,
                readers.rGenJet,
                readers.rUnmatchedGenEEC.iJet,
                readers.rUnmatchedGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['untransferedReco'] = self.binObserved(
                readers.rUntransferedRecoEEC.res4dipole,
                readers.rRecoJet,
                readers.rUntransferedRecoEEC.iReco,
                readers.rUntransferedRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['untransferedGen'] = self.binObserved(
                readers.rUntransferedGenEEC.res4dipole,
                readers.rGenJet,
                readers.rUntransferedGenEEC.iJet,
                readers.rUntransferedGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        return result
    
