import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

from .EECgeneric import EECgenericBinner

class EECres4teeBinner(EECgenericBinner):
    def __init__(self, config,
                 manualcov, 
                 poissonbootstrap, 
                 skipBtag,
                 statsplit,
                 sepPt):
        super(EECres4teeBinner, self).__init__(config,
                                            manualcov,
                                            poissonbootstrap,
                                            skipBtag,
                                            statsplit, sepPt)

    def binAll(self, readers, mask, evtMask, wt):
        result = {}
        result['reco'] = self.binObserved(
                readers.rRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        
        result['gen'] = self.binObserved(
                readers.rGenEEC.res4tee,
                readers.rGenJet,
                readers.rGenEEC.iJet,
                readers.rGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['unmatchedReco'] = self.binObserved(
                readers.rUnmatchedRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rUnmatchedRecoEEC.iJet,
                readers.rUnmatchedRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['unmatchedGen'] = self.binObserved(
                readers.rUnmatchedGenEEC.res4tee,
                readers.rGenJet,
                readers.rUnmatchedGenEEC.iJet,
                readers.rUnmatchedGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['untransferedReco'] = self.binObserved(
                readers.rUntransferedRecoEEC.res4tee,
                readers.rRecoJet,
                readers.rUntransferedRecoEEC.iReco,
                readers.rUntransferedRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)

        result['untransferedGen'] = self.binObserved(
                readers.rUntransferedGenEEC.res4tee,
                readers.rGenJet,
                readers.rUntransferedGenEEC.iJet,
                readers.rUntransferedGenEEC.iReco,
                readers.eventIdx,
                mask, wt)

        return result
    
