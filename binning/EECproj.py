import numpy as np
import os
import awkward as ak
import hist
from binning.util import *
from time import time

from .EECgeneric import EECgenericBinner

class EECprojBinner(EECgenericBinner):
    def __init__(self, *args, **kwargs):
        super(EECprojBinner, self).__init__(*args, **kwargs)

    def binAll(self, readers, mask, evtMask, wtVars, basepath):
        result = {}
        for order in range(2, 7):
            result['reco%d'%order] = self.binObserved(
                readers.rRecoEEC.proj(order),
                readers.rRecoEEC.ptDenom, order,
                readers.rRecoJet, 
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wtVars,
                os.path.join(basepath, 'reco%d'%order),
                isRes=False
            )

            if self.isMC:
                result['gen%d'%order] = self.binObserved(
                    readers.rGenEEC.proj(order),
                    readers.rGenEEC.ptDenom, order,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'gen%d'%order),
                    isRes=False
                )

                result['unmatchedReco%d'%order] = self.binObserved(
                    readers.rUnmatchedRecoEEC.proj(order),
                    readers.rUnmatchedRecoEEC.ptDenom, order,
                    readers.rRecoJet,
                    readers.rUnmatchedRecoEEC.iJet,
                    readers.rUnmatchedRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedReco%d'%order),
                    isRes=False
                )

                result['unmatchedGen%d'%order] = self.binObserved(
                    readers.rUnmatchedGenEEC.proj(order),
                    readers.rUnmatchedGenEEC.ptDenom, order,
                    readers.rGenJet,
                    readers.rUnmatchedGenEEC.iJet,
                    readers.rUnmatchedGenEEC.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'unmatchedGen%d'%order),
                    isRes=False
                )

                result['transfer%d'%order] = self.binTransfer(
                    readers.rTransfer.proj(order),
                    readers.rTransfer.ptDenomReco,
                    readers.rTransfer.ptDenomGen,
                    order,
                    readers.rGenJet,
                    readers.rRecoJet,
                    readers.rTransfer.iGen,
                    readers.rTransfer.iReco,
                    readers.eventIdx,
                    mask, wtVars,
                    os.path.join(basepath, 'transfer%d'%order),
                    isRes=False
                )

        return result
    
