import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

from .EECgeneric import EECgenericBinner

class EECprojBinner(EECgenericBinner):
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit,
                 sepPt):
        super(EECprojBinner, self).__init__(config,
                                            manualcov, poissonbootstrap,
                                            statsplit, sepPt)

    def binAll(self, readers, mask, evtMask, wt):
        if self.isMC:
            transfers = []
            for order in range(2, 7):
                transfers.append(
                    self.binTransfer(
                        readers.rTransfer.proj(order),
                        readers.rGenJet,
                        readers.rRecoJet,
                        readers.rTransfer.iGen,
                        readers.rTransfer.iReco,
                        readers.eventIdx,
                        mask, wt
                    )[None,:,:,:,:,:,:]
                )
            transfer = np.concatenate(transfers, axis=0)
        #print("SUMTRANSFER", transfer.sum())

        reco = self.binObserved(
                readers.rRecoEEC.allproj,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        if self.isMC:
            recopure = self.binObserved(
                    readers.rRecoEEC.allproj,
                    readers.rRecoJet,
                    readers.rRecoEEC.iJet,
                    readers.rRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rRecoEECUNMATCH.allproj)
            gen = self.binObserved(
                    readers.rGenEEC.allproj,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt)
            genpure = self.binObserved(
                    readers.rGenEEC.allproj,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rGenEECUNMATCH.allproj)

        result = {}
        if self.isMC:
            result['recopure'] = recopure
            result['genpure'] = genpure
            result['transfer'] = transfer
        if self.manualcov:
            result['reco'] = reco[0]
            result['covreco'] = reco[1]
            if self.isMC:
                result['gen'] = gen[0]
                result['covgen'] = gen[1]
        else:
            result['reco'] = reco
            if self.isMC:
                result['gen'] = gen

        return result
    
