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
                t0 = time()
                transfers.append(
                    self.binTransfer(
                        readers.rTransfer.proj(order),
                        readers.rGenJet,
                        readers.rRecoJet,
                        readers.rTransfer.iGen,
                        readers.rTransfer.iReco,
                        readers.eventIdx,
                        mask, wt
                    )[None,:]
                )
                #print("Transfer %d took %g"%(order, time()-t0))

            t0 = time()
            transfer = np.concatenate(transfers, axis=0)
            #print("Transfer concatenate took %g"%(time()-t0))

            t0 = time()
            transferPT = self.binTransferPTfactor(
                readers.rTransfer.proj(2),
                readers.rGenJet,
                readers.rRecoJet,
                readers.rTransfer.iGen,
                readers.rTransfer.iReco,
                readers.eventIdx,
                mask, wt
            )
            #print("TransferPT took %g"%(time()-t0))

            t0 = time()
            transferBtag = self.binTransferBTAGfactor(
                readers.rTransfer.proj(2),
                readers.rGenJet,
                readers.rRecoJet,
                readers.rTransfer.iGen,
                readers.rTransfer.iReco,
                readers.eventIdx,
                mask, wt
            )
            #print("TransferBtag took %g"%(time()-t0))
        #print("SUMTRANSFER", transfer.sum())

        t0 = time()
        reco = self.binObserved(
                readers.rRecoEEC.allproj,
                readers.rRecoJet,
                readers.rRecoEEC.iJet,
                readers.rRecoEEC.iReco,
                readers.eventIdx,
                mask, wt)
        #print("Reco took %g"%(time()-t0))
        if self.isMC:
            t0 = time()
            recopure = self.binObserved(
                    readers.rRecoEEC.allproj,
                    readers.rRecoJet,
                    readers.rRecoEEC.iJet,
                    readers.rRecoEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rRecoEECUNMATCH.allproj)
            #print("Recopure took %g"%(time()-t0))
            t0 = time()
            gen = self.binObserved(
                    readers.rGenEEC.allproj,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt)
            #print("Gen took %g"%(time()-t0))
            t0 = time()
            genpure = self.binObserved(
                    readers.rGenEEC.allproj,
                    readers.rGenJet,
                    readers.rGenEEC.iJet,
                    readers.rGenEEC.iReco,
                    readers.eventIdx,
                    mask, wt, 
                    noCov=True,
                    subtract = readers.rGenEECUNMATCH.allproj)
            #print("Genpure took %g"%(time()-t0))

        result = {}
        if self.isMC:
            result['recopure'] = recopure
            result['genpure'] = genpure
            result['transfer'] = transfer
            result['transferPT'] = transferPT
            result['transferBtag'] = transferBtag
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
    
