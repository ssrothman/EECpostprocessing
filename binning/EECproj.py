import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

class EECprojBinner:
    def __init__(self, binning_config, tagging_config):
        self.ptax = hist.axis.Variable(binning_config.bins.pt)

    def binTransfer(self, rTransfer,
                    rRecoEEC, rRecoJet,
                    rGenEEC, rGenJet,
                    m, wt):
        nPT = self.ptax.extent
        nProj = ak.flatten(rRecoEEC.nproj)[0]
        nOrder = 5
        nBtag = 2

        ans = np.zeros((nOrder, nBtag, nPT, nProj, nBtag, nPT, nProj), dtype=np.float64)

        iGenJet = rTransfer.iGen
        iRecoJet = rTransfer.iReco

        mask = m[iRecoJet]

        ptReco = rRecoJet.jets.corrpt[iRecoJet][mask]
        ptGen = rGenJet.jets.corrpt[iGenJet][mask]

        btagReco = ak.where(rRecoJet.jets.passTightB[iRecoJet][mask], 1, 0, dtype=np.int32)
        btagGen = ak.where(rRecoJet.jets.hadronFlavour[iRecoJet][mask] == 5, 1, 0, dtype=np.int32)

        for order in range(nOrder):
            vals = (wt * rTransfer.proj(order+2))[mask]

            iDRGen = ak.local_index(vals, axis=2)
            iDRReco = ak.local_index(vals, axis=3)

            ptGen_broadcast, ptReco_broadcast, \
                btagGen_broadcast, btagReco_broadcast, \
                iDRGen, iDRReco = ak.broadcast_arrays(ptGen, ptReco,
                                                      btagGen, btagReco,
                                                      iDRGen, iDRReco)

            iDRGen = squash(iDRGen)
            iDRReco = squash(iDRReco)
            iPTGen = self.ptax.index(squash(ptGen_broadcast))
            iPTReco = self.ptax.index(squash(ptReco_broadcast))
            iBtagGen = squash(btagGen_broadcast)
            iBtagReco = squash(btagReco_broadcast)

            indices = (iBtagReco, iPTReco, iDRReco, iBtagGen, iPTGen, iDRGen)

            np.add.at(ans[order], indices, squash(vals))

        return ans

    def binProj(self, rEEC, rJet, mask, wt, subtract=None, noCov=False):
        t0 = time()

        nEVT = len(mask)
        nPT = self.ptax.extent
        nBtag = 2
        nProj = ak.flatten(rEEC.nproj)[0]
        nOrder = 5

        ans = np.zeros((nOrder, nBtag, nPT, nProj, nEVT), dtype=np.float64)
        
        t1 = time()

        pt = rJet.jets.corrpt[rEEC.iJet][mask[rEEC.iReco]]

        btag = ak.where(rJet.jets.passTightB[rEEC.iJet][mask[rEEC.iReco]], 1, 0, dtype=np.int32)

        t2 = time()

        for order in range(nOrder):
            proj = (rEEC.proj(order+2) * wt)[mask[rEEC.iReco]]
            if subtract is not None:
                proj = proj-(subtract.proj(order+2) * wt)[mask[rEEC.iReco]]

            iEVT = ak.local_index(proj, axis=0)

            iDR = ak.local_index(proj, axis=2)
            pt_broadcast, btag_broadcast, iEVT, iDR = ak.broadcast_arrays(pt, btag, iEVT, iDR)
        
            iDR = squash(iDR)
            iPT = self.ptax.index(squash(pt_broadcast))
            iBtag = squash(btag_broadcast)
            iEVT = squash(iEVT)

            indices = (iBtag, iPT, iDR, iEVT)
            vals = squash(proj)

            #proj = ak.sum(ak.sum(proj, axis=0), axis=0)
            #proj = ak.to_numpy(proj)
            np.add.at(ans[order],
                      indices,
                      vals)

        t3 = time()

        proj = np.sum(ans, axis=-1)

        t4 = time()

        if not noCov:
            leftis = [i+1 for i in range(len(ans.shape)-1)] + [0]
            rightis = [i+11 for i in range(len(ans.shape)-1)] + [0]
            ansis = leftis[:-1] + rightis[:-1]

            cov = np.einsum(ans, leftis, ans, rightis, ansis, optimize=True)

        t5 = time()

        print("proj timing summary:")
        print("\tinit:", t1-t0)
        print("\tpt:", t2-t1)
        print("\tloop:", t3-t2)
        print("\tsum:", t4-t3)
        print("\tcov:", t5-t4)

        if noCov:
            return proj
        else:
            return proj, cov


    def binAll(self, readers, mask, evtMask, wt):
        reco, covreco = self.binProj(readers.rRecoEEC, readers.rRecoJet, mask, wt)
        recopure = self.binProj(readers.rRecoEEC, readers.rRecoJet, mask, wt,
                                subtract=readers.rRecoEECUNMATCH, noCov=True)

        gen, covgen = self.binProj(readers.rGenEEC, readers.rGenJet, mask, wt)
        genpure = self.binProj(readers.rGenEEC, readers.rGenJet, mask, wt,
                               subtract=readers.rGenEECUNMATCH, noCov=True)

        transfer = self.binTransfer(readers.rTransfer,
                                    readers.rRecoEEC, readers.rRecoJet,
                                    readers.rGenEEC, readers.rGenJet,
                                    mask, wt)

        return {"reco": reco, "gen": gen,
                'covreco' : covreco, 'covgen' : covgen,
                'recopure': recopure, 'genpure': genpure,
                'transfer': transfer}
