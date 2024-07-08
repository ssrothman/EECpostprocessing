import numpy as np
import awkward as ak
import hist
from binning.util import *
from time import time

def proj_reduce(ans, poissonwts, evtmask):
    if evtmask is None:
        if poissonwts is not None:
            return np.einsum('abcde,ef->abcdf', 
                             ans, 
                             poissonwts, 
                             optimize=True)
        else:
            return np.sum(ans, axis=-1)
    else:
        if poissonwts is not None:
            return np.einsum('abcde,ef,e->abcdf', 
                             ans, 
                             poissonwts,
                             evtmask,
                             optimize=True)
        else:
            return np.einsum('abcde,e->abcd',
                             ans,
                             evtmask,
                             optimize=True)

class EECprojBinner:
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit,
                 sepPt):
        self.ptax = hist.axis.Variable(config.binning.bins.pt)

        self.manualcov = manualcov
        self.poissonbootstrap = poissonbootstrap
        self.statsplit = statsplit
        self.sepPt = sepPt

    def binTransfer_full(self, rTransfer,
                    rRecoEEC, rRecoJet,
                    rGenEEC, rGenJet,
                    eventIdx,
                    m, wt):
        nPT = self.ptax.extent
        nProj = ak.flatten(rRecoEEC.nproj)[0]
        nOrder = 5
        nBtag = 2

        if self.statsplit > 0:
            N = self.statsplit
            ans = np.zeros((N, nOrder, nBtag, nPT, nProj, nBtag, nPT, nProj), dtype=np.float64)
            evt = ak.to_numpy(evtIdx)
            for k in range(N):
                statmask = (evt % N == k)
                self.actuallyBinTransfer_full(ans[k], rTransfer,
                                        rRecoEEC, rRecoJet,
                                        rGenEEC, rGenJet,
                                        m & statmask, wt)
        else:
            ans = np.zeros((nOrder, nBtag, nPT, nProj, nBtag, nPT, nProj), dtype=np.float64)
            self.actuallyBinTransfer_full(ans, rTransfer,
                                    rRecoEEC, rRecoJet,
                                    rGenEEC, rGenJet,
                                    m, wt)

        return ans

    def binTransfer_sepPt(self, rTransfer,
                          rRecoEEC, rRecoJet,
                          rGenEEC, rGenJet,
                          evtIdx,
                          m, wt):
        nPT = self.ptax.extent
        nProj = ak.flatten(rRecoEEC.nproj)[0]
        nOrder = 5
        nBtag = 2

        if self.statsplit > 0:
            N = self.statsplit
            ansPT = np.zeros((N, nPT, nPT))
            ansRest = np.zeros((N, nOrder, nPT, nBtag, nProj, nBtag, nProj), dtype=np.float64)
            evt = ak.to_numpy(evtIdx)
            for k in range(N):
                statmask = (evt % N == k)
                self.actuallyBinTransfer_sepPt(ansPT[k], ansRest[k],
                                              rTransfer,
                                              rRecoEEC, rRecoJet,
                                              rGenEEC, rGenJet,
                                              m & statmask, wt)
        else:
            ansPT = np.zeros((nPT, nPT))
            ansRest = np.zeros((nOrder, nPT, nBtag, nProj, nBtag, nProj), dtype=np.float64)
            self.actuallyBinTransfer_sepPt(ansPT, ansRest,
                                        rTransfer,
                                        rRecoEEC, rRecoJet,
                                        rGenEEC, rGenJet,
                                        m, wt)

        return ansPT, ansRest

    def actuallyBinTransfer_full(self, ans,
                            rTransfer,
                            rRecoEEC, rRecoJet, 
                            rGenEEC, rGenJet, 
                            m, wt):
        nOrder = 5

        iGenJet = rTransfer.iGen
        iRecoJet = rTransfer.iReco

        mask = m[iRecoJet]

        ptReco = rRecoJet.jets.pt[iRecoJet][mask]
        ptGen = rGenJet.jets.pt[iGenJet][mask]

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
            iPTGen = self.ptax.index(squash(ptGen_broadcast)) + 1
            iPTReco = self.ptax.index(squash(ptReco_broadcast)) + 1
            iBtagGen = squash(btagGen_broadcast)
            iBtagReco = squash(btagReco_broadcast)

            indices = (iBtagReco, iPTReco, iDRReco, iBtagGen, iPTGen, iDRGen)

            np.add.at(ans[order], indices, squash(vals))

    def actuallyBinTransfer_sepPt(self, ansPT, ansRest,
                                  rTransfer,
                                  rRecoEEC, rRecoJet,
                                  rGenEEC, rGenJet,
                                  m, wt):
        nOrder = 5

        iGenJet = rTransfer.iGen
        iRecoJet = rTransfer.iReco

        mask = m[iRecoJet]

        ptReco = rRecoJet.jets.pt[iRecoJet][mask]
        ptGen = rGenJet.jets.pt[iGenJet][mask]

        btagReco = ak.where(rRecoJet.jets.passTightB[iRecoJet][mask], 1, 0, dtype=np.int32)
        btagGen = ak.where(rRecoJet.jets.hadronFlavour[iRecoJet][mask] == 5, 1, 0, dtype=np.int32)


        iPTGen_flat = self.ptax.index(squash(ptGen)) + 1
        iPTReco_flat = self.ptax.index(squash(ptReco)) + 1
        wtflat,_ = ak.broadcast_arrays(wt, ptGen)
        wtflat = squash(wtflat)

        indicesPT = (iPTReco_flat, iPTGen_flat)
        np.add.at(ansPT, indicesPT, wtflat)

        ptGen_flat = squash(ptGen)
        ptReco_flat = squash(ptReco)
        ptGenReco = np.stack([ptReco_flat, ptGen_flat], axis=1)

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
            iPTGen = self.ptax.index(squash(ptGen_broadcast)) + 1
            iPTReco = self.ptax.index(squash(ptReco_broadcast)) + 1
            iBtagGen = squash(btagGen_broadcast)
            iBtagReco = squash(btagReco_broadcast)

            indicesRest = (iPTGen, iBtagReco, iDRReco, iBtagGen, iDRGen)

            np.add.at(ansRest[order], indicesRest, squash(vals))

    def binProj(self, rEEC, rJet, evtIdx,
                mask, wt, 
                subtract=None, noCov=False):
        t0 = time()

        nEVT = len(mask)
        nPT = self.ptax.extent
        nBtag = 2
        nProj = ak.flatten(rEEC.nproj)[0]
        nOrder = 5

        if self.poissonbootstrap > 0:
            poissonwts = np.random.poisson(1, size=(nEVT, self.poissonbootstrap))
            ones = np.ones((nEVT, 1), dtype=poissonwts.dtype)
            poissonwts = np.concatenate([ones, poissonwts], axis=1)
        else:
            poissonwts = None

        ansshape = [nOrder, nBtag, nPT, nProj, nEVT]
        ans = np.zeros(ansshape, dtype=np.float64)
        
        t1 = time()

        pt = rJet.jets.pt[rEEC.iJet][mask[rEEC.iReco]]

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
            iPT = self.ptax.index(squash(pt_broadcast)) + 1
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

        if self.statsplit > 0:
            if poissonwts is not None:
                proj = np.zeros((self.statsplit, 
                                 nOrder, nBtag, nPT, nProj, 
                                 self.poissonbootstrap+1))
            else:
                proj = np.zeros((self.statsplit, 
                                 nOrder, nBtag, nPT, nProj))

            N = self.statsplit
            evt = ak.to_numpy(evtIdx)
            for k in range(N):
                evtmask = (evt % N == k)
                proj[k] = proj_reduce(ans, poissonwts, evtmask)
        else:
            proj = proj_reduce(ans, poissonwts, None)

        t4 = time()

        if (not noCov) and self.manualcov:
            leftis = [i+1 for i in range(len(ans.shape)-1)] + [0]
            rightis = [i+11 for i in range(len(ans.shape)-1)] + [0]
            ansis = leftis[:-1] + rightis[:-1]

            if self.statsplit > 0:
                N = self.statsplit
                evt = ak.to_numpy(evtIdx)

                cov = np.zeros((N, 
                                nOrder, nBtag, nPT, nProj,
                                nOrder, nBtag, nPT, nProj))
                for k in range(N):
                    evtmask = (evt % N == k)
                    cov[k] = np.einsum(ans, leftis, 
                                       ans, rightis, 
                                       evtmask, [0],
                                       ansis,
                                       optimize=True)

            else:
                cov = np.einsum(ans, leftis, 
                                ans, rightis, 
                                ansis,
                                optimize=True)



        t5 = time()

        #print("proj timing summary:")
        #print("\tinit:", t1-t0)
        #print("\tpt:", t2-t1)
        #print("\tloop:", t3-t2)
        #print("\tsum:", t4-t3)
        #print("\tcov:", t5-t4)

        if noCov or (not self.manualcov):
            return proj
        else:
            return proj, cov


    def binAll(self, readers, mask, evtMask, wt):
        reco = self.binProj(readers.rRecoEEC, readers.rRecoJet, 
                            readers.eventIdx, mask, wt)
        recopure = self.binProj(readers.rRecoEEC, readers.rRecoJet, 
                                readers.eventIdx, mask, wt,
                                subtract=readers.rRecoEECUNMATCH, noCov=True)

        gen = self.binProj(readers.rGenEEC, readers.rGenJet, 
                           readers.eventIdx, mask, wt)
        genpure = self.binProj(readers.rGenEEC, readers.rGenJet,
                               readers.eventIdx, mask, wt,
                               subtract=readers.rGenEECUNMATCH, noCov=True)

        if not self.sepPt:
            transfer = self.binTransfer_full(readers.rTransfer,
                                        readers.rRecoEEC, readers.rRecoJet,
                                        readers.rGenEEC, readers.rGenJet,
                                        readers.eventIdx, mask, wt)
        else:
            transferPT, transferRest = self.binTransfer_sepPt(
                    readers.rTransfer,
                    readers.rRecoEEC, readers.rRecoJet,
                    readers.rGenEEC, readers.rGenJet,
                    readers.eventIdx, mask, wt)

        if self.manualcov:
            if self.sepPt:
                return {"reco": reco[0], "gen": gen[0],
                        'covreco' : reco[1], 'covgen' : gen[1],
                        'recopure': recopure, 'genpure': genpure,
                        'transferPT': transferPT, 'transferRest': transferRest}
            else:
                return {"reco": reco[0], "gen": gen[0],
                        'covreco' : reco[1], 'covgen' : gen[1],
                        'recopure': recopure, 'genpure': genpure,
                        'transfer': transfer}
        else:
            if self.sepPt:
                return {"reco": reco, "gen": gen,
                        'recopure': recopure, 'genpure': genpure,
                        'transferPT': transferPT, 'transferRest': transferRest}
            else:
                return {"reco": reco, "gen": gen,
                        'recopure': recopure, 'genpure': genpure,
                        'transfer': transfer}
