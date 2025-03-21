import awkward as ak
import numpy as np
import hist
from .util import squash
from time import time

class EECgenericBinner:
    def __init__(self, config,
                 manualcov,
                 poissonbootstrap,
                 skipBtag,
                 statsplit,
                 sepPt):
        self.ptax = hist.axis.Variable(config.binning.pt)

        self.manualcov = manualcov
        self.poissonbootstrap = poissonbootstrap
        self.statsplit = statsplit
        self.sepPt = sepPt

        self.skipBtag = skipBtag
        if skipBtag:
            self.nBtag = 1
        else:
            self.nBtag = 2

        self.nPT = self.ptax.extent

        self.config = config

    def binTransferBTAGfactor(self, transfer,
                              rGenJet, rRecoJet,
                              iGen, iReco,
                              evtIdx, jetMask, wt):
        EECmask = jetMask[iReco]

        btag_gen = ak.values_astype(
                rGenJet.jets.passB[iGen][EECmask], np.int32)
        btag_reco = ak.values_astype(
                rRecoJet.jets.passB[iReco][EECmask], np.int32)

        wt_f, _ = ak.broadcast_arrays(wt, btag_gen)

        ans = np.zeros((self.statsplit, self.nBtag, self.nBtag), dtype=np.float64)

        N = self.statsplit
        for k in range(N):
            statmask = ak.to_numpy(evtIdx % N == k)

            np.add.at(ans[k],
                      (squash(btag_reco[statmask]),
                       squash(btag_gen[statmask])),
                      squash(wt_f[statmask]))

        return ans


    def binTransferPTfactor(self, transfer,
                            rGenJet, rRecoJet,
                            iGen, iReco,
                            evtIdx, jetMask, wt):
        EECmask = jetMask[iReco]
        
        pt_gen = rGenJet.jets.pt[iGen][EECmask]
        pt_reco = rRecoJet.jets.pt[iReco][EECmask]
        numpt = squash(ak.num(pt_reco))
        iPT_gen = self.ptax.index(squash(pt_gen)) + 1
        iPT_gen = ak.unflatten(iPT_gen, numpt)
        iPT_reco = self.ptax.index(squash(pt_reco)) + 1
        iPT_reco = ak.unflatten(iPT_reco, numpt)

        wt_f, _ = ak.broadcast_arrays(wt, iPT_gen)

        ans = np.zeros((self.statsplit, self.nPT, self.nPT), dtype=np.float64)

        N = self.statsplit
        for k in range(N):
            statmask = ak.to_numpy(evtIdx % N == k)

            np.add.at(ans[k],
                      (squash(iPT_reco[statmask]),
                       squash(iPT_gen[statmask])),
                      squash(wt_f[statmask]))
        return ans

    def binTransfer(self, transfer,
                    rGenJet, rRecoJet, 
                    iGen, iReco,
                    evtIdx, jetMask, wt):
        EECmask = jetMask[iReco]

        EECshape = ak.to_numpy(ak.flatten(transfer, axis=1)).shape[1:]
        
        vals = (transfer * wt)[EECmask]

        pt_gen = rGenJet.jets.pt[iGen][EECmask]
        pt_reco = rRecoJet.jets.pt[iReco][EECmask]
        numpt = squash(ak.num(pt_reco))
        iPT_gen = self.ptax.index(squash(pt_gen)) + 1
        iPT_gen = ak.unflatten(iPT_gen, numpt)
        iPT_reco = self.ptax.index(squash(pt_reco)) + 1
        iPT_reco = ak.unflatten(iPT_reco, numpt)

        btag_gen = ak.values_astype(
                rGenJet.jets.passB[iGen][EECmask], np.int32)
        btag_reco = ak.values_astype(
                rRecoJet.jets.passB[iReco][EECmask], np.int32)

        ptmode = self.config.transfermode.pt
        btagmode = self.config.transfermode.btag
        extrashape = []
        if ptmode == 'included':
            extrashape += [self.nPT, self.nPT]
        elif ptmode in ['factoredGen', 'factoredReco']:
            extrashape += [self.nPT]
        elif ptmode == 'ignored':
            pass
        else:
            raise ValueError("Unknown pt transfer mode")

        if btagmode == 'included':
            extrashape += [self.nBtag, self.nBtag]
        elif btagmode in ['factoredGen', 'factoredReco']:
            extrashape += [self.nBtag]
        elif btagmode == 'ignored':
            pass
        else:
            raise ValueError("Unknown btag transfer mode")

        ansshape = [self.statsplit, 
                    *extrashape,
                    *EECshape]
        ans = np.zeros(ansshape, dtype=np.float64)

        N = self.statsplit
        for k in range(N):
            statmask = ak.to_numpy(evtIdx % N == k)

            iPT_gen_f = ak.to_numpy(
                ak.flatten(
                    ak.flatten(
                        iPT_gen[statmask],
                    axis=1),
                axis=0)
            )
            iPT_reco_f = ak.to_numpy(
                ak.flatten(
                    ak.flatten(
                        iPT_reco[statmask],
                    axis=1),
                axis=0)
            )
            btag_gen_f = ak.to_numpy(
                ak.flatten(
                    ak.flatten(
                        btag_gen[statmask], 
                    axis=1),
                axis=0)
            )
            btag_reco_f = ak.to_numpy(
                ak.flatten(
                    ak.flatten(
                        btag_reco[statmask],
                    axis=1),
                axis=0)
            )
            vals_f = ak.to_numpy(
                ak.flatten(
                    ak.flatten(
                        vals[statmask], 
                    axis=1), 
                axis=0)
            )

            indices = []
            if ptmode == 'included':
                indices += [iPT_reco_f, iPT_gen_f]
            elif ptmode == 'factoredGen':
                indices += [iPT_gen_f]
            elif ptmode == 'factoredReco':
                indices += [iPT_reco_f]
            elif ptmode == 'ignored':
                pass

            if btagmode == 'included':
                indices += [btag_reco_f, btag_gen_f]
            elif btagmode == 'factoredGen':
                indices += [btag_gen_f]
            elif btagmode == 'factoredReco':
                indices += [btag_reco_f]
            elif btagmode == 'ignored':
                pass

            np.add.at(ans[k],
                      tuple(indices),
                      vals_f)
            #print(ans.sum())
            #print(np.sum(vals_f))

        #print("PTMODE", ptmode)
        #print("BTAGMODE", btagmode)
        #print("ANS SHAPE", ans.shape)

        return ans

    def indicesPerEvt(self, EECs, 
                      rJet, iJet, iReco,
                      evtIdx, jetMask, wt,
                      subtract=None):

        EECmask = jetMask[iReco]

        EECshape = ak.to_numpy(ak.flatten(EECs, axis=1)).shape[1:]

        if subtract is not None:
            vals = ((EECs-subtract)*wt)[EECmask]
        else:
            vals = (EECs * wt)[EECmask] 


        nEVT = len(wt)
        ansshape = [nEVT, self.nBtag, self.nPT, *EECshape]

        pt = rJet.jets.pt[iJet][EECmask]
        numpt = squash(ak.num(pt))
    
        iPT = self.ptax.index(squash(pt)) + 1
        iPT = ak.unflatten(iPT, numpt)

        if self.skipBtag:
            btag = np.zeros_like(iPT)
        else:
            btag = ak.values_astype(rJet.jets.passB[iJet][EECmask], 
                                    np.int32)

        iEVT = ak.local_index(vals, axis=0)

        iEVT, _ = ak.broadcast_arrays(iEVT, iPT)

        #now iPT, btag, iEVT all have shape (event, jet)
        #and the vals should have shape (event, jet, *EECshape)
        
        iPT_f = ak.to_numpy(ak.flatten(ak.flatten(iPT, axis=1), axis=0))
        btag_f = ak.to_numpy(ak.flatten(ak.flatten(btag, axis=1), axis=0))
        iEVT_f = ak.to_numpy(ak.flatten(ak.flatten(iEVT, axis=1), axis=0))
        vals_f = ak.to_numpy(ak.flatten(ak.flatten(vals, axis=1), axis=0))

        ans = np.zeros(ansshape, dtype=np.float64)

        np.add.at(ans, (iEVT_f, btag_f, iPT_f), vals_f)
        #print(ak.sum(vals_f))
        #print(np.sum(ans))
        #print(ans.shape)

        return ans

    def binObserved(self, EECs, 
                    rJet, iJet, iReco,
                    evtIdx, jetMask, wt,
                    subtract=None, noCov=False):
        t0 = time()

        binned = self.indicesPerEvt(EECs, rJet,
                                   iJet, iReco,
                                   evtIdx, jetMask, wt,
                                   subtract=subtract) 

        #print("indicesPerEvt took %g"%(time()-t0))

        binnedshape = binned.shape
        Nevt = binnedshape[0]

        t0 = time()
        if self.poissonbootstrap > 0:
            sedstr = rJet._x.behavior['__events_factory__']._partition_key
            import hashlib
            seed = int(hashlib.md5(sedstr.encode()).hexdigest(), 16)
            generator = np.random.default_rng(seed=seed)

            Npoi = self.poissonbootstrap

            poissonwts = generator.poisson(1, size=(Npoi, Nevt))
            ones = np.ones((1, Nevt), dtype=poissonwts.dtype)
            poissonwts = np.concatenate((ones, poissonwts), 
                                        axis=0)
            #print(poissonwts)
        else:
            poissonwts = np.ones((1, Nevt))
        #print("Poisson took %g"%(time()-t0))

        t0 = time()
        if self.statsplit > 1:
            N = self.statsplit
            statmask_l = []
            for k in range(N):
                statmask_l += [ak.to_numpy(evtIdx % N == k)[None,:]]

            statmask = np.concatenate(statmask_l, axis=0)
        else:
            statmask = np.ones((1,Nevt))
        #print("Statmask took %g"%(time()-t0))

        binned_indices = np.arange(len(binnedshape))
        observed_indices = [8, 9, *binned_indices[1:]] #sum over events
        poisson_indices = [9, 0]
        statsplit_indices = [8, 0]

        #print("FOR OBSERVED:")
        #print("binned shape", binned.shape)
        #print("binned indices", binned_indices)
        #print("poisson shape", poissonwts.shape)
        #print("poisson indices", poisson_indices)
        #print("stat shape", statmask.shape)
        #print("stat indics", statsplit_indices)
        #print("observed indices", observed_indices)

        t0 = time()
        observed = np.einsum(binned, binned_indices,
                             poissonwts, poisson_indices,
                             statmask, statsplit_indices,
                             observed_indices,
                             optimize=True)
        #print("Observed einsum took %g"%(time()-t0))
        #print("observed shape", observed.shape)
        #print()

        if not noCov and self.manualcov:
            #we ignore poissonbootstrap for cov
            leftidxs = binned_indices
            rightidxs = binned_indices + 10
            rightidxs[0] = 0 #diagonal in events dimension
 
            covidxs = np.concatenate(([8], #statsplit
                                       leftidxs[1:], #sum over events
                                       rightidxs[1:])) #sum over events

            #print("FOR COV:")
            #print("binned shape", binned.shape)
            #print("leftidxs", leftidxs)
            #print("rightixs", rightidxs)
            #print("statidxs", statsplit_indices)
            #print("covidxs", covidxs)

            #print(binned.shape)
            #print(leftidxs)
            #print(rightidxs)
            #print(statmask.shape)
            #print(statsplit_indices)
            #print(covidxs)
            t0 = time()

            cov = np.empty((self.statsplit, 
                            *binnedshape[1:],
                            *binnedshape[1:]),
                           dtype=np.float64)
            #print(cov.shape)
            #print(binned.shape)

            N = self.statsplit
            for k in range(N):
                statmask = ak.to_numpy(evtIdx % N == k)
                cov[k] = np.einsum(binned[statmask], leftidxs,
                                   binned[statmask], rightidxs,
                                   covidxs[1:],
                                   optimize=True)
    
            #cov = np.einsum(binned, leftidxs,
            #                binned, rightidxs,
            #                statmask, statsplit_indices,
            #                covidxs,
            #                optimize=True)

            #print("cov einsum took %g"%(time()-t0))
            #print("cov shape", cov.shape)

        if noCov or not self.manualcov:
            return observed
        else:
            return observed, cov
