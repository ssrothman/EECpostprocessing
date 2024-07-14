import awkward as ak
import numpy as np
import hist
from .util import squash

class EECgenericBinner:
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit,
                 sepPt):
        self.ptax = hist.axis.Variable(config.binning.bins.pt)

        self.manualcov = manualcov
        self.poissonbootstrap = poissonbootstrap
        self.statsplit = statsplit
        self.sepPt = sepPt

        self.nBtag = 2
        self.nPT = self.ptax.extent

        self.config = config

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

        print("Btagging level is", self.config.tagging.wp)
        print("the wp is", self.config.tagging.bwps.tight)
        print("Max btag score [CHS] is", ak.max(rRecoJet.CHSjets.btagDeepB[iReco][EECmask], axis=None))
        print("Max btag int [CHS] is", ak.max(rRecoJet.CHSjets.passTightB[iReco][EECmask], axis=None))
        print("Max btag int [ak8] is", ak.max(rRecoJet.jets.passTightB[iReco][EECmask], axis=None))

        if self.config.tagging.wp == 'tight':
            btag_gen = ak.values_astype(
                    rGenJet.jets.passTightB[iGen][EECmask], np.int32)
            btag_reco = ak.values_astype(
                    rRecoJet.jets.passTightB[iReco][EECmask], np.int32)
        elif self.config.tagging.wp == 'medium':
            btag_gen = ak.values_astype(
                    rGenJet.jets.passMediumB[iGen][EECmask], np.int32)
            btag_reco = ak.values_astype(
                    rRecoJet.jets.passMediumB[iReco][EECmask], np.int32)
        elif self.config.tagging.wp == 'loose':
            btag_gen = ak.values_astype(
                    rGenJet.jets.passLooseB[iGen][EECmask], np.int32)
            btag_reco = ak.values_astype(
                    rRecoJet.jets.passLooseB[iReco][EECmask], np.int32)
        else:
            raise ValueError("Unknown tagging WP")

        iPT_gen_f = ak.to_numpy(ak.flatten(ak.flatten(iPT_gen, axis=1), axis=0))
        iPT_reco_f = ak.to_numpy(ak.flatten(ak.flatten(iPT_reco, axis=1), axis=0))
        btag_gen_f = ak.to_numpy(ak.flatten(ak.flatten(btag_gen, axis=1), axis=0))
        btag_reco_f = ak.to_numpy(ak.flatten(ak.flatten(btag_reco, axis=1), axis=0))
        vals_f = ak.to_numpy(ak.flatten(ak.flatten(vals, axis=1), axis=0))

        ansshape = [self.nBtag, self.nPT, self.nBtag, self.nPT, *EECshape]
        ans = np.zeros(ansshape, dtype=np.float64)

        #print(ans.shape)
        np.add.at(ans, 
                  (btag_reco_f, iPT_reco_f, btag_gen_f, iPT_gen_f), 
                  vals_f)
        #print(ans.sum())
        #print(np.sum(vals_f))

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

        if self.config.tagging.wp == 'tight':
            btag = ak.values_astype(rJet.jets.passTightB[iJet][EECmask], 
                                    np.int32)
        elif self.config.tagging.wp == 'medium':
            btag = ak.values_astype(rJet.jets.passMediumB[iJet][EECmask], 
                                    np.int32)
        elif self.config.tagging.wp == 'loose':
            btag = ak.values_astype(rJet.jets.passLooseB[iJet][EECmask], 
                                    np.int32)
        else:
            raise ValueError("Unknown tagging WP")

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

        binned = self.indicesPerEvt(EECs, rJet,
                                   iJet, iReco,
                                   evtIdx, jetMask, wt,
                                   subtract=subtract) 

        binnedshape = binned.shape
        Nevt = binnedshape[0]

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

        if self.statsplit > 1:
            N = self.statsplit
            statmask_l = []
            for k in range(N):
                statmask_l += [ak.to_numpy(evtIdx % N == k)[None,:]]

            statmask = np.concatenate(statmask_l, axis=0)
        else:
            statmask = np.ones((1,Nevt))

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

        observed = np.einsum(binned, binned_indices,
                             poissonwts, poisson_indices,
                             statmask, statsplit_indices,
                             observed_indices,
                             optimize=True)
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

            cov = np.einsum(binned, leftidxs,
                            binned, rightidxs,
                            statmask, statsplit_indices,
                            covidxs,
                            optimize=True)
            #print("cov shape", cov.shape)

        if noCov or not self.manualcov:
            return observed
        else:
            return observed, cov
