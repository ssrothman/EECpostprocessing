import awkward as ak
import pandas as pd
import numpy as np
import hist
from .util import squash
from time import time
from DataFrameAccumulator import DataFrameAccumulator as DFA

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

    def binTransfer(self, 
                    transfervals,
                    transfershape,
                    ptDenomReco, ptDenomGen,
                    order,
                    rGenJet, rRecoJet, 
                    iGen, iReco,
                    evtIdx, jetMask, wt):

        EECmask = jetMask[iReco]

        vals = transfervals[EECmask]
        DenomReco = ptDenomReco[EECmask]
        DenomGen = ptDenomGen[EECmask]

        pt_gen = rGenJet.jets.pt[iGen][EECmask]
        pt_reco = rRecoJet.jets.pt[iReco][EECmask]

        print(squash(pt_gen))
        print(squash(pt_reco))

        correction_gen = np.power(DenomGen/pt_gen, order)
        correction_reco = np.power(DenomReco/pt_reco, order)

        vals['wt_gen'] = vals['wt_gen'] * correction_gen
        vals['wt_reco'] = vals['wt_reco'] * correction_reco

        pt_reco_b, _ = ak.broadcast_arrays(pt_reco, vals.R_reco)
        pt_gen_b, _ = ak.broadcast_arrays(pt_gen, vals.R_gen)

        ptmode = self.config.transfermode.pt
        btagmode = self.config.transfermode.btag

        if btagmode != 'ignored':
            if not self.skipBtag:
                btag_gen = ak.values_astype(
                        rGenJet.jets.passB[iGen][EECmask], np.int32)
                btag_reco = ak.values_astype(
                        rRecoJet.jets.passB[iReco][EECmask], np.int32)
            else:
                btag_gen = np.zeros_like(iPT_gen)
                btag_reco = np.zeros_like(iPT_reco)

            btag_reco_b, _ = ak.broadcast_arrays(btag_reco, vals.R_reco)
            btag_gen_b, _ = ak.broadcast_arrays(btag_gen, vals.R_gen)

        #axes = [
        #    hist.axis.Integer(0, transfershape['R_reco'],
        #                      name='R_reco', label='$R_{reco}$',
        #                      overflow=False, underflow=False),
        #    hist.axis.Integer(0, transfershape['r_reco'],
        #                      name='r_reco', label='$r_{reco}$',
        #                      overflow=False, underflow=False),
        #    hist.axis.Integer(0, transfershape['c_reco'],
        #                      name='c_reco', label='$c_{reco}$',
        #                      overflow=False, underflow=False),
        #    hist.axis.Integer(0, transfershape['R_gen'],
        #                      name='R_gen', label='$R_{gen}$',
        #                      overflow=False, underflow=False),
        #    hist.axis.Integer(0, transfershape['r_gen'],
        #                      name='r_gen', label='$r_{gen}$',
        #                      overflow=False, underflow=False),
        #    hist.axis.Integer(0, transfershape['c_gen'],
        #                      name='c_gen', label='$c_{gen}$',
        #                      overflow=False, underflow=False),
        #    #hist.axis.Regular(20, 1e-11, 1e-3, 
        #    #                  name='wt_gen', label='$wt_{gen}$',
        #    #                  overflow=True, underflow=True,
        #    #                  transform=hist.axis.transform.log),
        #    #hist.axis.Regular(20, 1e-11, 1e-3, 
        #    #                  name='wt_reco', label='$wt_{reco}$',
        #    #                  overflow=True, underflow=True,
        #    #                  transform=hist.axis.transform.log),
        #]
        #if ptmode == 'included':
        #    axes += [
        #        hist.axis.Variable(self.config.binning.pt,
        #                           name='pt_reco', label='$p_{T,reco}$',
        #                           overflow=True, underflow=True),
        #        hist.axis.Variable(self.config.binning.pt,
        #                           name='pt_gen', label='$p_{T,gen}$',
        #                           overflow=True, underflow=True),
        #    ]
        #elif ptmode == 'factoredGen':
        #    axes += [
        #        hist.axis.Variable(self.config.binning.pt,
        #                           name='pt_gen', label='$p_{T,gen}$',
        #                           overflow=True, underflow=True),
        #    ]
        #elif ptmode == 'factoredReco':
        #    axes += [
        #        hist.axis.Variable(self.config.binning.pt,
        #                           name='pt_reco', label='$p_{T,reco}$',
        #                           overflow=True, underflow=True),
        #    ]
        #elif ptmode == 'ignored':
        #    pass

        #if btagmode == 'included':
        #    axes += [
        #        hist.axis.Integer(0, 2, name='btag_reco', label='btag_reco',
        #                          overflow=False, underflow=False),
        #        hist.axis.Integer(0, 2, name='btag_gen', label='btag_gen',
        #                          overflow=False, underflow=False),
        #    ]
        #elif btagmode == 'factoredGen':
        #    axes += [
        #        hist.axis.Integer(0, 2, name='btag_gen', label='btag_gen',
        #                          overflow=False, underflow=False),
        #    ]
        #elif btagmode == 'factoredReco':
        #    axes += [
        #        hist.axis.Integer(0, 2, name='btag_reco', label='btag_reco',
        #                          overflow=False, underflow=False),
        #    ]
        #elif btagmode == 'ignored':
        #    pass

        #ans = hist.Hist(
        #    *axes,
        #    storage=hist.storage.Double(),
        #)

        wt_b, _ = ak.broadcast_arrays(wt, vals.R_reco)

        fillvals = {
            'R_reco':   squash(vals.R_reco),
            'r_reco':   squash(vals.r_reco),
            'c_reco':   squash(vals.c_reco),
            'R_gen':    squash(vals.R_gen),
            'r_gen':    squash(vals.r_gen),
            'c_gen':    squash(vals.c_gen),
            'wt_gen':   squash(vals.wt_gen),
            'wt_reco':  squash(vals.wt_reco),
            'evtwt' :   squash(wt_b),
        }


        if ptmode == 'included':
            fillvals['pt_reco'] = squash(pt_reco_b)
            fillvals['pt_gen'] = squash(pt_gen_b)
        elif ptmode == 'factoredGen':
            fillvals['pt_gen'] = squash(pt_gen_b)
        elif ptmode == 'factoredReco':
            fillvals['pt_reco'] = squash(pt_reco_b)
        elif ptmode == 'ignored':
            pass

        if btagmode == 'included':
            fillvals['btag_reco'] = squash(btag_reco_b)
            fillvals['btag_gen'] = squash(btag_gen_b)
        elif btagmode == 'factoredGen':
            fillvals['btag_gen'] = squash(btag_gen_b)
        elif btagmode == 'factoredReco':
            fillvals['btag_reco'] = squash(btag_reco_b)
        elif btagmode == 'ignored':
            pass


        #ans.fill(
        #    **fillvals,
        #    weight=squash(wt_b*vals.wt_gen)
        #)

        df = DFA(fillvals)

        return df

    def indicesPerEvt(self, EECs, 
                      ptDenom, order,
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

        correction = np.power(ptDenom[EECmask]/pt, order)
        vals = vals * correction

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

    def binObserved(self, 
                    EECs, 
                    ptDenom, order,
                    rJet, 
                    iJet, iReco,
                    evtIdx, 
                    jetMask, wt,
                    subtract=None, noCov=False):
        t0 = time()

        binned = self.indicesPerEvt(EECs, 
                                    ptDenom, order,
                                    rJet,
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
