import awkward as ak
import pandas as pd
import numpy as np
import hist
from .util import squash
from time import time
from DataFrameAccumulator import DataFrameAccumulator as DFA

class EECgenericBinner:
    def __init__(self, 
                 config,
                 poissonbootstrap,
                 skipBtag,
                 statsplit,
                 sepPt,
                 baseRNG = 0,
                 **kwargs):
        self.ptax = hist.axis.Variable(config.binning.pt)

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
    
        self.baseRNG = baseRNG

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

        pt_gen = rGenJet.jets.corrpt[iGen][EECmask]
        pt_reco = rRecoJet.jets.corrpt[iReco][EECmask]

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
                    EECvals, 
                    ptDenom, 
                    order,
                    rJet, 
                    iJet, iReco,
                    evtIdx, 
                    jetMask, wt,
                    subtract=None):
        t0 = time()

        EECmask = jetMask[iReco]

        vals = EECvals[EECmask]
        denom = ptDenom[EECmask]

        pt = rJet.jets.corrpt[iJet][EECmask]

        correction = np.power(denom/pt, order)

        vals['wt'] = vals['wt'] * correction

        pt_b, _ = ak.broadcast_arrays(pt, vals.R)

        if not self.skipBtag:
            btag = ak.values_astype(
                rJet.jets.passB[iJet][EECmask], np.int32
            )
        else:
            btag = np.zeros_like(pt_b, dtype=np.int32)

        wt_b, _ = ak.broadcast_arrays(wt, vals.R)

        fillvals = {
            'R' : squash(vals.R),
            'r' : squash(vals.r),
            'c' : squash(vals.c),
            'wt' : squash(vals.wt),
            'pt' : squash(pt_b),
            'evtwt' : squash(wt_b)
        }

        if self.poissonbootstrap > 0:
            run = rJet._x.run
            event = rJet._x.event
            lumi = rJet._x.luminosityBlock

            #use some arbitrary large primes to "hash" the (run, lumi, event) into a unique-ish number
            #the actual requirement is that subsequent events must have different codes
            #and that a given (run, lumi, event) always has the same code
            #this satisfies that requirement with very high probability
            eventhash = event + lumi*1299827 + run*2038074743 

            eventhash_b, _ = ak.broadcast_arrays(eventhash, vals.R)
            fillvals['eventhash'] = squash(eventhash_b)

        df = DFA(fillvals)

        return df
