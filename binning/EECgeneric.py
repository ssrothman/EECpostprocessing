import awkward as ak
import pandas as pd
import numpy as np
import hist
from .util import squash
from time import time
from DataFrameAccumulator import DataFrameAccumulator as DFA
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os.path

class EECgenericBinner:
    def __init__(self, 
                 config,
                 **kwargs):
        self.ptax = hist.axis.Variable(config.binning.pt)

        self.skipBtag = config.skipBtag

        self.nPT = self.ptax.extent

        self.config = config

    def binTransfer(self, 
                    transfervals,
                    transfershape,
                    ptDenomReco, ptDenomGen,
                    order,
                    rGenJet, rRecoJet, 
                    iGen, iReco,
                    evtIdx, jetMask, 
                    wtVars,
                    outpath):

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


        fillvals = {
            'R_reco':   squash(vals.R_reco),
            'r_reco':   squash(vals.r_reco),
            'c_reco':   squash(vals.c_reco),
            'R_gen':    squash(vals.R_gen),
            'r_gen':    squash(vals.r_gen),
            'c_gen':    squash(vals.c_gen),
            'wt_gen':   squash(vals.wt_gen),
            'wt_reco':  squash(vals.wt_reco),
        }

        for variation in wtVars:
            thewt = wtVars[variation]
            thwt_b, _ = ak.broadcast_arrays(thewt, vals.R_reco)
            fillvals[variation] = squash(thwt_b)

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

        table = pa.Table.from_pandas(pd.DataFrame(fillvals),
                                     preserve_index=False)
         
        filekey = rGenJet._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        destination = os.path.join(outpath, filekey + '.parquet')
        os.makedirs(outpath, exist_ok=True)
        pq.write_table(table, destination)

        return ak.sum(vals.wt_reco)
        #return df

    def binObserved(self, 
                    EECvals, 
                    ptDenom, 
                    order,
                    rJet, 
                    iJet, iReco,
                    evtIdx, 
                    jetMask, 
                    wtVars,
                    outpath):

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


        fillvals = {
            'R' : squash(vals.R),
            'r' : squash(vals.r),
            'c' : squash(vals.c),
            'wt' : squash(vals.wt),
            'pt' : squash(pt_b),
        }

        for variation in wtVars:
            thewt = wtVars[variation]
            thewt_b, _ = ak.broadcast_arrays(thewt, vals.R)
            fillvals[variation] = squash(thewt_b)

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

        table = pa.Table.from_pandas(pd.DataFrame(fillvals),
                                     preserve_index=False)
         
        filekey = rJet._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        os.makedirs(outpath, exist_ok=True)
        destination = os.path.join(outpath, filekey + '.parquet')
        pq.write_table(table, destination)

        return ak.sum(vals.wt)
