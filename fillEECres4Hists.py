import numpy as np
from time import time
import hist
import awkward as ak

import pyarrow.dataset as ds

from tqdm import tqdm

class sm64_rng:
    def __init__(self, seedarr, repeats):
        self.repeats = repeats
        cumsum = np.cumsum(self.repeats) - 1
        self.state = seedarr[cumsum] # only unique values

    def next_uint64(self):
        self.state += 0x9e3779b97f4a7c15;
        z = self.state
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31))
        return z

    def next_uint64_masked(self, mask):
        self.state[mask] += 0x9e3779b97f4a7c15
        z = self.state[mask]
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        z = (z ^ (z >> 31))
        return z

    def next_float(self):
        return self.next_uint64().astype(np.float64) / (1<<64)

    def next_float_masked(self, mask):
        return self.next_uint64_masked(mask) / (1<<64)

    def next_poisson(self, lam):
        # Poisson RNG using the rejection method
        if lam < 0:
            raise ValueError("Lambda must be non-negative")
        elif lam == 0:
            return 0

        L = np.exp(-lam)
        p = self.next_float()
        k = np.ones_like(p, dtype=np.int64)
        while True:
            mask = p > L
            if (not np.any(mask)):
                break
            p[mask] = p[mask] * self.next_float_masked(mask)
            k[mask] += 1
        return np.repeat(k - 1, self.repeats) #unpack the run lengths

def fill_hist_from_parquet(basepath, bootstrap, random_seed=0):

    dataset = ds.dataset(basepath, format="parquet")

    # discover value ranges
    maxR = None
    maxr = None
    maxc = None
    minR = None
    minr = None
    minc = None

    for batch in tqdm(dataset.to_batches()):
        maxR = max(maxR, np.max(batch['R'])) if maxR is not None else np.max(batch['R'])
        maxr = max(maxr, np.max(batch['r'])) if maxr is not None else np.max(batch['r'])
        maxc = max(maxc, np.max(batch['c'])) if maxc is not None else np.max(batch['c'])
        minR = min(minR, np.min(batch['R'])) if minR is not None else np.min(batch['R'])
        minr = min(minr, np.min(batch['r'])) if minr is not None else np.min(batch['r'])
        minc = min(minc, np.min(batch['c'])) if minc is not None else np.min(batch['c'])

    H = hist.Hist(
        hist.axis.Integer(minR, maxR+1,
                          name="R", label = '$R$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr, maxr+1,
                          name="r", label = '$r$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc, maxc+1,
                          name="c", label = '$c$',
                          underflow=False, overflow=False),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt', label='$p_{T}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Integer(0, bootstrap+1,
                          name='bootstrap', label='bootstrap',
                          underflow=False, overflow=False),
        storage=hist.storage.Double()
    )

    for batch in tqdm(dataset.to_batches(batch_size=1000000)):
        R = batch['R'].to_numpy(zero_copy_only=True)
        r = batch['r'].to_numpy(zero_copy_only=True)
        c = batch['c'].to_numpy(zero_copy_only=True)
        pt = batch['pt'].to_numpy(zero_copy_only=True)
        evtwt = batch['evtwt'].to_numpy(zero_copy_only=True)
        wt = batch['wt'].to_numpy(zero_copy_only=True)
        evthash = batch['eventhash'].to_numpy(zero_copy_only=True)

        H.fill(
            R=R,
            r=r,
            c=c,
            pt=pt,
            bootstrap = 0,
            weight=evtwt*wt,
        )

        repeats = ak.to_numpy(ak.run_lengths(evthash))
        rng = sm64_rng(evthash + random_seed, repeats)

        for i in range(bootstrap):
            boot = rng.next_poisson(1.0)
            H.fill(
                R=R,
                r=r,
                c=c,
                pt=pt,
                bootstrap = i+1,
                weight=evtwt*wt*boot,
            )

    return H

def fill_transferhist_from_parquet(basepath):

    dataset = ds.dataset(basepath, format="parquet")

    # discover value ranges
    maxR_reco = None
    maxR_gen = None
    maxr_reco = None
    maxr_gen = None
    maxc_reco = None
    maxc_gen = None
    minR_reco = None
    minR_gen = None
    minr_reco = None
    minr_gen = None
    minc_reco = None
    minc_gen = None

    for batch in tqdm(dataset.to_batches()):
        maxR_reco = max(maxR_reco, np.max(batch['R_reco'])) if maxR_reco is not None else np.max(batch['R_reco'])
        maxR_gen = max(maxR_gen, np.max(batch['R_gen'])) if maxR_gen is not None else np.max(batch['R_gen'])
        maxr_reco = max(maxr_reco, np.max(batch['r_reco'])) if maxr_reco is not None else np.max(batch['r_reco'])
        maxr_gen = max(maxr_gen, np.max(batch['r_gen'])) if maxr_gen is not None else np.max(batch['r_gen'])
        maxc_reco = max(maxc_reco, np.max(batch['c_reco'])) if maxc_reco is not None else np.max(batch['c_reco'])
        maxc_gen = max(maxc_gen, np.max(batch['c_gen'])) if maxc_gen is not None else np.max(batch['c_gen'])
        minR_reco = min(minR_reco, np.min(batch['R_reco'])) if minR_reco is not None else np.min(batch['R_reco'])
        minR_gen = min(minR_gen, np.min(batch['R_gen'])) if minR_gen is not None else np.min(batch['R_gen'])
        minr_reco = min(minr_reco, np.min(batch['r_reco'])) if minr_reco is not None else np.min(batch['r_reco'])
        minr_gen = min(minr_gen, np.min(batch['r_gen'])) if minr_gen is not None else np.min(batch['r_gen'])
        minc_reco = min(minc_reco, np.min(batch['c_reco'])) if minc_reco is not None else np.min(batch['c_reco'])
        minc_gen = min(minc_gen, np.min(batch['c_gen'])) if minc_gen is not None else np.min(batch['c_gen'])

    H = hist.Hist(
        hist.axis.Integer(minR_reco, maxR_reco+1,
                          name="R_reco", label = '$R_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minR_gen, maxR_gen+1,
                          name="R_gen", label = '$R_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr_reco, maxr_reco+1,
                          name="r_reco", label = '$r_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr_gen, maxr_gen+1,
                          name="r_gen", label = '$r_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc_reco, maxc_reco+1,
                          name="c_reco", label = '$c_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc_gen, maxc_gen+1,
                          name="c_gen", label = '$c_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt_reco', label='$p_{T,reco}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt_gen', label='$p_{T,gen}$ [GeV]',
                           underflow=True, overflow=True),
        storage=hist.storage.Double()
    )

    for batch in tqdm(dataset.to_batches()):
        H.fill(
            R_reco=batch['R_reco'].to_numpy(),
            R_gen=batch['R_gen'].to_numpy(),
            r_reco=batch['r_reco'].to_numpy(),
            r_gen=batch['r_gen'].to_numpy(),
            c_reco=batch['c_reco'].to_numpy(),
            c_gen=batch['c_gen'].to_numpy(),
            pt_reco=batch['pt_reco'].to_numpy(),
            pt_gen=batch['pt_gen'].to_numpy(),
            weight=batch['evtwt'].to_numpy()*batch['wt_reco'].to_numpy(),
        )

    return H
