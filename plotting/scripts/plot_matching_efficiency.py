#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

from unfolding.detectormodel import DetectorModel

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='/eos/user/d/dponman/proj_unfold_workspace')
parser.add_argument('--output',    type=str, default='plots/proj')
args = parser.parse_args()

JPT_BINS = [
    (40,   100),
    (100,  200),
    (200,  340),
    (340,  520),
    (520,  740),
    (740,  1000),
    (1000, 1500),
    (1500, 2500),
    (2500, 5000),
]

valid = np.load(os.path.join(args.workspace, 'valid_bins.npy'))
model = DetectorModel.from_disk(os.path.join(args.workspace, 'detectormodel'))

# gamma0 = unmatchedGen / totalGen, shape (nGen,)
# apply valid mask to recover per-bin values
gamma0 = model._gamma0  # already masked to valid bins in workspace

n_first = int(valid.sum()) - 8 * 50
r_start = 50 - n_first

# sum over R bins per Jpt bin to get integrated matching efficiency
jpt_centers = [np.sqrt(lo * hi) for lo, hi in JPT_BINS]
efficiencies = []

idx = 0
for i in range(len(JPT_BINS)):
    if i == 0:
        sl = slice(idx, idx + n_first)
        idx += n_first
    else:
        sl = slice(idx + r_start, idx + 50)
        idx += 50

    gamma_slice = gamma0[sl]
    eff = 1.0 - np.mean(gamma_slice)
    efficiencies.append(eff)

efficiencies = np.array(efficiencies)
jpt_centers  = np.array(jpt_centers)
jpt_lo = np.array([lo for lo, hi in JPT_BINS])
jpt_hi = np.array([hi for lo, hi in JPT_BINS])
xerr = [jpt_centers - jpt_lo, jpt_hi - jpt_centers]

os.makedirs(args.output, exist_ok=True)
hep.style.use('CMS')

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(jpt_centers, efficiencies, xerr=xerr, fmt='o', color='#3B2F2F')
ax.set_xscale('log')
ax.set_xlabel(r'Jet $p_T$ [GeV]', fontsize=12)
ax.set_ylabel('Matching Efficiency', fontsize=12)
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
hep.cms.label(ax=ax, data=False, text='Private', com=13)

fig.tight_layout()
fname = os.path.join(args.output, 'matching_efficiency.png')
fig.savefig(fname, dpi=150)
plt.close(fig)
print('Saved', fname)
