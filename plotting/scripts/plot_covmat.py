#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to covariance matrix .npy file")
parser.add_argument("--output", type=str, default="covmat.png")
args = parser.parse_args()

cov = np.load(args.path)

diag = np.sqrt(np.diag(cov))
denom = np.outer(diag, diag)
denom = np.where(denom == 0, 1.0, denom)
corr = cov / denom

fig, ax = plt.subplots(figsize=(8, 7))
hep.style.use("CMS")

im = ax.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax)

ax.set_xlabel("Bin index")
ax.set_ylabel("Bin index")

hep.cms.label(ax=ax, data=False, label='Private Simulation', com=13)

fig.tight_layout()
fig.savefig(args.output, dpi=150)
print("Saved to", args.output)
