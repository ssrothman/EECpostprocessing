import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use('CMS')

EOS_BASE = '/eos/user/d/dponman/EvtMCprojConfig/new_v3/DYJetsToLL_Pythia/NOM'

R_bins = np.array([
    0.001, 0.001139, 0.001298, 0.001479, 0.001685, 0.00192, 0.002187, 0.002492,
    0.002839, 0.003235, 0.003686, 0.004199, 0.004784, 0.005451, 0.00621, 0.007075,
    0.008061, 0.009184, 0.010464, 0.011922, 0.013583, 0.015476, 0.017632, 0.020089,
    0.022888, 0.026077, 0.02971, 0.03385, 0.038566, 0.04394, 0.050062, 0.057037,
    0.064984, 0.074038, 0.084354, 0.096107, 0.109498, 0.124755, 0.142137, 0.161941,
    0.184505, 0.210212, 0.239501, 0.272871, 0.310891, 0.354208, 0.40356, 0.459789,
    0.523852, 0.596841, 0.68
])
R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

jpt_bins = [40, 100, 200, 340, 520, 740, 1000]

os.makedirs('plots/proj', exist_ok=True)

def load_table(table):
    files = glob.glob(os.path.join(EOS_BASE, table, '*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    R_idx = df['R'].astype(int)
    valid = (R_idx >= 0) & (R_idx < len(R_centers))
    return df[valid].copy()

def eec_hist(df, lo, hi):
    mask = (df['Jpt'] >= lo) & (df['Jpt'] < hi)
    sub = df[mask]
    h = np.zeros(len(R_centers))
    np.add.at(h, sub['R'].astype(int).values, (sub['wt'] * sub['wt_nominal']).values)
    return h

df_reco = load_table('proj_Reco')
df_gen  = load_table('proj_Gen')

fig, axes = plt.subplots(2, 3, figsize=(24, 14), sharey=False)
plt.rcParams.update({'font.size': 15})
axes = axes.flatten()

for i, (lo, hi) in enumerate(zip(jpt_bins[:-1], jpt_bins[1:])):
    h_reco = eec_hist(df_reco, lo, hi)
    h_gen  = eec_hist(df_gen,  lo, hi)

    ax = axes[i]
    ax.step(R_centers, h_reco, where='mid', color='red',   label='Reco')
    ax.step(R_centers, h_gen,  where='mid', color='green', label='Gen')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('R')
    ax.set_ylabel('A.U.')
    ax.set_title(f'Jpt [{lo}, {hi}] GeV')
    ax.legend()
    hep.cms.label('Private', ax=ax, loc=0, fontsize=10)

plt.tight_layout()
plt.savefig('plots/proj/EEC_R_per_jpt_pythia.pdf')
plt.close()
print('Saved plots/proj/EEC_R_per_jpt_pythia.pdf')

# Jpt distribution
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(df_reco['Jpt'], bins=np.linspace(4, 1000, 101),
        weights=df_reco['wt_nominal'], histtype='step', color='red',   label='Reco')
ax.hist(df_gen['Jpt'],  bins=np.linspace(4, 1000, 101),
        weights=df_gen['wt_nominal'],  histtype='step', color='green', label='Gen')
ax.set_xlabel('Jet $p_T$ [GeV]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
hep.cms.label('Private', ax=ax, loc=0, fontsize=10)
plt.tight_layout()
plt.savefig('plots/proj/Jpt_pythia.pdf')
plt.close()
print('Saved plots/proj/Jpt_pythia.pdf')
