import json

from general.fslookup.skim_path import lookup_skim_path
import os
from binning.main import build_hist, build_transfer_config, fill_hist, fill_cov
import pyarrow.dataset as ds

binpkgpath = os.path.dirname(__file__)
print(binpkgpath)
bincfgpath = os.path.join(
    binpkgpath,
    'config',
    'res4tee'
)
with open(bincfgpath + '.json') as f:
    bincfg = json.load(f)

fs, skimpath = lookup_skim_path(
    'scratch-submit',
    'BasicConfig',
    'Apr_23_2025',
    'Pythia_inclusive',
    'nominal',
    'res4tee_totalReco'
)

Hreco, prebinned_reco = build_hist(bincfg['reco'])
ds_reco = ds.dataset(skimpath, format='parquet', filesystem=fs)
print("Filling hist")
Hreco = fill_hist(
    Hreco,
    prebinned_reco, 
    ds_reco, 
    'wt_nominal', 
    itemwt = 'wt',
    statN = 100,
    statK = 0,
    reweight = None
)
print("Filling cov")
covreco = fill_cov(
    Hreco,
    prebinned_reco, 
    ds_reco, 
    'wt_nominal', 
    itemwt = 'wt',
    statN = 100,
    statK = 0,
    reweight = None
)

print("Filling transfer")
Htransfer, prebinned_transfer = build_hist(
    build_transfer_config(
        bincfg['gen'],
        bincfg['reco']
    )
)
fs, skimpath = lookup_skim_path(
    'scratch-submit',
    'BasicConfig',
    'Apr_23_2025',
    'Pythia_inclusive',
    'nominal',
    'res4tee_transfer'
)
ds_transfer = ds.dataset(skimpath, format="parquet", filesystem=fs)
Htransfer = fill_hist(
    Htransfer, 
    prebinned_transfer, 
    ds_transfer, 
    'wt_nominal', 
    itemwt = 'wt_reco',
    statN = 100,
    statK = 0,
    reweight = None
)