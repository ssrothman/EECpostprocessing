import json

with open("binning/config/res4tee.json", 'r') as f:
    binning_cfg = json.load(f)

dspath_gen = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_gen/'
dspath_reco = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_reco/'
dspath_transfer = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_transfer/'

from binning.main import build_hist, build_transfer_config, fill_hist, fill_cov
import pyarrow.dataset as ds

gencfg = binning_cfg['gen']
recocfg = binning_cfg['reco']
transfercfg = build_transfer_config(gencfg, recocfg)

Hgen, prebinned_gen = build_hist(gencfg)
ds_gen = ds.dataset(dspath_gen, format="parquet")
Hgen = fill_hist(Hgen, prebinned_gen, ds_gen, 'wt_nominal', 'wt')
covgen = fill_cov(Hgen, ds_gen, 'wt_nominal', 'wt')


Hreco, prebinned_reco = build_hist(recocfg)
ds_reco = ds.dataset(dspath_reco, format="parquet")
Hreco = fill_hist(Hreco, prebinned_reco, ds_reco, 'wt_nominal', 'wt')
covreco = fill_cov(Hreco, ds_reco, 'wt_nominal', 'wt')

Htransfer, prebinned_transfer = build_hist(transfercfg)
ds_transfer = ds.dataset(dspath_transfer, format="parquet")
Htransfer = fill_hist(Htransfer, prebinned_transfer, ds_transfer, 'wt_nominal', 'wt_reco')