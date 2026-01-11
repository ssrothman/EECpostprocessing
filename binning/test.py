import json

with open("binning/config/res4tee.json", 'r') as f:
    binning_cfg = json.load(f)

dspath_gen = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_gen/'
dspath_reco = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_reco/'
dspath_transfer = '/ceph/submit/data/group/cms/store/user/srothman/EEC_v2/BasicConfig/Apr_23_2025/Pythia_inclusive/nominal/res4tee_transfer/'

from binning.main import build_hist, build_transfer_hist, fill_hist
import pyarrow.dataset as ds

Hgen, names_gen = build_hist(binning_cfg['gen'])
ds_gen = ds.dataset(dspath_gen, format="parquet")
fill_hist(Hgen, names_gen, ds_gen, 'wt_nominal')

Hreco, names_reco = build_hist(binning_cfg['reco'])
ds_reco = ds.dataset(dspath_reco, format="parquet")
fill_hist(Hreco, names_reco, ds_reco, 'wt_nominal')

Htransfer, names_transfer = build_transfer_hist(
    binning_cfg['gen'], binning_cfg['reco']
)
ds_transfer = ds.dataset(dspath_transfer, format="parquet")
fill_hist(Htransfer, names_transfer, ds_transfer, 'wt_nominal')