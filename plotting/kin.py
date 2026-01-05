from torch import dsmm
import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import pyarrow.dataset as ds

evtds_htsum = build_pq_dataset_stack(
    'BasicConfig',
    'Apr_23_2025',
    'Pythia_HTsum',
    'nominal',
    'events'
)
evtds_incl = build_pq_dataset(
    'BasicConfig',
    'Apr_23_2025',
    'Pythia_inclusive',
    'nominal',
    'events'
)


evtvars = [splt.variable.BasicVariable(name) for name in evtds_incl.schema.names if 'wt_' not in name and name != 'event_id']
cut = splt.cut.NoCut()
weight = splt.variable.BasicVariable('wt_ISRDown')
binning = splt.binning.AutoBinning()

for var in evtvars:
    print(f"Plotting variable {var.key}")
    splt.plot_histogram(
        var,
        cut,
        weight,
        [  
            evtds_incl,
            evtds_htsum
        ],
        binning,
        output_folder='testplots/kin',
        output_prefix='evt',
    )