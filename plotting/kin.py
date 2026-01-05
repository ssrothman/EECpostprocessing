from torch import dsmm
import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import pyarrow.dataset as ds

evtds_allMC = build_pq_dataset_stack(
    'BasicConfig',
    'Apr_23_2025',
    'allMC',
    'nominal',
    'events'
)

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
evtds_data = build_pq_dataset_stack(
    'BasicConfig',
    'Apr_23_2025',
    'DATA',
    'DATA',
    'events'
)


evtvars = [splt.variable.BasicVariable(name) for name in evtds_data._datasets[0].schema.names if 'wt_' not in name and name != 'event_id']
cut = splt.cut.NoCut()
#weight = splt.variable.BasicVariable('wt_ISRDown')
weight = splt.variable.ConstantVariable(1.0)
binning = splt.binning.AutoBinning()

for var in evtvars:
    print(f"Plotting variable {var.key}")
    splt.plot_histogram(
        var,
        cut,
        weight,
        [
            evtds_allMC,
            evtds_data
        ],
        binning,
        output_folder='testplots/kin',
        output_prefix='evt',
    )
    faslkjafsdlkj