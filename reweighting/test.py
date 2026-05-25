from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import reweighting.reweighting_backend
import reweighting.smoothing
import simonplot as splt
import numpy as np

TT = build_pq_dataset(
    'VetoConfig1',
    'Mar_01_2026',
    'TT',
    'nominal',
    'jets',
    location = 'xrootd-submit',
)
ST = build_pq_dataset_stack(
    'VetoConfig1',
    'Mar_01_2026',
    'ST',
    'nominal',
    'jets',
    location = 'xrootd-submit',
)
diboson = build_pq_dataset_stack(
    'VetoConfig1',
    'Mar_01_2026',
    'diboson',
    'nominal',
    'jets',
    location = 'xrootd-submit',
)
pythia = build_pq_dataset_stack(
    'VetoConfig1',
    'Mar_01_2026',
    'Pythia_HTsum',
    'nominal',
    'jets',
    location = 'xrootd-submit',
)
pythia.compute_weight(1.0)
TT.compute_weight(1.0)
ST.compute_weight(1.0)
diboson.compute_weight(1.0)

import pyarrow.compute as pc
cut = None
variable = pc.field('Jpt')
bins = np.logspace(np.log10(30), np.log10(1000), 30, base=10)

evtwt_nominal = pc.field('wt_nominal')

labels = [
    "Spline smoothing (smoothing_factor=0.3)",
    "Spline smoothing (smoothing_factor=1.0)",
    "Spline smoothing (smoothing_factor=3.0)",
    "Spline smoothing (smoothing_factor=10)",
    "Spline smoothing (smoothing_factor=30)",
    "Spline smoothing (smoothing_factor=100)"
]

for denom, name in [(ST, "ST"), (diboson, "diboson"), (TT, "TT")]:
    smoothings = [
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=0.3
        ),
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=1.0
        ),
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=3
        ),
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=10
        ),
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=30
        ),
        reweighting.smoothing.SplineSmoothing(
            degree=3, smoothing_factor=100
        ),
    ]
    reweighting.reweighting_backend.compare_smothings(
        pythia,
        denom,
        cut,
        variable,
        wtvar_num = evtwt_nominal,
        wtvar_denom = evtwt_nominal,
        bins = bins,
        smoothings = smoothings,
        labels = labels,
        logx=True,
        logy=False,
        isMC = True,
        lumi = None,
        plot_path = f'reweighting/data/{name}'
    )
    reweighting.reweighting_backend.dump_smoothings(
        smoothings,
        labels,
        varname = 'Jpt',
        vardesc = 'pc.field("Jpt")',
        output_path = 'reweighting/data/reweightings_%s.json' % name
    )