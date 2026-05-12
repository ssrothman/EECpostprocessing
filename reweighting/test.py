from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import reweighting.reweighting_backend
import reweighting.smoothing
import simonplot as splt
import numpy as np

herwig = build_pq_dataset(
    'VetoConfig1',
    'Mar_01_2026',
    'Herwig_inclusive',
    'nominal',
    'events',
    location = 'xrootd-submit',
)
pythia = build_pq_dataset(
    'VetoConfig1',
    'Mar_01_2026',
    'Pythia_inclusive',
    'nominal',
    'events',
    location = 'xrootd-submit',
)

cut = splt.cut.NoCut()
variable = splt.variable.BasicVariable(
    'Zpt'
)
bins = np.logspace(0, 3, 30, base=10)

evtwt_nominal = splt.variable.BasicVariable(
    'wt_nominal'
)

smoothings = [
    None,
    reweighting.smoothing.SplineSmoothing(
        degree=3, smoothing_factor=0.5
    ),
    reweighting.smoothing.SplineSmoothing(
        degree=3, smoothing_factor=50
    ),
    reweighting.smoothing.SplineSmoothing(
        degree=3, smoothing_factor=500
    )
]
labels = [
    "No smoothing",
    "Spline smoothing (smoothing_factor=0.5)",
    "Spline smoothing (smoothing_factor=50)",
    "Spline smoothing (smoothing_factor=500)"
]

reweighting.reweighting_backend.compare_smothings(
    pythia,
    herwig,
    cut,
    variable,
    wtvar_num = evtwt_nominal,
    wtvar_denom = evtwt_nominal,
    bins = bins,
    smoothings = smoothings,
    labels = labels,
    logx=True,
    logy=True
)