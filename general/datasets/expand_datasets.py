_ALL_HT_DATASETS = [
    'Pythia_HT-0to70',
    'Pythia_HT-70to100',
    'Pythia_HT-100to200',
    'Pythia_HT-200to400',
    'Pythia_HT-400to600',
    'Pythia_HT-600to800',
    'Pythia_HT-800to1200',
    'Pythia_HT-1200to2500',
    'Pythia_HT-2500toInf',
]
_ALL_SIGNAL_DATASETS = [
    *_ALL_HT_DATASETS,
    "Pythia_inclusive",
    "Herwig_inclusive"
]
_ALL_BACKGROUND_DATASETS = [
    "ZZ",
    "WZ",
    "WW",
    "TT",
    "ST_t",
    "ST_t_anti",
    "ST_tW",
    "ST_tW_anti",
]
_ALL_DATA_DATASETS = [
    "DATA_2018A",
    "DATA_2018B",
    "DATA_2018C",
    "DATA_2018D"
]
_ALL_GENONLY_DATASETS = [
    "herwig_glu_1200",
    "herwig_glu_250",
    "herwig_glu_50",
    "herwig_glu_nohad_1200",
    "herwig_glu_nohad_250",
    "herwig_glu_nohad_50",
    "herwig_glu_nohardspin_1200",
    "herwig_glu_nohardspin_250",
    "herwig_glu_nohardspin_50",
    "herwig_glu_nosoftspin_1200",
    "herwig_glu_nosoftspin_250",
    "herwig_glu_nosoftspin_50",
    "herwig_glu_nospin_1200",
    "herwig_glu_nospin_250",
    "herwig_glu_nospin_50",
    "herwig_q_1200",
    "herwig_q_250",
    "herwig_q_50",
    "herwig_q_nohad_1200",
    "herwig_q_nohad_250",
    "herwig_q_nohad_50",
    "herwig_q_nohardspin_1200",
    "herwig_q_nohardspin_250",
    "herwig_q_nohardspin_50",
    "herwig_q_nosoftspin_1200",
    "herwig_q_nosoftspin_250",
    "herwig_q_nosoftspin_50",
    "herwig_q_nospin_1200",
    "herwig_q_nospin_250",
    "herwig_q_nospin_50",
    "pythia_glu_1200",
    "pythia_glu_250",
    "pythia_glu_50",
    "pythia_glu_nohad_1200",
    "pythia_glu_nohad_250",
    "pythia_glu_nohad_50",
    "pythia_glu_nospin_1200",
    "pythia_glu_nospin_250",
    "pythia_glu_nospin_50",
    "pythia_q_1200",
    "pythia_q_250",
    "pythia_q_50",
    "pythia_q_nohad_1200",
    "pythia_q_nohad_250",
    "pythia_q_nohad_50",
    "pythia_q_nospin_1200",
    "pythia_q_nospin_250",
    "pythia_q_nospin_50",
]


_ALL_GENONLY_Q_DATASETS = [
    "herwig_q_1200",
    "herwig_q_250",
    "herwig_q_50",
    "herwig_q_nohad_1200",
    "herwig_q_nohad_250",
    "herwig_q_nohad_50",
    "herwig_q_nohardspin_1200",
    "herwig_q_nohardspin_250",
    "herwig_q_nohardspin_50",
    "herwig_q_nosoftspin_1200",
    "herwig_q_nosoftspin_250",
    "herwig_q_nosoftspin_50",
    "herwig_q_nospin_1200",
    "herwig_q_nospin_250",
    "herwig_q_nospin_50",
    "pythia_q_1200",
    "pythia_q_250",
    "pythia_q_50",
    "pythia_q_nohad_1200",
    "pythia_q_nohad_250",
    "pythia_q_nohad_50",
    "pythia_q_nospin_1200",
    "pythia_q_nospin_250",
    "pythia_q_nospin_50",
]


def expand_one_dataset(dataset:str):
    if dataset == 'allSignal':
        return _ALL_SIGNAL_DATASETS
    elif dataset == 'allBackground':
        return _ALL_BACKGROUND_DATASETS
    elif dataset == 'allData':
        return _ALL_DATA_DATASETS
    elif dataset == 'allGenonly':
        return _ALL_GENONLY_DATASETS
    elif dataset == 'allGenonlyQ':
        return _ALL_GENONLY_Q_DATASETS
    else:
        return [dataset]

def expand_datasets(datasets):
    expanded = []
    for dataset in datasets:
        expanded.extend(expand_one_dataset(dataset))
    return expanded
