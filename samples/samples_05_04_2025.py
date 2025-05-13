from samples.samples import *

SAMPLE_LIST = SampleSet("May_04_2025")

samples = [
    'herwig_glu_gg',
    'herwig_glu',
    'herwig_glu_nospin_gg',
    'herwig_glu_nospin',
    'herwig_glu_TeV_gg',
    'herwig_glu_TeV',
    'herwig_glu_TeV_nospin_gg',
    'herwig_glu_TeV_nospin',
    'herwig_q',
    'herwig_q_nogg',
    'herwig_q_nospin',
    'herwig_q_nospin_nogg',
    'herwig_q_TeV',
    'herwig_q_TeV_nogg',
    'herwig_q_TeV_nospin',
    'herwig_q_TeV_nospin_nogg',
    'pythia_glu_gg',
    'pythia_glu',
    'pythia_glu_nospin_gg',
    'pythia_glu_nospin',
    'pythia_glu_TeV_gg',
    'pythia_glu_TeV',
    'pythia_glu_TeV_nospin_gg',
    'pythia_glu_TeV_nospin',
    'pythia_q',
    'pythia_q_nospin',
    'pythia_q_TeV',
    'pythia_q_TeV_nospin'
]

for sample in samples:
    SAMPLE_LIST.add_sample(Sample(
        name=sample,
        tag=sample+"_May_04_2025",
        location="LPC",
        JEC="skip",
        flags=None
    ))
