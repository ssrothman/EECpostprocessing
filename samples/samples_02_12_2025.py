from samples.samples import *

SAMPLE_LIST = SampleSet("Jan_22_2024")

SAMPLE_LIST.add_sample(Sample(
    name="Herwig_inclusive",
    tag='crab_Feb_12_2025/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_inclusive",
    tag='crab_Feb_12_2025/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8',
    location="LPC",
    JEC="MC",
    flags=None
))

