from samples.samples import *

SAMPLE_LIST = SampleSet("Aug23_2024")

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL",
    tag='Aug23_2024_charged/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_Herwig",
    tag='Aug23_2024_charged/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='test',
    tag='test',
    location='test',
    JEC='MC',
    flags=None
))
