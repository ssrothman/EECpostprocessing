from samples.samples import *

SAMPLE_LIST = SampleSet("Jul14_2024")

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL",
    tag="Jul12_2024/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_Herwig",
    tag="Jul12_2024/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-0to70",
    tag="Jul12_2024/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=["HTcut70"]
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-70to100",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-100to200",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-200to400",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-400to600",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-600to800",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))
SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-800to1200",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-1200to2500",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-2500toInf",
    tag="Jul12_2024/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None,
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_tW_antitop',
    tag='Jul12_2024/2018/ST/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_tW_top',
    tag='Jul12_2024/2018/ST/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_antitop',
    tag='Jul12_2024/2018/ST/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_top',
    tag='Jul12_2024/2018/ST/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='TTTo2L2Nu',
    tag='Jul12_2024/2018/TT',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ZZ',
    tag='Jul12_2024/2018/ZZ',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='WZ',
    tag='Jul12_2024/2018/WZ',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='WW',
    tag='Jul12_2024/2018/WW',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018A",
    tag="Jul12_2024/2018/SingleMuon/SingleMuon/2018A",
    location="LPC",
    JEC="2018A",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018B",
    tag="Jul12_2024/2018/SingleMuon/SingleMuon/2018B",
    location="LPC",
    JEC="2018B",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018C",
    tag="Jul12_2024/2018/SingleMuon/SingleMuon/2018C",
    location="LPC",
    JEC="2018C",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018D",
    tag="Jul12_2024/2018/SingleMuon/SingleMuon/2018D",
    location="LPC",
    JEC="2018D",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018UL",
    tag='Jul12_2024',
    location=None,
    JEC=None,
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_allHT",
    tag='Jul12_2024',
    location=None,
    JEC=None,
    flags=None
))

