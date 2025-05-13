from samples.samples import *

SAMPLE_LIST = SampleSet("Apr_23_2025")

base = 'crab/crab_Apr_23_2025/2018/'

SAMPLE_LIST.add_sample(Sample(
    name = 'DATA_2018A',
    tag = [base+'SingleMuon/SingleMuon/2018A',
           base+'SingleMuon/SingleMuon/2018A_recovery',
           base+'SingleMuon/SingleMuon/2018A_recovery2',
           base+'SingleMuon/SingleMuon/2018A_recovery3'],
    location ='LPC',
    JEC='DATA_2018A',
    flags= None
))

SAMPLE_LIST.add_sample(Sample(
    name = 'DATA_2018B',
    tag = [base+'SingleMuon/SingleMuon/2018B'],
    location ='LPC',
    JEC='DATA_2018B',
    flags= None
))

SAMPLE_LIST.add_sample(Sample(
    name = 'DATA_2018C',
    tag = [base+'SingleMuon/SingleMuon/2018C_2',
           base+'SingleMuon/SingleMuon/2018C_recovery',
           base+'SingleMuon/SingleMuon/2018C_recovery2'],
    location ='LPC',
    JEC='DATA_2018C',
    flags= None
))

SAMPLE_LIST.add_sample(Sample(
    name = 'DATA_2018D',
    tag = [base+'SingleMuon/SingleMuon/2018D'],
    location ='LPC',
    JEC='DATA_2018D',
    flags= None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_inclusive",
    tag=base+'DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_inclusive',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Herwig_inclusive",
    tag=base+'DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/Herwig_inclusive_2',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-0to70",
    tag=base+'DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_inclusive',
    location="LPC",
    JEC="MC",
    flags=['HTcut70']
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-70to100",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-70to100',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-100to200",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-100to200',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-200to400",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-200to400',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-400to600",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-400to600',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-600to800",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-600to800',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-800to1200",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-800to1200',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-1200to2500",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-1200to2500',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="Pythia_HT-2500toInf",
    tag=base+'DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/DYJetsToLL_HT-2500toInf',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="WW",
    tag=base+'WW/WW_TuneCP5_13TeV-pythia8/WW',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="WZ",
    tag=base+'WZ/WZ_TuneCP5_13TeV-pythia8/WZ',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="ZZ",
    tag=base+'ZZ/ZZ_TuneCP5_13TeV-pythia8/ZZ',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="TT",
    tag=base+'TT/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="ST_t",
    tag=base+'ST/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_t_5f',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="ST_t_anti",
    tag=base+'ST/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_t_anti_5f',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="ST_tW",
    tag=base+'ST/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_5f',
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="ST_tW_anti",
    tag=base+'ST/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_anti_5f',
    location="LPC",
    JEC="MC",
    flags=None
))
