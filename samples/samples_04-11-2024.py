from samples.samples import *

SAMPLE_LIST = SampleSet("Apr24_2024")

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL",
    tag="Mar31_2024_nom_highstats_wbugfix/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
    location="scratch",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-0to70",
    tag="Mar31_2024_nom_highstats_wbugfix/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=["HTcut70"]
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-70to100",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-100to200",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-200to400",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-400to600",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-600to800",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))
SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-800to1200",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-1200to2500",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_HT-2500toInf",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL_Pythia/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
    location="LPC",
    JEC="MC",
    flags=None,
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_Herwig",
    tag="Mar31_2024_nom_highstats_wbugfix/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_aMCatNLO",
    tag="Apr09_2024_nom_highstats/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    location="LPC",
    JEC="MC",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='TTTo2L2Nu',
    tag='Mar29_2024_nom_highstats/2018/TT',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ZZ',
    tag='Mar29_2024_nom_highstats/2018/ZZ',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='WZ',
    tag='Mar29_2024_nom_highstats/2018/WZ',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='WW',
    tag='Mar29_2024_nom_highstats/2018/WW',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_tW_antitop',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_tW_top',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_antitop_4f',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_antitop_5f',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_top_4f',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name='ST_t_top_5f',
    tag='Apr09_2024_nom_highstats/2018/ST/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    location='LPC',
    JEC='MC',
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018A",
    tag="Mar29_2024_nom_highstats/2018/SingleMuon/SingleMuon/2018A",
    location="scratch",
    JEC="2018A",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018B",
    tag="Mar29_2024_nom_highstats/2018/SingleMuon/SingleMuon/2018B",
    location="scratch",
    JEC="2018B",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018C",
    tag="Mar29_2024_nom_highstats/2018/SingleMuon/SingleMuon/2018C",
    location="scratch",
    JEC="2018C",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018D",
    tag="Mar29_2024_nom_highstats/2018/SingleMuon/SingleMuon/2018D",
    location="LPC",
    JEC="2018D",
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DATA_2018UL",
    tag='Mar29_2024_nom_highstats/',
    location=None,
    JEC=None,
    flags=None
))

SAMPLE_LIST.add_sample(Sample(
    name="DYJetsToLL_allHT",
    tag='Apr09_2024_nom_highstats/',
    location=None,
    JEC=None,
    flags=None
))

