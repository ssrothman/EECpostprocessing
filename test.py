from coffea.nanoevents import NanoEventsFactory
import json
from RecursiveNamespace import RecursiveNamespace
from processing.EECProcessor import EECProcessor

x = NanoEventsFactory.from_root('root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Jan31_2024_pythia_highstats_fixed_fixed/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/240131_170925/0000/NANO_miniAOD_395.root', entry_start=0, entry_stop=100).events()

with open("config.json", 'r') as f:
    config = RecursiveNamespace(**json.load(f))

p = EECProcessor(config)

p.process(x)
