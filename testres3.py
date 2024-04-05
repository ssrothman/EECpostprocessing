from coffea.nanoevents import NanoEventsFactory
from reading.reader import EECreader, transferreader

import awkward as ak
import numpy as np

x = NanoEventsFactory.from_root("root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Feb20_2024_charged_CORR/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/Herwig_ak8/240220_214659/0000/NANO_miniAOD_1.root").events()

rReco = EECreader(x, 'RecoChargedEECs')
rRecoUM = EECreader(x, 'RecoChargedEECsPU')
rTrans = transferreader(x, 'ChargedEECsTransfer')
