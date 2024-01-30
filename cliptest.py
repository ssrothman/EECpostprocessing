from coffea.nanoevents import NanoEventsFactory
from reading.reader import EECreader

f = '/home/submit/srothman/work/EEC/CMSSW_10_6_26/src/SRothman/NANO_miniAOD_clip.root'
x = NanoEventsFactory.from_root(f).events()

full = EECreader(x, 'RecoCaloShareEEC')
pu = EECreader(x, 'RecoCaloShareEECPU')
