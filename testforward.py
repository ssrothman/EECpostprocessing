from plotting.EECutil import EEChistReader

pythia = EEChistReader('Jan31_2024_pythia_highstats_fixed_fixed/EEC/hists.pkl')
herwig = EEChistReader('Jan31_2024_herwig_highstats_fixed_fixed/EEC/hists.pkl')

from plotting.plotEEC import *

for ptbin in range(5):
    compareForward(pythia, 'TrackDR', herwig, 'TrackDR',
                   ptbin = ptbin, etabin=None, pubin=None,
                   doTemplates = True, density=True)

for ptbin in range(5):
    compareForward(pythia, 'TrackDR', herwig, 'TrackDR',
                   ptbin = ptbin, etabin=None, pubin=None,
                   doTemplates = False, density=True)
