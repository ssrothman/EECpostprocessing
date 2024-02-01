from plotting.EECutil import EEChistReader

x = EEChistReader('Jan31_2024_pythia_highstats_fixed_fixed/EEC/hists.pkl')

a, b = x.forward('TrackDR')
