from samples.latest import SAMPLE_LIST

from plotting.EECutil import EEChistReader
from plotting.plotEEC import *

processes = [
    'DYJetsToLL',
    'TTTo2L2Nu',
    #'WW',
    #'ZZ',
    #'WZ',
#    'ST_tW_top',
#    'ST_tW_antitop'
]

# Load histograms
hists = {}
for process in processes:
    hists[process] = EEChistReader(SAMPLE_LIST.get_hist(process, 'EEC', ['tight', 'asData']))

# Plot
objs = [hists[process] for process in processes]
labels = processes
bins = {'order' : 0, 'pt' : 1}
density=True

compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)
bins['btag'] = 0
compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)

bins['btag'] = 1
for pt in [hist.underflow, 0, 1, 2, hist.overflow]:
    bins['pt'] = pt
    compareEEC_perObj(objs, None, 'Hreco', labels,
                      bins, density)
compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)
del bins['btag']
bins['genflav'] = 0
compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)
bins['genflav'] = 1
compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)
bins['genflav'] = 2
compareEEC_perObj(objs, None, 'Hreco', labels,
                  bins, density)
