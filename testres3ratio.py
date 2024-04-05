import plotting.EECutil
import plotting.plotEEC
import matplotlib.pyplot as plt


x = plotting.EECutil.EEChistReader('Mar28_2024_binningtest_quick/EEC/')
plotting.plotEEC.plotRes3Ratio(x, x,
                               'EECs', 'EECs',
                               'Hres3', 'Hres3',
                               {'dRbin':2, 'btag':0}, {'dRbin' : 2, 'btag':1},
                               density=True, logcolor=False,
                               ratiomode='sigma')
