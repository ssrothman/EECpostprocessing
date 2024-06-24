from samples.latest import SAMPLE_LIST
import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt

names = ['WW', 'ZZ', 'WZ',
         'TTTo2L2Nu', 
         'ST_tW_top', 'ST_tW_antitop',
         'ST_t_antitop_5f', 
         'DYJetsToLL',
         'ST_t_top_5f', 
         'DYJetsToLL']

for i in range(len(names)):
    #print(samples[i].keys())
    H = SAMPLE_LIST.get_hist(names[i], 'Kin', ['noMETveto', 'tight'])['HMET']
    print(names[i], H.sum(flow=True).value)
    proj = H.project("METsig")#.plot(density=True, label=labels[i])
    total = proj.sum(flow=True).value
    for cut in [2, 3, 4, 5, 7, 10, 20, 50]:
        print("\t",cut, proj[hist.underflow:cut:hist.sum].value/total)

plt.legend()
plt.show()
