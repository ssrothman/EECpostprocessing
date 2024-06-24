from samples.latest import SAMPLE_LIST
import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt

names = ['WW', 'ZZ', 'WZ',
         'TTTo2L2Nu', 
         'ST_tW_top', 'ST_tW_antitop',
         'ST_t_top_5f', 
         'ST_t_antitop_5f', 
         'DYJetsToLL_allHT']

from plotting.util import *

for i in range(len(names)):
    #print(samples[i].keys())
    try:
        H = SAMPLE_LIST.get_hist(names[i], 'Kin', ['tight'])
    except Exception as e:
        print(e)
        continue

    if names[i].endswith("5f"):
        xsec = vars(config.xsecs)[names[i][:-3]]
    elif names[i].endswith("allHT"):
        xsec = vars(config.xsecs)['DYJetsToLL']
    else:
        xsec = vars(config.xsecs)[names[i]]
    factor = 1000 * config.totalLumi * xsec / H['sumwt']

    print("\tfactor          %0.0f"%factor)
    print("\ttotal events    %0.0f"%(H['sumwt'] * factor/1))
    print("\tpassing events  %0.0f"%(H['sumwt_pass'] * factor/1))
    print("\tnum jets        %0.0f"%(H['HJet'].sum(flow=True).value * factor/1))
    print("\tnum udsg jets   %0.0f"%(H['HJet'][{'btag_tight' : 0}].sum(flow=True).value * factor/1))
    print("\tnum b jets      %0.0f"%(H['HJet'][{'btag_tight' : 1}].sum(flow=True).value * factor/1))

Hdata = SAMPLE_LIST.get_hist('DATA_2018UL', 'Kin', ['tight'])
print("DATA")
print("\ttotal events    %0.0f"%(Hdata['sumwt']/1))
print("\tpassing events  %0.0f"%(Hdata['sumwt_pass']/1))
print("\tnum jets        %0.0f"%(Hdata['HJet'].sum(flow=True).value/1))
print("\tnum udsg jets   %0.0f"%(Hdata['HJet'][{'btag_tight' : 0}].sum(flow=True).value/1))
print("\tnum b jets      %0.0f"%(Hdata['HJet'][{'btag_tight' : 1}].sum(flow=True).value/1))
