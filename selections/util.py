import numpy as np
import correctionlib
import json

def findBinEdges(cset, correction, variable):
    csetdata = json.loads(cset._data)
    names = [corr['name'] for corr in csetdata['corrections']]

    icorr = names.index(correction)
    if icorr == -1:
        raise ValueError(f"Correction {correction} not found in CorrectionSet")

    corrdata = csetdata['corrections'][icorr]

    return walk_datatree(corrdata['data'], variable)

def walk_datatree(data, variable):
    if type(data) in [list, tuple]:
        data = data[0]
    
    if not 'nodetype' in data:
        raise ValueError("Could not find binning for %s" % variable)

    if data['nodetype'] == 'binning' and data['input'] == variable:
            return data['edges'][0], data['edges'][-1]

    return walk_datatree(data['content'], variable)

