import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet
import json

'''
Utility methods - will be moved to utility package later
'''
def walk_datatree(data, variable : str) -> tuple[float, float]:
    if type(data) in [list, tuple]:
        data = data[0]
    
    if not 'nodetype' in data:
        raise ValueError("Could not find binning for %s" % variable)

    if data['nodetype'] == 'binning' and data['input'] == variable:
        return data['edges'][0], data['edges'][-1]

    return walk_datatree(data['content'], variable)

def findBinEdges(cset : CorrectionSet, correction : str, variable : str) -> tuple[float, float]:
    csetdata = json.loads(cset._data)
    names = [corr['name'] for corr in csetdata['corrections']]

    icorr = names.index(correction)
    if icorr == -1:
        raise ValueError(f"Correction {correction} not found in CorrectionSet")

    corrdata = csetdata['corrections'][icorr]

    return walk_datatree(corrdata['data'], variable)

def getMuonSF(cset : CorrectionSet, name : str, eta : ak.Array, pt : ak.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    badnone = ak.is_none(eta) | ak.is_none(pt)
    pt = ak.fill_none(pt, 0)
    eta = ak.fill_none(eta, 0)

    minpt, _ = findBinEdges(cset, name, 'pt')
    _, maxeta = findBinEdges(cset, name, 'eta')

    ptthresh = minpt
    etathresh = maxeta - 1e-8

    badpt = (pt < ptthresh)
    badeta = (np.abs(eta) > etathresh)
    bad = badpt | badeta | badnone

    etaEval = ak.fill_none(eta, 0)
    ptEval = ak.fill_none(pt, 0)

    etaEval = ak.to_numpy(ak.where(bad, etathresh, etaEval))
    ptEval = ak.to_numpy(ak.where(bad, ptthresh, ptEval))

    nom = cset[name].evaluate(
        etaEval,
        ptEval,
        'nominal'
    )
    up = cset[name].evaluate(
        etaEval,
        ptEval,
        'systup'
    )
    dn = cset[name].evaluate(
        etaEval,
        ptEval,
        'systdown'
    )
    
    nom = np.where(bad, 1, nom)
    up = np.where(bad, 1, up)
    dn = np.where(bad, 1, dn)

    return nom, up, dn

