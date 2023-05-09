import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt

def getPtAxis(name="pt", label="$p_{T}$ [GeV]", minpt=0.1, maxpt=100):
    return hist.axis.Regular(20, minpt, maxpt, name=name, label=label, transform=hist.axis.transform.log)

def getEtaAxis(name="eta", label="|$\eta$|"):
    return hist.axis.Regular(20, 0, 3, name=name, label=label, underflow=False)

def getPIDaxis(name='pdgid', label='pdgid'):
    return hist.axis.IntCategory([11, 13, 22, 130, 211], name=name, label=label)

def getMatchAxis(name='nmatch', label='nmatch'):
    return hist.axis.Integer(0, 3, name=name, label=label, underflow=False)

def getMatchHist():
    return hist.Hist(
        getPtAxis(),
        getEtaAxis(),
        getPIDaxis(),
        getMatchAxis(),
        getMatchAxis(name='onmatch'),
        getPtAxis("jetpt", "$p_{T, Jet}$ [GeV]", minpt=30, maxpt=1000),
        getPtAxis("fracpt", "$p_{T}/p_{T, Jet}$ [GeV]", minpt=0.01, maxpt=1),
        storage=hist.storage.Weight())

def fillMatchHist(h, r, evtwt=1, EECmask=None):
    parts = r.parts
    jets = r.jets
    if EECmask is None:
        EECmask = np.ones(len(parts), dtype=bool)

    nonzeroMask = ak.sum(parts.nmatch, axis=-1) > 0 #mask out unmatched entire jets
    EECmask = EECmask & nonzeroMask

    wt, jetpt, _ = ak.broadcast_arrays(evtwt, jets.pt, parts.pt)
    h.fill(pt = ak.flatten(parts.pt[EECmask], axis=None),
           eta = np.abs(ak.flatten(parts.eta[EECmask], axis=None)),
           pdgid = ak.flatten(parts.pdgid[EECmask], axis=None),
           nmatch = ak.flatten(parts.nmatch[EECmask], axis=None),
           onmatch = ak.flatten(parts.onmatch[EECmask], axis=None),
           jetpt = ak.flatten(jetpt[EECmask], axis=None),
           fracpt = ak.flatten((parts.pt/jetpt)[EECmask], axis=None),
           weight = ak.flatten(wt[EECmask], axis=None))

    def postprocess(self, accumulator):
            pass
