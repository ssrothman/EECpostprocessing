import numpy as np
import awkward as ak
import hist
from util import ensure_mask

def getPtAxis(name="pt", label="$p_{T}$ [GeV]", minpt=30, maxpt=1000):
    return hist.axis.Regular(20, minpt, maxpt, name=name, label=label, 
                             transform=hist.axis.transform.log)

def getEtaAxis(name="eta", label="|$\eta$|"):
    return hist.axis.Regular(20, 0, 3, name=name, label=label, underflow=False)

def getPUAxis(name="pu", label="PU"):
    return hist.axis.IntCategory([0,1], name=name, label=label) 

def getPUfracAxis(name="pufrac", label="Pilup fraction"):
    return hist.axis.Regular(20, 0, 1, name=name, label=label, 
                             underflow=False, overflow=False)

def getNPartAxis(name="nPart", label="$N_{Constituents}$"):
    return hist.axis.Regular(20, 0, 50, name=name, label=label, underflow=False)

def getJetHist():
    return hist.Hist(
        getPtAxis(),
        getEtaAxis(),
        getPUAxis(),
        getPUfracAxis(),
        getNPartAxis(),
        storage=hist.storage.Weight())

def fillJetHist(h, r, evtwt=1, mask=None):
    parts = r.parts
    jets = r.jets
    mask = ensure_mask(mask, parts)

    wt, _ = ak.broadcast_arrays(evtwt, jets.pt)
    partPU = parts.nmatch == 0
    PU = ak.all(partPU==0, axis=-1)
    PUfrac = ak.sum(parts.pt * partPU, axis=-1) / ak.sum(parts.pt, axis=-1)

    h.fill(pt = ak.flatten(jets.pt[mask], axis=None),
           eta = np.abs(ak.flatten(jets.eta[mask], axis=None)),
           pu = ak.flatten(PU[mask], axis=None),
           pufrac = ak.flatten(PUfrac[mask], axis=None),
           nPart = ak.flatten(jets.nConstituents[mask], axis=None),
           weight = ak.flatten(wt[mask], axis=None))
