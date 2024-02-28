import numpy as np
import awkward as ak
from hist.axis import Variable, Integer, IntCategory
from hist.storage import Double
from hist import Hist
from time import time

def squash(arr):
    return ak.to_numpy(ak.flatten(arr, axis=None))

def getGenFlav(rJet, iReco, mask):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask]
        else:
            return rJet.jets.pt[iReco][mask] * 0
    else:
        if hasattr(rJet.CHSjets, 'hadronFlavour'):
            flavors = ak.max(rJet.CHSjets.hadronFlavour[iReco][mask], axis=-1)
        else:
            flavors = ak.zeros_like(rJet.jets.pt[iReco][mask])
        return flavors

def getBdisc(rJet, iReco, mask, config):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        else:
            return ak.zeros_like(rJet.jets.pt[iReco][mask])

    CHS = rJet.CHSjets
    if config['algo'] == 'deepjet':
        btag = CHS.btagDeepFlavB[iReco][mask]
    elif config['algo'] == 'deepcsv':
        btag = CHS.btagDeepB[iReco][mask]
    else:
        raise NotImplementedError("deepjet and deepcsv are the only available btagging algos")

    return ak.max(btag, axis=-1)

def getCvLdisc(rJet, iReco, mask, config):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        else:
            return ak.zeros_like(rJet.jets.pt[iReco][mask])

    CHS = rJet.CHSjets
    if config['algo'] == 'deepjet':
        CvL = CHS.btagDeepFlavCvL[iReco][mask]
    elif config['algo'] == 'deepcsv':
        CvL = CHS.btagDeepCvL[iReco][mask]
    else:
        raise NotImplementedError("deepjet and deepcsv are the only available ctagging algos")

    return ak.max(CvL, axis=-1)

def getCvBdisc(rJet, iReco, mask, config):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        else:
            return ak.zeros_like(rJet.jets.pt[iReco][mask])

    CHS = rJet.CHSjets
    if config['algo'] == 'deepjet':
        CvB = CHS.btagDeepFlavCvB[iReco][mask]
    elif config['algo'] == 'deepcsv':
        CvB = CHS.btagDeepCvB[iReco][mask]
    else:
        raise NotImplementedError("deepjet and deepcsv are the only available ctagging algos")

    return ak.max(CvB, axis=-1)

def passBtag(rJet, iReco, mask, config):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        else:
            return ak.zeros_like(rJet.jets.pt[iReco][mask])

    CHS = rJet.CHSjets
    if config['algo'] == 'deepjet':
        btag = CHS.btagDeepFlavB[iReco][mask]
    elif config['algo'] == 'deepcsv':
        btag = CHS.btagDeepB[iReco][mask]
    else:
        raise NotImplementedError("deepjet and deepcsv are the only available btagging algos")

    blabel = config['wp']
    if blabel == 'tight':
        bwp = config['WPcuts'].tight
    elif blabel == 'medium':
        bwp = config['WPcuts'].medium
    elif blabel == 'loose':
        bwp = config['WPcuts'].loose
    else:
        raise NotImplementedError("WP needs to be 'loose', 'medium', or 'tight'")

    return ak.max(btag > bwp, axis=-1)
    
def passCtag(rJet, iReco, mask, config):
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        else:
            return ak.zeros_like(rJet.jets.pt[iReco][mask])

    CHS = rJet.CHSjets
    if config['algo'] == 'deepjet':
        CvL = CHS.btagDeepFlavCvL[iReco][mask]
        CvB = CHS.btagDeepFlavCvB[iReco][mask]
    elif config['algo'] == 'deepcsv':
        CvL = CHS.btagDeepCvL[iReco][mask]
        CvB = CHS.btagDeepCvB[iReco][mask]
    else:
        raise NotImplementedError("deepjet and deepcsv are the only available btagging algos")

    blabel = config['wp']
    if blabel == 'tight':
        CvLwp = config['CvLcuts'].tight
        CvBwp = config['CvBcuts'].tight
    elif blabel == 'medium':
        CvLwp = config['CvLcuts'].medium
        CvBwp = config['CvBcuts'].medium
    elif blabel == 'loose':
        CvLwp = config['CvLcuts'].loose
        CvBwp = config['CvBcuts'].loose

    return ak.max((CvL > CvLwp) & (CvB > CvBwp), axis=-1)

def getAxis(name, config, suffix=''):
    if name == 'pt':
        return Variable(config['pt'], 
                        name='pt'+suffix,
                        label = 'Jet $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == 'dRbin':
        return Integer(0, config['dRbin'], 
                       name='dRbin'+suffix,
                       label = '$\Delta R$ bin',
                       underflow=False, overflow=False)
    elif name == 'nPU':
        return Variable(config['nPU'], 
                        name='nPU'+suffix,
                        label='Number of PU vertices',
                        overflow=True, underflow=False)
    elif name == 'order':
        return Integer(2, config['order']+1, 
                       name='order'+suffix, 
                       label = 'EEC Order',
                       underflow=False, overflow=False)
    elif name == 'eta':
        return Variable(config['eta'],
                        name='eta'+suffix,
                        label = 'Jet $\eta$',
                        overflow=False, underflow=False)
    elif name == 'btag':
        return Integer(0, 2, 
                       name='btag' + suffix,
                       label = 'btagging',
                       underflow=False, overflow=False)
    elif name == 'ctag':
        return Integer(0, 2, 
                       name='ctag' + suffix,
                       label = 'ctagging',
                       underflow=False, overflow=False)
    elif name == 'genflav':
        return IntCategory([0, 4, 5],
                           name='genflav' + suffix,
                           label = 'Gen-level flavor',
                           growth=False)
    elif name == 'xi3':
        return Integer(0, config['xi3'],
                        name='xi'+suffix,
                        label = '$\\xi$',
                        overflow=False, underflow=False)
    elif name == 'phi3':
        return Integer(0, config['phi3'],
                        name ='phi'+suffix,
                        label = '$\phi$',
                        overflow=False, underflow=False)
    elif name == 'RM4':
        return Integer(0, config['RM4'],
                        name = 'RM'+suffix,
                        label = '$R_2/R_L$',
                        overflow=False, underflow=False)
    elif name == 'phi4':
        return Integer(0, config['phi4'],
                        name='phi'+suffix,
                        label = '$\phi$',
                        overflow=False, underflow=False)
    elif name == 'Zmass':
        return Variable(config['Zmass'],
                        name='Zmass'+suffix,
                        label = 'Z mass [GeV]',
                        overflow=True, underflow=True)
    elif name == 'Zpt':
        return Variable(config['Zpt'],
                        name='Zpt'+suffix,
                        label = 'Z $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == 'Zy':
        return Variable(config['Zy'],
                        name='Zy'+suffix,
                        label = 'Z $y$',
                        overflow=True, underflow=True)
    elif name == 'MUpt':
        return Variable(config['MUpt'],
                        name='MUpt'+suffix,
                        label = 'Muon $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == 'MUeta':
        return Variable(config['MUeta'],
                        name='MUeta'+suffix,
                        label = 'Muon $\eta$',
                        overflow=True, underflow=True)
    else:
        raise ValueError('Unknown axis name: %s'%name)


