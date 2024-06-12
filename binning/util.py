import numpy as np
import awkward as ak
from hist.axis import Variable, Integer, IntCategory, Regular
from hist.storage import Double
from hist import Hist
from time import time

def squash(arr):
    return ak.to_numpy(ak.flatten(arr, axis=None))

def getGenFlav(rJet, iReco, mask):
    if 'Cone' in rJet._simonjetsname:
        return ak.zeros_like(rJet.simonjets.jetPt[iReco][mask], dtype=np.int32)
    if hasattr(rJet.jets, 'hadronFlavour'):
        return ak.values_astype(rJet.jets.hadronFlavour[iReco][mask],
                                np.int32)
    else:
        return ak.zeros_like(rJet.jets.pt[iReco][mask] , dtype=np.int32)

def getBdisc(rJet, iReco, mask, config):
    if 'Cone' in rJet._simonjetsname:
        return ak.zeros_like(rJet.simonjets.jetPt[iReco][mask])
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
    if 'Cone' in rJet._simonjetsname:
        return ak.zeros_like(rJet.simonjets.jetPt[iReco][mask])
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
    if 'Cone' in rJet._simonjetsname:
        return ak.zeros_like(rJet.simonjets.jetPt[iReco][mask])
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

def getTag(rJet, iReco, mask, config):
    if 'Cone' in rJet._simonjetsname:
        zeros = ak.zeros_like(rJet.simonjets.jetPt[iReco][mask]) != 0
        return zeros
    if rJet._CHSjetsname is None:
        if hasattr(rJet.jets, 'hadronFlavour'):
            b = rJet.jets.hadronFlavour[iReco][mask] == 5
            c = rJet.jets.hadronFlavour[iReco][mask] == 4
            if config['mode'] == 'region':
                return ak.where(b, 2, ak.where(c, 1, 0))
            elif config['mode'] == 'btag':
                return b
            elif config['mode'] == 'ctag':
                return c
            else:
                raise ValueError('Unknown mode: %s'%config['mode'])
        else:
            zeros = ak.zeros_like(rJet.jets.pt[iReco][mask]) != 0
            return zeros

    CHS = rJet.CHSjets
    
    if config['mode'] == 'region':
        CvL = CHS.btagDeepFlavCvL[iReco][mask]
        CvB = CHS.btagDeepFlavCvB[iReco][mask]

        intercept = config['bregion'].intercept
        slope = config['bregion'].slope
        passB = CvL > (intercept + slope*CvB)

        ccut = config['cregion'].minCvL
        passC = CvL > ccut 

        passB = ak.max(passB, axis=-1)
        passC = ak.max(passC, axis=-1)

        region = ak.where(passB, 5, ak.where(passC, 4, 0))

        return region
    elif config['mode'] == 'btag':
        wp = vars(config['bwps'])[config['wp']]

        B = CHS.btagDeepFlavB[iReco][mask]

        passB = B > wp

        return ak.max(passB, axis=-1)
    elif config['mode'] == 'ctag':
        wp_CvL, wp_CvB = config['cwps'][config['wp']]

        CvL = CHS.btagDeepFlavCvL[iReco][mask]
        CvB = CHS.btagDeepFlavCvB[iReco][mask]

        passCvL = CvL > wp_CvL
        passCvB = CvB > wp_CvB

        return ak.max(passCvL & passCvB, axis=-1) 
    else:
        raise ValueError('Unknown mode: %s'%config['mode'])
    
def getAxis(name, config, suffix=''):
    if name == 'tag':
        return IntCategory([0, 4, 5],
                           name='tag'+suffix,
                           label = 'Tagging',
                           flow=False)
    elif name == 'pt':
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
        return IntCategory([0, 1],
                           name='btag' + suffix,
                           label = 'btagging',
                           flow=False)
    elif name == 'ctag':
        return IntCategory([0, 1], 
                           name='ctag' + suffix,
                           label = 'ctagging',
                           flow=False)
    elif name == 'genflav':
        return IntCategory([0, 4, 5],
                           name='genflav' + suffix,
                           label = 'Gen-level flavor',
                           growth=False, flow=False)
    elif name == 'dRbin3':
        return Integer(0, config['dRbin3'],
                        name='dRbin'+suffix,
                        label = '$\Delta R$ bin',
                        underflow=False, overflow=False)
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
    elif name == "dRbin4":
        return Integer(0, config['dRbin4'],
                        name='dRbin'+suffix,
                        label = '$\Delta R$ bin',
                        underflow=False, overflow=False)
    elif name == "shape4":
        return Integer(0, config['shape4'],
                        name='shape'+suffix,
                        label = 'Shape',
                        underflow=False, overflow=False)
    elif name == "r4":
        return Integer(0, config['r4'],
                        name='r'+suffix,
                        label = '$r$',
                        underflow=False, overflow=False)
    elif name == "ct4":
        return Integer(0, config['ct4'],
                       name='ct'+suffix,
                       label = '$cos(\\theta)$',
                       underflow=False, overflow=False)
    elif name == 'partPt':
        return Variable(config['partPt'],
                        name='partPt'+suffix,
                        label = 'Particle $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == 'partPtGt1':
        return Integer(0, 2,
                       name='partPtGt1'+suffix,
                       label = 'Particle $p_{T} > 1$',
                       overflow=False, underflow=False)
    elif name == 'DRaxis':
        return Variable(config['DRaxis'],
                        name='DRaxis'+suffix,
                        label = '$\Delta R$',
                        overflow=True, underflow=True)
    elif name == "partCharge":
        return Integer(0, 2,
                       name="partCharge"+suffix,
                       label = 'Charge',
                       overflow=False, underflow=False)
    elif name == "partSpecies":
        return IntCategory([0, 1, 2],
                           name='partSpecies'+suffix,
                           label = 'Species',
                           flow=False)
    elif name == "Beffpt":
        return Variable(config['Beffpt'],
                        name='pt'+suffix,
                        label = 'Jet $p_{T}$ [GeV]',
                        overflow=True, underflow=False)
    elif name == "Beffeta":
        return Variable(config['Beffeta'],
                        name='eta'+suffix,
                        label = 'Jet $\eta$',
                        overflow=False, underflow=False)
    elif name == "NJet":
        return Integer(config['NJet'][0], config['NJet'][1],
                       name='NJet'+suffix,
                       label = 'Number of jets',
                       overflow=True, underflow=True)
    elif name == "Jpt":
        return Variable(config['Jpt'],
                        name='Jpt'+suffix,
                        label = 'Jet $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == "Jeta":
        return Variable(config['Jeta'],
                        name='Jeta'+suffix,
                        label = 'Jet $\eta$',
                        overflow=False, underflow=False)
    elif name == "NumBMatch":
        return Integer(0, 3, 
                       name='NumBMatch'+suffix,
                       label = 'Number of matched AK4 CHS jets',
                       overflow=True, underflow=False)
    elif name == 'nMatch':
        return Integer(0, 4,
                       name = 'nMatch'+suffix,
                       label = 'Number of matched particles',
                       overflow=True, underflow=False)
    elif name == 'hasMatch':
        return IntCategory([0,1],
                           name='hasMatch'+suffix,
                           label = 'Has Match',
                           flow=False)
    elif name == 'fracMatched':
        return Variable(config['fracMatched'],
                        name='fracMatched'+suffix,
                        label = 'Fraction of matched particles',
                        overflow=True, underflow=True)
    elif name == 'HT':
        return Regular(config['HT'][0], config['HT'][1], config['HT'][2],
                       name='HT'+suffix,
                       label = 'Gen $H_{T}$ [GeV]',
                       overflow=True, underflow=True)
    elif name == 'rho':
        return Regular(config['rho'][0], config['rho'][1], config['rho'][2],
                       name='rho'+suffix,
                       label = 'rho',
                       overflow=True, underflow=True)
    elif name == "partPtCategory":
        return Variable(config['partPtCategory'],
                        name='partPtCategory'+suffix,
                        label = 'Particle $p_{T}$ [GeV]',
                        overflow=True, underflow=True)
    elif name == 'EM0dpt':
        return Regular(config['EM0dpt'][0], config['EM0dpt'][1], 
                       config['EM0dpt'][2],
                       name='dpt'+suffix,
                       label = '$\Delta p_{T}/p_T^{Gen}$',
                       overflow=True, underflow=True)
    elif name == 'HAD0dpt':
        return Regular(config['HAD0dpt'][0], config['HAD0dpt'][1],
                       config['HAD0dpt'][2],
                       name='dpt'+suffix,
                       label = '$\Delta p_{T}/p_T^{Gen}$',
                       overflow=True, underflow=True)
    elif name == "TRKdpt":
        return Regular(config['TRKdpt'][0], config['TRKdpt'][1],
                       config['TRKdpt'][2],
                       name='dpt'+suffix,
                        label = '$\Delta p_{T}/p_T^{Gen}$',
                       overflow=True, underflow=True)
    elif name == 'EM0deta':
        return Regular(config['EM0deta'][0], config['EM0deta'][1], 
                       config['EM0deta'][2],
                       name='deta'+suffix,
                       label = '$\Delta \eta$',
                       overflow=True, underflow=True)
    elif name == 'HAD0deta':
        return Regular(config['HAD0deta'][0], config['HAD0deta'][1],
                       config['HAD0deta'][2],
                       name='deta'+suffix,
                       label = '$\Delta \eta$',
                       overflow=True, underflow=True)
    elif name == "TRKdeta":
        return Regular(config['TRKdeta'][0], config['TRKdeta'][1],
                       config['TRKdeta'][2],
                       name='deta'+suffix,
                        label = '$\Delta \eta$',
                       overflow=True, underflow=True)
    elif name == 'EM0dphi':
        return Regular(config['EM0dphi'][0], config['EM0dphi'][1], 
                       config['EM0dphi'][2],
                       name='dphi'+suffix,
                       label = '$\Delta \phi$',
                       overflow=True, underflow=True)
    elif name == 'HAD0dphi':
        return Regular(config['HAD0dphi'][0], config['HAD0dphi'][1],
                       config['HAD0dphi'][2],
                       name='dphi'+suffix,
                       label = '$\Delta \phi$',
                       overflow=True, underflow=True)
    elif name == "TRKdphi":
        return Regular(config['TRKdphi'][0], config['TRKdphi'][1],
                       config['TRKdphi'][2],
                       name='dphi'+suffix,
                        label = '$\Delta \phi$',
                       overflow=True, underflow=True)
    elif name == "Jdphi":
        return Regular(config['Jdphi'][0], config['Jdphi'][1],
                       config['Jdphi'][2],
                       name='dphi'+suffix,
                        label = '$\Delta \phi$',
                       overflow=True, underflow=True)
    elif name == "Jdeta":
        return Regular(config['Jdeta'][0], config['Jdeta'][1],
                       config['Jdeta'][2],
                       name='deta'+suffix,
                        label = '$\Delta \eta$',
                       overflow=True, underflow=True)
    elif name == "Jdpt":
        return Regular(config['Jdpt'][0], config['Jdpt'][1],
                       config['Jdpt'][2],
                       name='dpt'+suffix,
                        label = '$\Delta p_{T}/p_T^{Gen}$',
                       overflow=True, underflow=True)
    elif name == 'nTruePU':
        return Integer(config['nTruePU'][0], config['nTruePU'][1],
                       name='nTruePU'+suffix,
                       label = 'nPU',
                       overflow=True, underflow=True)
    elif name == 'METpt':
        return Regular(config['METpt'][0], config['METpt'][1],
                       config['METpt'][2],
                       name='METpt'+suffix,
                       label = 'MET $p_{T}$ [GeV]',
                       overflow=True, underflow=True)
    elif name == 'METsig':
        return Regular(config['METsig'][0], config['METsig'][1],
                       config['METsig'][2],
                       name='METsig'+suffix,
                       label = 'MET significance',
                       overflow=True, underflow=True)
    elif name == 'nBtag':
        return Integer(config['nBtag'][0], config['nBtag'][1],
                       name='nBtag'+suffix,
                       label = 'Number of b-tagged jets',
                       overflow=True, underflow=True)
    else:
        raise ValueError('Unknown axis name: %s'%name)
