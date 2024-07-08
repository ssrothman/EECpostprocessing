import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def getScaleWts7pt(weights, readers):
    '''
    LHE scale variation weights (w_var / w_nominal); 
    [0] is MUF="0.5" MUR="0.5"; 
    [1] is MUF="1.0" MUR="0.5"; 
    [2] is MUF="2.0" MUR="0.5"; 
    [3] is MUF="0.5" MUR="1.0"; 
    [4] is MUF="1.0" MUR="1.0"; 
    [5] is MUF="2.0" MUR="1.0"; 
    [6] is MUF="0.5" MUR="2.0"; 
    [7] is MUF="1.0" MUR="2.0"; 
    [8] is MUF="2.0" MUR="2.0"
    '''
    var_weights = readers.scalewt

    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)
 
    up = np.maximum.reduce([var_weights[:,0],
                            var_weights[:,1],
                            var_weights[:,3],
                            var_weights[:,5],
                            var_weights[:,7],
                            var_weights[:,8]])

    down = np.minimum.reduce([var_weights[:,0],
                              var_weights[:,1],
                              var_weights[:,3],
                              var_weights[:,5],
                              var_weights[:,7],
                              var_weights[:,8]])

    weights.add('wt_scale', nom, up, down)

def getScaleWts3pt(weights, readers):
    var_weights = readers.scalewt

    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)

    up = np.maximum(var_weights[:,0], var_weights[:,8])
    down = np.minimum(var_weights[:,0], var_weights[:,8])

    weights.add('wt_scale_3pt', nom, up, down)

def getPSWts(weights, readers):
    ps_weights = readers.psweight
    if ak.num(ps_weights)[0] < 4:
        return

    nweights = len(weights.weight())

    nom  = np.ones(nweights)

    up_isr   = np.ones(nweights)
    down_isr = np.ones(nweights)

    up_fsr   = np.ones(nweights)
    down_fsr = np.ones(nweights)

    up_isr = ps_weights[:,0]
    down_isr = ps_weights[:,2]

    up_fsr = ps_weights[:,1]
    down_fsr = ps_weights[:,3]
        
    weights.add('wt_ISR', nom, up_isr, down_isr)
    weights.add('wt_FSR', nom, up_fsr, down_fsr)

def getPDFweights(weights, readers):
    nweights = len(weights.weight())
    pdf_weights = readers.pdfwt

    nom   = np.ones(nweights)

    # Hessian PDF weights
    # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
    arg = pdf_weights[:,1:-2]-np.ones((nweights,100))
    summed = ak.sum(np.square(arg),axis=1)
    pdf_unc = np.sqrt( (1./99.) * summed )
    pdf_up = pdf_unc + nom
    pdf_dn = nom - pdf_unc
    weights.add('wt_PDF', nom, pdf_up, pdf_dn)
    # alpha_S weights
    # Eq. 27 of same ref
    as_unc = 0.5*(pdf_weights[:,102] - pdf_weights[:,101])
    weights.add('wt_aS', nom, as_unc + nom, nom - as_unc)

    # PDF + alpha_S weights
    # Eq. 28 of same ref
    pdfas_unc = np.sqrt( np.square(pdf_unc) + np.square(as_unc) )
    weights.add('wt_PDFaS', nom, pdfas_unc + nom, nom - pdfas_unc) 


def getAllTheorySFs(weights, readers):
    weights.add('generator', readers.genwt)

    getScaleWts7pt(weights, readers)
    getScaleWts3pt(weights, readers)
    getPDFweights(weights, readers)
    getPSWts(weights, readers)

