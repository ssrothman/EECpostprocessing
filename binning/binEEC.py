import numpy as np
import awkward as ak
from hist.axis import Variable, Integer
from hist.storage import Double
from hist import Hist

ptbins = [30, 50, 100, 150, 250, 500]
PUbins = [0, 20, 40, 60, 80]

def squash(arr):
    return ak.to_numpy(ak.flatten(arr, axis=None))

def getEECaxes(nDR, suffix=''):
    return [
        Variable(ptbins, name='pt'+suffix, label = 'Jet $p_{T}$ [GeV]',
                 overflow=True, underflow=True),
        Integer(0, nDR, name='dRbin'+suffix, label = '$\Delta R$ bin',
                underflow=False, overflow=False),
        Variable(PUbins, name='nPU'+suffix, label = 'Number of PU vertices',
                 overflow=True, underflow=False)
    ]

def getEECHist(nDR):
    return Hist(
        *getEECaxes(nDR),
        storage=Double()
    )

def getCovHist(nDR):
    return Hist(
        *getEECaxes(nDR, '_1'),
        *getEECaxes(nDR, '_2'),
        storage=Double()
    )

def getTransferHist(nDR):
    return Hist(
        *getEECaxes(nDR, '_Reco'),
        *getEECaxes(nDR, '_Gen'),
        storage=Double()
    )

def make_and_fill_transfer(nDR, rTransfer, rGenEEC, rGenEECUNMATCH,
                           rRecoJet, rGenJet, nPU, wt, mask):
    Htrans = getTransferHist(nDR)
    fillTransfer(Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                 rRecoJet, rGenJet, nPU, wt, mask)
    return Htrans

def fillTransfer(Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                 rRecoJet, rGenJet, nPU, wt, mask):
    proj = rTransfer.proj
    iReco = rTransfer.iReco
    iGen = rTransfer.iGen

    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return;
               
    recoPt = rRecoJet.simonjets.jetPt[iReco][mask]
    genPt = rGenJet.simonjets.jetPt[iGen][mask]

    genwt = (rGenEEC.proj - rGenEECUNMATCH.proj)[mask]
    proj = (wt*proj)[mask]

    iDRGen = ak.local_index(proj, axis=2)
    iDRReco = ak.local_index(proj, axis=3)

    recoPt, genPt, nPU, iDRGen, genwt, _ = ak.broadcast_arrays(
            recoPt, genPt, nPU, iDRGen, genwt, proj
    )

    mask2 = proj > 0

    Htrans.fill(
        pt_Reco = squash(recoPt[mask2]),
        dRbin_Reco = squash(iDRReco[mask2]),
        nPU_Reco = squash(nPU[mask2]),
        pt_Gen = squash(genPt[mask2]),
        dRbin_Gen = squash(iDRGen[mask2]),
        nPU_Gen = squash(nPU[mask2]),
        weight = squash(proj[mask2])
    )

def make_and_fill_EEC(nDR, rEEC, rJet, nPU, wt, mask):
    Hproj = getEECHist(nDR)
    Hcov = getCovHist(nDR)
    fillEEC(Hproj, Hcov, rEEC, rJet, nPU, wt, mask)
    return Hproj, Hcov

def fillEEC(Hproj, Hcov, rEEC, rJet, nPU, wt, mask):
    proj = rEEC.proj
    iReco = rEEC.iReco
    iJet = rEEC.iJet

    mask = mask[iReco]
    if(ak.sum(mask)==0):
        return;

    pt = rJet.simonjets.jetPt[iJet][mask]
    vals = (proj * wt)[mask]

    dRbin = ak.local_index(vals, axis=2)

    pt, nPU, _ = ak.broadcast_arrays(pt, nPU, dRbin)

    mask2 = vals > 0

    Hproj.fill(
        pt     = squash(pt[mask2]),
        dRbin  = squash(dRbin[mask2]),
        nPU    = squash(nPU[mask2]),
        weight = squash(vals[mask2])
    )

    Nevt = len(pt)
    extent = Hproj.axes.extent
    left = np.zeros((Nevt, *extent))

    pTindex, dRindex, PUindex = Hproj.axes.index(
            squash(pt[mask2]), 
            squash(dRbin[mask2]), 
            squash(nPU[mask2])
    )
    evt = ak.local_index(pt, axis=0)
    evt, _ = ak.broadcast_arrays(evt, pt)
    evt = squash(evt[mask2])

    if Hproj.axes['nPU'].traits.underflow:
        print("incrememting PUindex")
        PUindex += 1
    if Hproj.axes['pt'].traits.underflow:
        print("incrememting pTindex")
        pTindex += 1
    if Hproj.axes['dRbin'].traits.underflow:
        print("incrememting dRindex")
        dRindex += 1

    indextuple = (squash(evt), squash(pTindex), 
                  squash(dRindex), squash(PUindex))
    
    #there's a little bit of floating point error
    #probably because I am forcing the sum order
    #I think NBD though
    np.add.at(left, indextuple, squash(vals[mask2]))

    print('sum left', np.sum(left))
    print('should be', np.sum(vals[mask2]))

    cov = np.einsum('ijkl,iabc->jklabc', left, left)
    Hcov += cov #I love that this just works!

def binAll(readers, nDR, mask, wt):
    Htrans = make_and_fill_transfer(
            nDR, readers.rTransfer, readers.rGenEEC, readers.rGenEECUNMATCH, 
            readers.rRecoJet, readers.rGenJet, readers.nPU, wt, mask) 
    Hreco, HcovReco = make_and_fill_EEC(
            nDR, readers.rRecoEEC, readers.rRecoJet, 
            readers.nPU, wt, mask)
    HrecoUNMATCH, HcovRecoUNMATCH = make_and_fill_EEC(
            nDR, readers.rRecoEECUNMATCH, readers.rRecoJet,
            readers.nPU, wt, mask)
    Hgen, HcovGen = make_and_fill_EEC(
            nDR, readers.rGenEEC, readers.rGenJet,
            readers.nPU, wt, mask)
    HgenUNMATCH, HcovGenUNMATCH = make_and_fill_EEC(
            nDR, readers.rGenEECUNMATCH, readers.rGenJet,
            readers.nPU, wt, mask)

    return {
        'Htrans': Htrans,
        'Hreco': Hreco,
        'HcovReco': HcovReco,
        'HrecoUNMATCH': HrecoUNMATCH,
        'HcovRecoUNMATCH': HcovRecoUNMATCH,
        'Hgen': Hgen,
        'HcovGen': HcovGen,
        'HgenUNMATCH': HgenUNMATCH,
        'HcovGenUNMATCH': HcovGenUNMATCH
    }
