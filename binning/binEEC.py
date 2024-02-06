import numpy as np
import awkward as ak
from hist.axis import Variable, Integer
from hist.storage import Double
from hist import Hist

#https://cds.cern.ch/record/2767703/files/SMP-19-009-pas.pdf
#first jet pT from above record
#ptbins = [30, 47, 69, 96, 128, 166, 210, 261, 319,
#          386, 460, 544, 638, 460, 544, 638, 751, 870, 1500]
#https://www.hepdata.net/record/ins1920187
#Extra Figure u4cL[Z] (vs.PT)
#or even better, in the AN
#https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2019/098
#ptbins = [50, 65, 88, 120, 150, 186, 254, 326, 408, 1500]
#Meng et al
#ptbins = [30, 97, 220, 330, 468, 638, 846, 1101, 1410, 1784, 20000]

def squash(arr):
    return ak.to_numpy(ak.flatten(arr, axis=None))

class EECbinner:
    def __init__(self, config, config_btag, config_ctag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = config.bins
        self._config['skipTrans'] = config.skipTransfer
        self._config['diagTrans'] = config.diagTransfer

        self._config['btag'] = config_btag
        self._config['ctag'] = config_ctag

    def _passBtag(self, rJet, iReco, mask):
        if rJet._CHSjetsname is None:
            return rJet.jets.hadronFlavour[iReco][mask] == 5
        
        CHS = rJet.CHSjets
        if self._config['btag'].algo == 'deepjet':
            btag = CHS.btagDeepFlavB[iReco][mask]
        elif self._config['btag'].algo == 'deepcsv':
            btag = CHS.btagDeepB[iReco][mask]
        else:
            raise NotImplementedError("deepjet and deepcsv are the only available btagging algos")

        blabel = self._config['btag'].wp
        if blabel == 'tight':
            bwp = self._config['btag'].WPcuts.tight
        elif blabel == 'medium':
            bwp = self._config['btag'].WPcuts.medium
        elif blabel == 'loose':
            bwp = self._config['btag'].WPcuts.loose
        else:
            raise NotImplementedError("WP needs to be 'loose', 'medium', or 'tight'")

        return btag > bwp
        

    def _getAxis(self, name, suffix=''):
        if name == 'pt':
            return Variable(self._config['bins'].pt, 
                            name='pt'+suffix,
                            label = 'Jet $p_{T}$ [GeV]',
                            overflow=True, underflow=True)
        elif name == 'dRbin':
            return Integer(0, self._config['bins'].dRbin, 
                           name='dRbin'+suffix,
                           label = '$\Delta R$ bin',
                           underflow=False, overflow=False)
        elif name == 'nPU':
            return Variable(self._config['bins'].nPU, 
                            name='nPU'+suffix,
                            label='Number of PU vertices',
                            overflow=True, underflow=False)
        elif name == 'order':
            return Integer(2, self._config['bins'].order+1, 
                           name='order'+suffix, 
                           label = 'EEC Order',
                           underflow=False, overflow=False)
        elif name == 'eta':
            return Variable(self._config['bins'].eta,
                            name='eta'+suffix,
                            label = 'Jet $\eta$',
                            overflow=False, underflow=True)
        elif name == 'btag':
            return Integer(0, 2, 
                           name='btag' + suffix,
                           label = 'btagging',
                           underflow=False, overflow=False)
        else:
            raise ValueError('Unknown axis name: %s'%name)

    def _getEECaxes(self, suffix='', transfer=False):
        axes = []
        for axis in self._config['axes']:
            if transfer and (getattr(self._config['skipTrans'],axis)
                             or getattr(self._config['diagTrans'],axis)):
                continue
            axes.append(self._getAxis(axis,suffix))
        return axes

    def _getTransferDiagAxes(self):
        axes = []
        for axis in self._config['axes']:
            if getattr(self._config['diagTrans'],axis):
                axes.append(self._getAxis(axis))
        return axes
    
    def _getEECHist(self):
        return Hist(
            *self._getEECaxes(),
            storage=Double()
        )

    def _getCovHist(self):
        return Hist(
            *self._getEECaxes('_1'),
            *self._getEECaxes('_2'),
            storage=Double()
        )

    def _getTransferHist(self):
        return Hist(
            *self._getEECaxes('_Reco', transfer=True),
            *self._getEECaxes('_Gen', transfer=True),
            *self._getTransferDiagAxes(),
            storage=Double()
        )

    def _make_and_fill_transfer(self, rTransfer, rGenEEC, rGenEECUNMATCH,
                               rRecoJet, rGenJet, nPU, wt, mask):
        Htrans = self._getTransferHist()
        self._fillTransfer(Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                     rRecoJet, rGenJet, nPU, wt, mask)
        return Htrans

    def _fillTransfer(self, Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                     rRecoJet, rGenJet, nPU_o, wt, mask):
        iReco = rTransfer.iReco
        iGen = rTransfer.iGen

        mask = mask[iReco]
        if(ak.sum(mask)==0):
            return;

        recoPt_o = rRecoJet.simonjets.jetPt[iReco][mask]
        genPt_o = rGenJet.simonjets.jetPt[iGen][mask]

        pass_btag_o  = self._passBtag(rRecoJet, iReco, mask)
        genflav_o = rRecoJet.CHSjets.hadronFlavour[iReco][mask]

        maxorder = self._config['bins'].order
        for order in range(2, maxorder+1):
            proj = rTransfer.proj(order)
                       
            genwt = (rGenEEC.proj(order) - rGenEECUNMATCH.proj(order))[mask]
            proj = (wt*proj)[mask]

            iDRGen = ak.local_index(proj, axis=2)
            iDRReco = ak.local_index(proj, axis=3)

            recoPt, genPt, pass_btag, genflav, nPU, iDRGen, genwt, _ = \
                ak.broadcast_arrays(recoPt_o, genPt_o, pass_btag_o, 
                                    genflav_o, nPU_o, iDRGen, genwt, proj)

            mask2 = proj > 0

            fills = {}
            if 'pt' in self._config['axes'] and not self._config['skipTrans'].pt:
                if self._config['diagTrans'].pt:
                    fills['pt'] = squash(recoPt[mask2])
                else:
                    fills['pt_Reco'] = squash(recoPt[mask2])
                    fills['pt_Gen'] = squash(genPt[mask2])
            if 'dRbin' in self._config['axes'] and not self._config['skipTrans'].dRbin:
                if self._config['diagTrans'].dRbin:
                    fills['dRbin'] = squash(iDRReco[mask2])
                else:
                    fills['dRbin_Reco'] = squash(iDRReco[mask2])
                    fills['dRbin_Gen'] = squash(iDRGen[mask2])
            if 'nPU' in self._config['axes'] and not self._config['skipTrans'].nPU:
                if self._config['diagTrans'].nPU:
                    fills['nPU'] = squash(nPU[mask2])
                else:
                    fills['nPU_Reco'] = squash(nPU[mask2])
                    fills['nPU_Gen'] = squash(nPU[mask2])
            if 'order' in self._config['axes'] and not self._config['skipTrans'].order:
                if self._config['diagTrans'].order:
                    fills['order'] = order
                else:
                    raise RuntimeError("Cannot transfer along order axis")
            if 'btag' in self._config['axes'] and not self._config['skipTrans'].order:
                if self._config['diagTrans'].btag:
                    fills['btag'] = squash(pass_btag[mask2])
                else:
                    fills['btag_Reco'] = squash(pass_btag[mask2])
                    fills['btag_Gen'] = squash(genflav[mask2] == 5)

            Htrans.fill(
                **fills,
                weight = squash(proj[mask2])
            )

    def _make_and_fill_EEC(self, rEEC, rJet, nPU, wt, mask):
        Hproj = self._getEECHist()
        Hcov = self._getCovHist()
        self._fillEEC(Hproj, Hcov, rEEC, rJet, nPU, wt, mask)
        return Hproj, Hcov

    def _fillEEC(self, Hproj, Hcov, rEEC, rJet, nPU, wt, mask):
        maxorder = self._config['bins'].order
        projs = [rEEC.proj(order) for order in range(2, maxorder+1)]
        nums = [ak.num(proj, axis=-1) for proj in projs]
        projs = ak.concatenate(projs, axis=-1)
        nums = ak.concatenate(nums, axis=-1)
        projs = ak.unflatten(projs, squash(nums), axis=-1)
        #projs has shape [event, jet, order, DR]

        iReco = rEEC.iReco
        iJet = rEEC.iJet

        mask = mask[iReco]
        if(ak.sum(mask)==0):
            return;

        pt = rJet.simonjets.jetPt[iJet][mask]
        vals = (projs * wt)[mask]

        pass_btag  = self._passBtag(rJet, iJet, mask)

        order = ak.local_index(vals, axis=2)+2
        dRbin = ak.local_index(vals, axis=3)

        pt, pass_btag, nPU, order, _ = ak.broadcast_arrays(
                pt, pass_btag, nPU, order, dRbin)

        mask2 = vals > 0

        projfills = {}
        if 'pt' in self._config['axes']:
            projfills['pt'] = squash(pt[mask2])
        if 'dRbin' in self._config['axes']:
            projfills['dRbin'] = squash(dRbin[mask2])
        if 'nPU' in self._config['axes']:
            projfills['nPU'] = squash(nPU[mask2])
        if 'order' in self._config['axes']:
            projfills['order'] = squash(order[mask2])
        if 'btag' in self._config['axes']:
            projfills['btag'] = squash(pass_btag[mask2])

        Hproj.fill(
            **projfills,
            weight = squash(vals[mask2])
        )

        Nevt = len(pt)
        extent = Hproj.axes.extent
        left = np.zeros((Nevt, *extent))

        indexargs = []
        for ax in Hproj.axes:
            indexargs.append(projfills[ax.name])

        indices = list(Hproj.axes.index(*indexargs))
        for i, ax in enumerate(Hproj.axes):
            if ax.traits.underflow:
                indices[i] += 1

        evt = ak.local_index(pt, axis=0)
        evt, _ = ak.broadcast_arrays(evt, pt)
        evt = squash(evt[mask2])

        indextuple = (squash(evt), *indices)
        
        #there's a little bit of floating point error
        #probably because I am forcing the sum order
        #I think NBD though
        np.add.at(left, indextuple, squash(vals[mask2]))

        leftis = [0] + [i+1 for i in range(len(Hproj.axes))]
        rightis = [0] + [i+11 for i in range(len(Hproj.axes))]
        ansis = leftis[1:] + rightis[1:]
        cov = np.einsum(left, leftis, left, rightis, ansis, optimize=True)
        Hcov += cov #I love that this just works!

    def binAll(self, readers, mask, wt):
        Htrans = self._make_and_fill_transfer(
                readers.rTransfer, readers.rGenEEC, readers.rGenEECUNMATCH, 
                readers.rRecoJet, readers.rGenJet, readers.nPU, wt, mask) 
        Hreco, HcovReco = self._make_and_fill_EEC(
                readers.rRecoEEC, readers.rRecoJet, 
                readers.nPU, wt, mask)
        HrecoUNMATCH, HcovRecoUNMATCH = self._make_and_fill_EEC(
                readers.rRecoEECUNMATCH, readers.rRecoJet,
                readers.nPU, wt, mask)
        Hgen, HcovGen = self._make_and_fill_EEC(
                readers.rGenEEC, readers.rGenJet,
                readers.nPU, wt, mask)
        HgenUNMATCH, HcovGenUNMATCH = self._make_and_fill_EEC(
                readers.rGenEECUNMATCH, readers.rGenJet,
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
