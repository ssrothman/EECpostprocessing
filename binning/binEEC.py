import numpy as np
import awkward as ak
from hist.axis import Variable, Integer, IntCategory
from hist.storage import Double
from hist import Hist
from time import time

from .util import *

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

class EECbinner:
    def __init__(self, config_bin, config_tag, config_cjet):
        self._config = {}
        self._config['axes'] = config_bin.axes
        self._config['bins'] = vars(config_bin.bins)
        self._config['skipTrans'] = vars(config_bin.skipTransfer)
        self._config['diagTrans'] = vars(config_bin.diagTransfer)
        self._config['tag'] = vars(config_tag)
        self._config['controlJetSelection'] = vars(config_cjet)

        self.isMC = False;
    
    def _getEECaxes(self, suffix='', transfer=False):
        axes = []
        for axis in self._config['axes']:
            if transfer and (self._config['skipTrans'][axis]
                             or self._config['diagTrans'][axis]):
                continue
            axes.append(getAxis(axis, self._config['bins'], suffix))
        return axes

    def _getTransferDiagAxes(self):
        axes = []
        for axis in self._config['axes']:
            if self._config['diagTrans'][axis] and not self._config['skipTrans'][axis]:
                axes.append(getAxis(axis, self._config['bins']))
        return axes
    
    def _getEECHist(self):
        return Hist(
            *self._getEECaxes(),
            getAxis('dRbin', self._config['bins']),
            getAxis('order', self._config['bins']),
            storage=Double()
        )

    def _getRes3Hist(self):
        return Hist(
            *self._getEECaxes(),
            getAxis("dRbin3",self._config['bins']),
            getAxis('xi3', self._config['bins']),
            getAxis('phi3', self._config['bins']),
            storage=Double()
        )

    def getRes4Hist(self):
        return Hist(
            *self._getEECaxes(),
            getAxis("shape4",self._config['bins']),
            getAxis("dRbin4",self._config['bins']),
            getAxis("r4",self._config['bins']),
            getAxis("ct4",self._config['bins']),
            storage=Double()
        )

    def _getCovHist(self):
        return Hist(
            *self._getEECaxes('_1'),
            getAxis('dRbin', self._config['bins'], '_1'),
            getAxis('order', self._config['bins'], '_1'),
            *self._getEECaxes('_2'),
            getAxis('dRbin', self._config['bins'],'_2'),
            getAxis('order', self._config['bins'],'_2'),
            storage=Double()
        )

    def _getCovRes3Hist(self):
        return Hist(
            *self._getEECaxes('_1'),
            getAxis("dRbin3",self._config['bins'], '_1'),
            getAxis('xi3', self._config['bins'],'_1'),
            getAxis('phi3', self._config['bins'],'_1'),
            *self._getEECaxes('_2'),
            getAxis("dRbin3",self._config['bins'], '_2'),
            getAxis('xi3', self._config['bins'],'_2'),
            getAxis('phi3', self._config['bins'],'_2'),
            storage=Double()
        )

    def _getCovRes4Hist(self):
        return Hist(
            *self._getEECaxes('_1'),
            getAxis("shape4",self._config['bins'], '_1'),
            getAxis("dRbin4",self._config['bins'], '_1'),
            getAxis("r4",self._config['bins'], '_1'),
            getAxis("ct4",self._config['bins'], '_1'),
            *self._getEECaxes('_2'),
            getAxis("shape4",self._config['bins'], '_2'),
            getAxis("dRbin4",self._config['bins'], '_2'),
            getAxis("r4",self._config['bins'], '_2'),
            getAxis("ct4",self._config['bins'], '_2'),
            storage=Double()
        )

    def _getTransferHist(self):
        return Hist(
            *self._getEECaxes('_Reco', transfer=True),
            getAxis('dRbin', self._config['bins'], '_Reco'),
            *self._getEECaxes('_Gen', transfer=True),
            getAxis('dRbin', self._config['bins'], '_Gen'),
            *self._getTransferDiagAxes(),
            getAxis('order', self._config['bins']),
            storage=Double()
        )

    def _getTransferRes3Hist(self):
        return Hist(
            *self._getEECaxes('_Reco', transfer=True),
            getAxis("dRbin3",self._config['bins'], '_Reco'),
            getAxis("xi3", self._config['bins'], "_Reco"),
            getAxis("phi3", self._config['bins'], "_Reco"),
            *self._getEECaxes('_Gen', transfer=True),
            getAxis("dRbin3",self._config['bins'], '_Gen'),
            getAxis("xi3", self._config['bins'], "_Gen"),
            getAxis("phi3", self._config['bins'], "_Gen"),
            *self._getTransferDiagAxes(),
            storage=Double()
        )

    def _getTransferRes4Hist(self):
        return Hist(
            *self._getEECaxes('_Reco', transfer=True),
            getAxis("shape4",self._config['bins'], '_Reco'),
            getAxis("dRbin4",self._config['bins'], '_Reco'),
            getAxis("r4",self._config['bins'], '_Reco'),
            getAxis("ct4",self._config['bins'], '_Reco'),
            *self._getEECaxes('_Gen', transfer=True),
            getAxis("shape4",self._config['bins'], '_Gen'),
            getAxis("dRbin4",self._config['bins'], '_Gen'),
            getAxis("r4",self._config['bins'], '_Gen'),
            getAxis("ct4",self._config['bins'], '_Gen'),
            *self._getTransferDiagAxes(),
            storage=Double()
        )

    def _make_and_fill_transfer(self, rTransfer, rGenEEC, rGenEECUNMATCH,
                               rRecoJet, rGenJet, nPU, wt, mask,
                                which='proj'):
        if which=='proj':
            Htrans = self._getTransferHist()
        elif which=='res3':
            Htrans = self._getTransferRes3Hist()
        elif which=='res4':
            Htrans = self._getTransferRes4Hist()
        else:
            raise ValueError("Invalid which %s"%which)

        self._fillTransfer(Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                     rRecoJet, rGenJet, nPU, wt, mask, which)
        return Htrans

    def _fillTransfer(self, Htrans, rTransfer, rGenEEC, rGenEECUNMATCH,
                     rRecoJet, rGenJet, nPU_o, wt, mask,
                      which='proj'):
        iReco = rTransfer.iReco
        iGen = rTransfer.iGen

        mask = mask[iReco]
        if(ak.sum(mask)==0):
            return;
        
        if 'pt' in self._config['axes']:
            recoPt_o = rRecoJet.jets.corrpt[iReco][mask]
            genPt_o = rGenJet.simonjets.jetPt[iGen][mask]

        if 'eta' in self._config['axes']:
            recoEta_o = rRecoJet.simonjets.jetEta[iReco][mask]
            genEta_o = rGenJet.simonjets.jetEta[iGen][mask]

        if 'btag' in self._config['axes']:
            btag_o = getTag(rRecoJet, iReco, mask, self._config['tag'])
            genflav_o = getGenFlav(rGenJet, iGen, mask) == 5
        elif 'ctag' in self._config['axes']:
            raise NotImplementedError
        elif 'tag' in self._config['axes']:
            raise NotImplementedError
        
        if which == 'proj':
            maxorder = self._config['bins']['order']
        else:
            maxorder = 2
        for order in range(2, maxorder+1):
            t0 = time()

            if which == 'proj':
                vals = wt * rTransfer.proj(order)[mask]
                #genwt = wts * (rGenEEC.proj(order) - rGenEECUNMATCH.proj(order))[mask]

                iDRGen = ak.local_index(vals, axis=2)
                iDRReco = ak.local_index(vals, axis=3)
            else:
                vals = rTransfer.res3[mask] * wt
                #genwt = (rGenEEC.res3 - rGenEECUNMATCH.res3)[mask]

                iDRGen = ak.local_index(vals, axis=2)
                iXIGen = ak.local_index(vals, axis=3)
                iPHIGen = ak.local_index(vals, axis=4)
                iDRReco = ak.local_index(vals, axis=5)
                iXIReco = ak.local_index(vals, axis=6)
                iPHIReco = ak.local_index(vals, axis=7)

            mask2 = vals > 0

            fills = {}
            if which=='proj':
                order, iDRGen, iDRReco, _ = ak.broadcast_arrays(order, iDRGen, iDRReco, vals)
                fills['order'] = squash(order[mask2])
                fills['dRbin_Reco'] = squash(iDRReco[mask2])
                fills['dRbin_Gen'] = squash(iDRGen[mask2])
            else:
                iDRGen, iXIGen, iPHIGen, iDRReco, iXIReco, iPHIReco, _ = ak.broadcast_arrays(
                        iDRGen, iXIGen, iPHIGen, iDRReco, iXIReco, iPHIReco, vals)
                fills['dRbin_Reco'] = squash(iDRReco[mask2])
                fills['dRbin_Gen'] = squash(iDRGen[mask2])
                fills['xi_Reco'] = squash(iXIReco[mask2])
                fills['xi_Gen'] = squash(iXIGen[mask2])
                fills['phi_Reco'] = squash(iPHIReco[mask2])
                fills['phi_Gen'] = squash(iPHIGen[mask2])

            if 'btag' in self._config['axes'] and not self._config['skipTrans']['btag']:
                btag, genflav, _ = ak.broadcast_arrays(btag_o, 
                                                       genflav_o, 
                                                       vals)
                if self._config['diagTrans']['btag']:
                    fills['btag'] = squash(btag[mask2])
                else:
                    fills['btag_Reco'] = squash(btag[mask2])
                    fills['btag_Gen'] = squash(genflav[mask2])

            if 'pt' in self._config['axes'] and not self._config['skipTrans']['pt']:
                recoPt, genPt, _ = ak.broadcast_arrays(recoPt_o, 
                                                       genPt_o, 
                                                       vals)
                if self._config['diagTrans']['pt']:
                    fills['pt'] = squash(recoPt[mask2])
                else:
                    fills['pt_Reco'] = squash(recoPt[mask2])
                    fills['pt_Gen'] = squash(genPt[mask2])
            if 'nPU' in self._config['axes'] and not self._config['skipTrans']['nPU']:
                nPU, _ = ak.broadcast_arrays(nPU_o, vals)
                if self._config['diagTrans']['nPU']:
                    fills['nPU'] = squash(nPU[mask2])
                else:
                    fills['nPU_Reco'] = squash(nPU[mask2])
                    fills['nPU_Gen'] = squash(nPU[mask2])
            if 'eta' in self._config['axes'] and not self._config['skipTrans']['eta']:
                recoEta, genEta, _ = ak.broadcast_arrays(recoEta_o,
                                                         genEta_o,
                                                         vals)
                if self._config['diagTrans']['eta']:
                    fills['eta'] = np.abs(squash(recoEta[mask2]))
                else:
                    fills['eta_Reco'] = np.abs(squash(recoEta[mask2]))
                    fills['eta_Gen'] = np.abs(squash(genEta[mask2]))

            #print("transfer %d setup took %0.3f seconds" % (order, time()-t0))
            t0 = time()

            Htrans.fill(
                **fills,
                weight = squash(vals[mask2])
            )
            #print("transfer %d fill took %0.3f seconds" % (order, time()-t0))

    def _make_and_fill_EEC(self, rEEC, rJet, nPU, wt, mask, subtract=None,
                           which = 'proj',
                           rControlEEC=None, rControlJet=None):

        if which == 'proj':
            Hproj = self._getEECHist()
            Hcov = self._getCovHist()
        elif which == 'res3':
            Hproj = self._getRes3Hist()
            Hcov = self._getCovRes3Hist()
        elif which == 'res4':
            Hproj = self.getRes4Hist()
            Hcov = self._getCovRes4Hist()

        if rControlEEC is not None:
            if which == 'proj':
                Hcontrol = self._getEECHist()
                HcovControl = self._getCovHist()
            else:
                raise NotImplementedError
        else:
            Hcontrol = None
            HcovControl = None

        self._fillEEC(Hproj, Hcov, rEEC, rJet, nPU, wt, mask, subtract, which,
                      Hcontrol, HcovControl, rControlEEC, rControlJet)

        if rControlEEC is not None:
            return Hproj, Hcov, Hcontrol, HcovControl
        else:
            return Hproj, Hcov

    def _fillEEC(self, Hproj, Hcov, rEEC, rJet, nPU, wt, mask, subtract=None,
                 which = 'proj', 
                 Hcontrol=None, HcovControl=None, 
                 rControlEEC=None, rControlJet=None):
        t0 = time()
        maxorder = self._config['bins']['order']

        iReco = rEEC.iReco
        iJet = rEEC.iJet

        if which == 'proj':
            projs = [rEEC.proj(order) for order in range(2, maxorder+1)]

            if subtract is not None:
                for order in range(2, maxorder+1):
                    projs[order-2] = projs[order-2] - subtract.proj(order)

            nums = [ak.num(proj, axis=-1) for proj in projs]
            projs = ak.concatenate(projs, axis=-1)
            nums = ak.concatenate(nums, axis=-1)
            projs = ak.unflatten(projs, squash(nums), axis=-1)
            #projs has shape [event, jet, order, DR]

            if Hcontrol is not None:
                cprojs = [ak.firsts(rControlEEC.proj(order)) for order in range(2, maxorder+1)]
                controlPt = rControlJet.simonjets.jetPt
                controlEta = np.abs(rControlJet.simonjets.jetEta)

                cmask = (controlPt < self._config['controlJetSelection']['maxPt']) \
                        & (controlEta < self._config['controlJetSelection']['maxEta'])

                controlPt = ak.firsts(controlPt[cmask])
                jetPt = rJet.jets.corrpt

                factor = (controlPt / jetPt)[iJet]
                factor = ak.ones_like(factor)
                cRescale = factor * factor
                for order in range(2, maxorder+1):
                    cprojs[order-2] = cprojs[order-2][:,None,:] * cRescale
                    cRescale = cRescale * factor

                cnums = [ak.num(cproj, axis=-1) for cproj in cprojs]
                cprojs = ak.concatenate(cprojs, axis=-1)
                cnums = ak.concatenate(cnums, axis=-1)
                cprojs = ak.unflatten(cprojs, squash(cnums), axis=-1)
                #cprojs has shape [event, jet, order, DR]

        elif which == 'res3':
            projs = rEEC.res3

            if subtract is not None:
                projs = projs - subtract.res3

            if Hcontrol is not None:
                raise NotImplementedError

        elif which == 'res4':
            projs = rEEC.res4

            if subtract is not None:
                projs = projs - subtract.res4

            if Hcontrol is not None:
                raise NotImplementedError

        else:
            raise ValueError("Invalid which %s"%which)

        mask = mask[iReco]
        if(ak.sum(mask)==0):
            return;

        vals = (projs * wt)[mask]

        if which == 'proj':
            order = ak.local_index(vals, axis=2)
            dRbin = ak.local_index(vals, axis=3)

            if Hcontrol is not None:
                cvals = (cprojs * wt)[mask]
                corder = ak.local_index(cvals, axis=2)
                cdRbin = ak.local_index(cvals, axis=3)

        elif which == 'res3':
            RL = ak.local_index(vals, axis=2)
            xi3 = ak.local_index(vals, axis=3)
            phi3 = ak.local_index(vals, axis=4)

            if Hcontrol is not None:
                raise NotImplementedError

        elif which == 'res4':
            shape = ak.local_index(vals, axis=2)
            dRbin = ak.local_index(vals, axis=3)
            r = ak.local_index(vals, axis=4)
            ct = ak.local_index(vals, axis=5)

        mask2 = vals > 0

        projfills = {}

        if Hcontrol is not None:
            cmask2 = cvals > 0
            cprojfills = {}

        if which == 'proj':
            order, dRbin, _ = ak.broadcast_arrays(order, dRbin, vals)
            projfills['dRbin'] = squash(dRbin[mask2])
            projfills['order'] = squash(order[mask2])

            if Hcontrol is not None:
                corder, cdRbin, _ = ak.broadcast_arrays(corder, cdRbin, cvals)
                cprojfills['dRbin'] = squash(cdRbin[cmask2])
                cprojfills['order'] = squash(corder[cmask2])

        elif which =='res3':
            RL, xi3, phi3, _ = ak.broadcast_arrays(RL, xi3, phi3, vals)
            projfills['dRbin'] = squash(RL[mask2])
            projfills['xi'] = squash(xi3[mask2])
            projfills['phi'] = squash(phi3[mask2])

            if Hcontrol is not None:
                raise NotImplementedError

        elif which == 'res4':
            shape, dRbin, r, ct, _ = ak.broadcast_arrays(shape, dRbin, 
                                                         r, ct, vals)
            projfills['shape'] = squash(shape[mask2])
            projfills['dRbin'] = squash(dRbin[mask2])
            projfills['r'] = squash(r[mask2])
            projfills['ct'] = squash(ct[mask2])

            if Hcontrol is not None:
                raise NotImplementedError

        if 'btag' in self._config['axes']:
            btag = getTag(rJet, iJet, mask, self._config['tag'])
            btag, _ = ak.broadcast_arrays(btag, vals)
            projfills['btag'] = squash(btag[mask2])

            if Hcontrol is not None:
                cbtag = getTag(rJet, iJet, mask, self._config['tag'])
                cbtag, _ = ak.broadcast_arrays(cbtag, cvals)
                cprojfills['btag'] = squash(cbtag[cmask2])

        if 'pt' in self._config['axes']:
            pt = rJet.jets.corrpt[iJet][mask]
            pt, _ = ak.broadcast_arrays(pt, vals)
            projfills['pt'] = Hproj.axes['pt'].index(squash(pt[mask2]))

            if Hcontrol is not None:
                cpt = rJet.jets.corrpt[iJet][mask]
                cpt, _ = ak.broadcast_arrays(cpt, cvals)
                cprojfills['pt'] = Hcontrol.axes['pt'].index(squash(cpt[cmask2]))

        if 'nPU' in self._config['axes']:
            nPU, _ = ak.broadcast_arrays(nPU, vals)
            projfills['nPU'] = Hproj.axes['nPU'].index(squash(nPU[mask2]))

            if Hcontrol is not None:
                cnPU, _ = ak.broadcast_arrays(nPU, cvals)
                cprojfills['nPU'] = Hcontrol.axes['nPU'].index(squash(cnPU[cmask2]))

        if 'tag' in self._config['axes']:
            region = tagRegion(rJet, iJet, mask, self._config['tag'])
            region, _ = ak.broadcast_arrays(region, vals)
            projfills['tag'] = squash(region[mask2])

            if Hcontrol is not None:
                cregion = tagRegion(rJet, iJet, mask, self._config['tag'])
                cregion, _ = ak.broadcast_arrays(cregion, cvals)
                cprojfills['tag'] = squash(cregion[cmask2])

        if 'genflav' in self._config['axes']:
            genflav = getGenFlav(rJet, iJet, mask)
            genflav, _ = ak.broadcast_arrays(genflav, vals)
            projfills['genflav'] = Hproj.axes['genflav'].index(squash(genflav[mask2]))

            if Hcontrol is not None:
                cgenflav = getGenFlav(rJet, iJet, mask)
                cgenflav, _ = ak.broadcast_arrays(cgenflav, cvals)
                cprojfills['genflav'] = Hcontrol.axes['genflav'].index(squash(cgenflav[cmask2]))

        if 'eta' in self._config['axes']:
            eta = rJet.simonjets.jetEta[iJet][mask]
            eta, _ = ak.broadcast_arrays(eta, vals)
            projfills['eta'] = Hproj.axes['eta'].index(np.abs(squash(eta[mask2])))

            if Hcontrol is not None:
                ceta = rJet.simonjets.jetEta[iJet][mask]
                ceta, _ = ak.broadcast_arrays(ceta, cvals)
                cprojfills['eta'] = Hcontrol.axes['eta'].index(np.abs(squash(ceta[cmask2])))

        #print("setup took %0.3f seconds" % (time()-t0))
        t0 = time()
        for ax in projfills:
            if Hproj.axes[ax].traits.underflow:
                projfills[ax] += 1
            projfills[ax] = projfills[ax].astype(np.int32)
            print(ax, ak.type(projfills[ax]))
            print("\t", ak.min(projfills[ax]), ak.max(projfills[ax]))
    
        print(Hproj.view(flow=True).shape)
        projfill_tuple = tuple([projfills[ax.name] for ax in Hproj.axes])
        #print(projfill_tuple)
        
        np.add.at(Hproj.view(flow=True), 
                  projfill_tuple, 
                  squash(vals[mask2]))
        
        #Hproj.fill(
        #    **projfills,
        #    weight = squash(vals[mask2])
        #)
        #print("proj fill took %0.3f seconds" % (time()-t0))
        t0 = time()

        if Hcontrol is not None:
            for ax in cprojfills:
                if Hcontrol.axes[ax].traits.underflow:
                    cprojfills[ax] += 1
                cprojfills[ax] = cprojfills[ax].astype(np.int32)
            
            cprojfill_tuple = tuple(cprojfills[ax.name] for ax in Hproj.axes)
            
            np.add.at(Hcontrol.view(flow=True), 
                      cprojfill_tuple, 
                      squash(cvals[cmask2]))

            #Hcontrol.fill(
            #    **cprojfills,
            #    weight = squash(cvals[cmask2])
            #)

        Nevt = len(pt)
        extent = Hproj.axes.extent
        left = np.zeros((Nevt, *extent))

        evt = ak.local_index(vals, axis=0)
        evt, _ = ak.broadcast_arrays(evt, vals)
        evt = squash(evt[mask2])

        covtuple = (squash(evt), *projfill_tuple)
        
        #there's a little bit of floating point error
        #probably because I am forcing the sum order
        #I think NBD though
        np.add.at(left, covtuple, squash(vals[mask2]))

        leftis = [0] + [i+1 for i in range(len(Hproj.axes))]
        rightis = [0] + [i+11 for i in range(len(Hproj.axes))]
        ansis = leftis[1:] + rightis[1:]
        #print("covsetup %0.3f seconds" % (time()-t0))
        t0 = time()
        cov = np.einsum(left, leftis, left, rightis, ansis, optimize=True)
        #print("einsum took %0.3f seconds" % (time()-t0))
        t0 = time()
        Hcov += cov #I love that this just works!
        #print("addition took %0.3f seconds" % (time()-t0))
        t0 = time()

    def binAll(self, readers, mask, evtMask, wt):
        Hreco, HcovReco, Hcontrol, HcovControl = self._make_and_fill_EEC(
                readers.rRecoEEC, readers.rRecoJet, 
                readers.nPU, wt, mask,
                which = 'proj',
                rControlEEC = readers.rControlEEC, 
                rControlJet = readers.rControlJet)

        Hres3, HcovRes3 = self._make_and_fill_EEC(
                readers.rRecoEEC, readers.rRecoJet,
                readers.nPU, wt, mask,
                which = 'res3')

        #Hres4, HcovRes4 = self._make_and_fill_EEC(
        #        readers.rRecoEEC, readers.rRecoJet,
        #        readers.nPU, wt, mask,
        #        which = 'res4')

        if self.isMC:
            Htrans = self._make_and_fill_transfer(
                    readers.rTransfer, readers.rGenEEC, readers.rGenEECUNMATCH, 
                    readers.rRecoJet, readers.rGenJet, readers.nPU, wt, mask,
                    which='proj') 

            HtransRes3 = self._make_and_fill_transfer(
                    readers.rTransfer, readers.rGenEEC, readers.rGenEECUNMATCH, 
                    readers.rRecoJet, readers.rGenJet, readers.nPU, wt, mask,
                    which='res3') 

            #HrecoUNMATCH, HcovRecoUNMATCH = self._make_and_fill_EEC(
            #        readers.rRecoEECUNMATCH, readers.rRecoJet,
            #        readers.nPU, wt, mask,
            #        which='proj')
            #Hres3UNMATCH, HcovRes3UNMATCH = self._make_and_fill_EEC(
            #        readers.rRecoEECUNMATCH, readers.rRecoJet,
            #        readers.nPU, wt, mask,
            #        which = 'res3')

            HrecoPure, HcovRecoPure = self._make_and_fill_EEC(
                    readers.rRecoEEC, readers.rRecoJet,
                    readers.nPU, wt, mask,
                    subtract = readers.rRecoEECUNMATCH,
                    which='proj')
            Hres3Pure, HcovRes3Pure = self._make_and_fill_EEC(
                    readers.rRecoEEC, readers.rRecoJet,
                    readers.nPU, wt, mask,
                    subtract = readers.rRecoEECUNMATCH,
                    which = 'res3')

            Hgen, HcovGen = self._make_and_fill_EEC(
                    readers.rGenEEC, readers.rGenJet,
                    readers.nPU, wt, mask,
                    which='proj')
            Hres3Gen, HcovRes3Gen = self._make_and_fill_EEC(
                    readers.rGenEEC, readers.rGenJet,
                    readers.nPU, wt, mask,
                    which = 'res3')

            #HgenUNMATCH, HcovGenUNMATCH = self._make_and_fill_EEC(
            #        readers.rGenEECUNMATCH, readers.rGenJet,
            #        readers.nPU, wt, mask,
            #        which='proj')
            #Hres3GenUNMATCH, HcovRes3GenUNMATCH = self._make_and_fill_EEC(
            #        readers.rGenEECUNMATCH, readers.rGenJet,
            #        readers.nPU, wt, mask,
            #        which = 'res3')

            HgenPure, HcovGenPure = self._make_and_fill_EEC(
                    readers.rGenEEC, readers.rGenJet,
                    readers.nPU, wt, mask,
                    subtract = readers.rGenEECUNMATCH)
            Hres3GenPure, HcovRes3GenPure = self._make_and_fill_EEC(
                    readers.rGenEEC, readers.rGenJet,
                    readers.nPU, wt, mask,
                    subtract = readers.rGenEECUNMATCH,
                    which = 'res3')

        if self.isMC:
            return {
                'Htrans': Htrans,
                'HtransRes3': HtransRes3,

                'Hreco': Hreco,
                'Hres3' : Hres3,
                #'Hres4' : Hres4,

                'HcovReco': HcovReco,
                'HcovRes3' : HcovRes3,
                #'HcovRes4' : HcovRes4,

                'Hcontrol': Hcontrol,
                'HcovControl': HcovControl,

                #'HrecoUNMATCH': HrecoUNMATCH,
                #'Hres3UNMATCH': Hres3UNMATCH,

                #'HcovRecoUNMATCH': HcovRecoUNMATCH,
                #'HcovRes3UNMATCH': HcovRes3UNMATCH,

                'HrecoPure' : HrecoPure,
                'Hres3Pure' : Hres3Pure,

                'HcovRecoPure' : HcovRecoPure,
                'HcovRes3Pure' : HcovRes3Pure,

                'Hgen': Hgen,
                'Hres3Gen': Hres3Gen,

                'HcovGen': HcovGen,
                'HcovRes3Gen': HcovRes3Gen,

                #'HgenUNMATCH': HgenUNMATCH,
                #'Hres3GenUNMATCH': Hres3GenUNMATCH,

                #'HcovGenUNMATCH': HcovGenUNMATCH,
                #'HcovRes3GenUNMATCH': HcovRes3GenUNMATCH,

                'HgenPure' : HgenPure,
                'Hres3GenPure' : Hres3GenPure,

                'HcovGenPure' : HcovGenPure,
                'HcovRes3GenPure' : HcovRes3GenPure,

                'config' : self._config,
            }
        else:
            return {
                'Hreco' : Hreco,
                'HcovReco' : HcovReco,
                'Hres3' : Hres3,
                'HcovRes3' : HcovRes3,
                'config' : self._config,
           }
