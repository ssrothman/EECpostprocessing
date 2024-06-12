import hist
import os
import awkward as ak
import numpy as np
import pickle
import numbers
from .EECstats import *


class EEChistReader:
    def __init__(self, Hdict):
        self.Hdict = Hdict
    
    def project_out(self, H, axes_to_remove):
        axes = list(H.axes.name)
        for ax in axes_to_remove:
            if ax in axes:
                axes.remove(ax)
        return H.project(*axes)

    def projectCovAsym(self, Hcov, bins1, bins2, targets=['dRbin']):
        names = Hcov.axes.name
        for axis in names:
            if '_1' in axis:
                thesebins = bins1
            elif '_2' in axis:
                thesebins = bins2
            else:
                raise ValueError(f'Axis {axis} does not have a _1 or _2')
            axisname = axis[:-2]
            if axisname in thesebins:
                Hcov = Hcov[{axis : thesebins[axisname]}]
            if axisname not in targets:
                Hcov = self.project_out(Hcov, [axis])
        #make sure axes are in correct order
        #Hcov = Hcov.project('dRbin_1', 'dRbin_2')
        return Hcov

    def project(self, H, Hcov, bins, targets=['dRbin'], to_np=True):
        #print("in project()")
        #print("\t", targets)
        #print("\t", bins)
        #Confirmed to handle covariances correctly :)
        names = H.axes.name
        for axis in names:
            #print(axis)
            if axis in bins:
                if type(bins[axis]) is slice:
                    if axis not in targets and bins[axis].stop is None:
                        bins[axis] = slice(bins[axis].start, 
                                           bins[axis].stop, 
                                           hist.sum)
                #print("binning")
                H = H[{axis : bins[axis]}]
                #print("binned H")
                Hcov = Hcov[{axis+'_1':bins[axis], axis+'_2':bins[axis]}]
                #print("binned Hcov")
            if axis not in targets:
                #print("projecting")
                H = self.project_out(H, [axis])
                Hcov = self.project_out(Hcov, [axis+'_1', axis+'_2'])

        if to_np:
            values = H.values(flow=True)
            covariance = Hcov.values(flow=True)

            return values, covariance
        else:
            return H, Hcov

    @property
    def projToCov(self):
        return {
            'Hreco' : 'HcovReco',
            'Hgen' : 'HcovGen',
            'HrecoUNMATCH' : 'HcovRecoUNMATCH',
            'HgenUNMATCH' : 'HcovGenUNMATCH',
            'HrecoPure' : 'HcovRecoPure',
            'HgenPure' : 'HcovGenPure',
            'Hres3' : 'HcovRes3',
            'Hres3Pure' : 'HcovRes3Pure',
            'Hres3Gen' : 'HcovRes3Gen',
            'Hres3GenPure' : 'HcovRes3GenPure',
            'Hcontrol' : 'HcovControl',
        }

    def getFlavorFracs(self, ptbin=None):
        if not hasattr(self, 'evtdict'):
            raise AttributeError('No event data loaded')

        Hjet = self.evtdict['Events']['recoJet']
        vals = Hjet.project('pt', 'tag', 'genflav').values(flow=True)

        if ptbin is not None:
            vals = vals[ptbin]
        else:
            vals = np.sum(vals, axis=0)

        vals = np.nan_to_num(vals/np.sum(vals, axis=1, keepdims=True))

        return vals

    def invFlavorFracs(self, ptbin=None):
        return np.linalg.pinv(self.getFlavorFracs(ptbin))

    def getPureFlavor(self, name, key, bins):
        bins = bins.copy()
        pureFlav = bins['pureflav']
        bins.pop('pureflav')

        lbins = bins.copy()
        lbins['tag'] = 0
        vals_ltag, _ = self.getProjValsErrs(name, key, lbins, True)
        cbins = bins.copy()
        cbins['tag'] = 1
        vals_ctag, _ = self.getProjValsErrs(name, key, cbins, True)
        bbins = bins.copy()
        bbins['tag'] = 2
        vals_btag, _ = self.getProjValsErrs(name, key, bbins, True)

        vals = np.stack([vals_ltag, vals_ctag, vals_btag], axis=0)
        print("VALS")
        print(vals.shape)
        print(vals[0])
        print(vals[1])
        print(vals[2])

        ptbin = None
        if 'pt' in bins:
            ptbin = bins['pt']
        invF = self.invFlavorFracs(ptbin)
        print("INVF")
        print(invF)

        ans = np.einsum('ij,jk -> ik', invF, vals, optimize=True)
        print("ANS")
        print(ans.shape)
        print(ans[0])
        print(ans[1])
        print(ans[2])
        print("INDEXED")
        print(pureFlav)
        print(ans[pureFlav])
        return ans[pureFlav, :]

    def getHists(self, name, key):
        return self.Hdict[key], \
               self.Hdict[self.projToCov[key]]

    def getProjValsErrs(self, name, key, bins,
                        density=False):
        #should have correct statistical treatment 
        Hproj, Hcov = self.getHists(name, key)
        if 'pureflav' in bins:
            ans  = self.getPureFlavor(name, key, bins)
            return ans, np.zeros_like(ans)

        vals, cov = self.project(Hproj, Hcov, bins)
        vals, cov = maybe_density(vals, cov, density)

        return vals, np.sqrt(diagonal(cov))

    def getRes3ValsErrs(self, name, key, bins,
                        density=False):
        Hres3, HcovRes3 = self.getHists(name, key)
        if 'pureflav' in bins:
            raise NotImplementedError

        vals, cov = self.project(Hres3, HcovRes3, bins, targets=['xi', 'phi'])
        vals, cov = maybe_density(vals, cov, density)

        return vals, np.sqrt(diagonal(cov))

    def getRelationValsErrs(self, name, key, bins, density,
                            other, oname, okey, obins, odensity,
                            ratiomode='ratio'):
        if key == 'factor':
            vals, _ = self.getFactorizedTransfer(name, bins)
            cov = np.zeros((*vals.shape, *vals.shape))

            oval, _ = other.getFactorizedTransfer(oname, obins)
            ocov = np.zeros((*oval.shape, *oval.shape))

            N = 1
            oN = 1
        else:
            Hproj, Hcov = self.getHists(name, key)
            oHproj, oHcov = other.getHists(oname, okey)
            if 'res3' in key:
                vals, cov = self.project(Hproj, Hcov, bins, targets=['xi', 'phi'])
                oval, ocov = self.project(oHproj, oHcov, obins, targets=['xi', 'phi'])
            else:
                vals, cov = self.project(Hproj, Hcov, bins, targets=['dRbin'])
                oval, ocov = self.project(oHproj, oHcov, obins, targets=['dRbin'])

            vals, cov, N = maybe_density(vals, cov, density, return_N=True)
            oval, ocov, oN = maybe_density(oval, ocov, odensity, return_N=True)

        if self is not other or key != okey or name != oname or key=='factor':
            #treat them as independent
            cov1x2 = None
        else:
            _, cov1x2 = self.getHists(name, key)
            if 'res3' in key:
                cov1x2 = self.projectCovAsym(cov1x2, bins, obins, targets=['xi', 'phi'])
            else:
                cov1x2 = self.projectCovAsym(cov1x2, bins, obins)

            cov1x2 = cov1x2.values(flow=True)

        cov1x2 = maybe_density_cross(vals, oval, density, odensity,
                                     N, oN, cov1x2)
        ans, covans = applyRelation(vals, cov, oval, ocov, cov1x2, ratiomode)
        return ans, np.sqrt(diagonal(covans))

    def getTransfer(self, name, mode):
        if mode == 'proj':
            return self.Hdict['Htrans']
        elif mode=='res3':
            return self.Hdict['HtransRes3']
        elif mode=='res4':
            return self.Hdict['HtransRes4']

    def getTransferedGen(self, name, pure, bins, keepaxes, mode):
        if pure:
            if mode == 'proj':
                gen, covgen = self.getHists(name, 'HgenPure')
            elif mode == 'res3':
                gen, covgen = self.getHists(name, 'Hres3GenPure')
        else:
            if mode == 'proj':
                gen, covgen = self.getHists(name, 'Hgen')
            elif mode == 'res3':
                gen, covgen = self.getHists(name, 'Hres3Gen')

        if bins is None:
            bins = {}
        if keepaxes is None:
            keepaxes = gen.axes.name


        names = gen.axes.name
        for axis in names:
            if self.Hdict['config']['skipTrans'][axis]:
                gen = self.project_out(gen, [axis])
                covgen = self.project_out(covgen, [axis+'_1', axis+'_2'])
    
        #print("keepaxes is", keepaxes)
        gen, covgen = self.project(gen, covgen, bins=bins, 
                                   targets=keepaxes, to_np=False)

        return gen, covgen

    def getTransferedReco(self, name, pure, bins, keepaxes, mode):
        if pure:
            if mode == 'proj':
                reco, covreco = self.getHists(name, 'HrecoPure')
            elif mode == 'res3':
                reco, covreco = self.getHists(name, 'Hres3Pure')
        else:
            if mode == 'proj':
                reco, covreco = self.getHists(name, 'Hreco')
            elif mode == 'res3':
                reco, covreco = self.getHists(name, 'Hres3')
    
        if bins is None:
            bins = {}
        if keepaxes is None:
            keepaxes = reco.axes.name

        names = reco.axes.name
        for axis in names:
            if self.Hdict['config']['skipTrans'][axis]:
                reco = self.project_out(reco, [axis])
                covreco = self.project_out(covreco, [axis+'_1', axis+'_2'])

        #print("keepaxes is", keepaxes)
        reco, covreco = self.project(reco, covreco, bins=bins,
                                     targets=keepaxes, to_np=False)

        return reco, covreco
    
    def getAxisIds(self, name, mode):
        gen = {}
        reco = {}
        diag = {}
        Htrans = self.getTransfer(name, mode)
        for i, axis in enumerate(Htrans.axes.name):
            if 'Gen' in axis:
                gen[axis[:-4]] = i
            elif 'Reco' in axis:
                reco[axis[:-5]] = i+10
            else:
                diag[axis] = i+20

        return gen, reco, diag

    def getRecoAxisIds(self, name, keepaxes, mode):
        genIds, recoIds, diagIds = self.getAxisIds(name, mode)
        Hreco = self.getTransferedReco(name, False, {}, keepaxes, mode)[0]
        ans = []

        if keepaxes is None:
            keepaxes = Hreco.axes.name

        for axis in Hreco.axes.name:
            if axis not in keepaxes:
                continue
            
            if axis in recoIds:
                ans.append(recoIds[axis])
            else:
                ans.append(diagIds[axis])
        return ans

    def getGenAxisIds(self, name, keepaxes, mode):
        genIds, recoIds, diagIds = self.getAxisIds(name, mode)
        Hgen = self.getTransferedGen(name, False, {}, keepaxes, mode)[0]
        ans = []

        if keepaxes is None:
            keepaxes = Hgen.axes.name

        for axis in Hgen.axes.name:
            if axis not in keepaxes:
                continue

            if axis in genIds:
                ans.append(genIds[axis])
            else:
                ans.append(diagIds[axis])
        return ans
    
    def getTransferAxisIds(self, name, keepaxes, mode):
        #print("in getTransferAxisIds")
        #print("\tkeepaxes", keepaxes)
        genIds, recoIds, diagIds = self.getAxisIds(name, mode)
        Htrans = self.getTransfer(name, mode)
        ans = []

        if keepaxes is None:
            keepaxes = self.getTransferedReco(name, False, {}, None, mode)[0].axes.name

        for axis in Htrans.axes.name:
            if 'Gen' in axis:
                if axis[:-4] not in keepaxes:
                    continue
                ans.append(genIds[axis[:-4]])
            elif 'Reco' in axis:
                if axis[:-5] not in keepaxes:
                    continue
                ans.append(recoIds[axis[:-5]])
            else:
                if axis not in keepaxes:
                    continue
                ans.append(diagIds[axis])
        return ans

    def getTransferIndex(self, name, key, mode):
        Htrans = self.getTransfer(name, mode)
        return Htrans.axes.name.index(key)

    def getTransferMat(self, name, bins, keepaxes, mode):
        mat = self.getTransfer(name, mode)
        print("RAW TRANSFER")
        print(mat)

        if bins is None:
            bins = {}
        if keepaxes is None:
            keepaxes = self.getTransferedReco(name, False, bins, keepaxes, mode)[0].axes.name

        for axis in mat.axes.name:
            stripgen = axis[:-4]
            stripreco = axis[:-5]

            if stripgen in bins:
                binning = bins[stripgen]
            elif stripreco in bins:
                binning = bins[stripreco]
            elif axis in bins:
                binning = bins[axis]
            else:
                binning = None

            if binning is not None:
                print("binning")
                mat = mat[{axis: binning}]

            if (axis not in keepaxes) and (stripgen not in keepaxes) and (stripreco not in keepaxes):
                print("projecting out", axis)
                mat = self.project_out(mat, [axis])

        print("PROJECTED TRANFER")
        print(mat)

        #print("keepaxes is", keepaxes)
        mat = mat.values(flow=True)
        #print("mat shape", mat.shape)

        genIds = self.getGenAxisIds(name, keepaxes, mode)
        #print("genIds", genIds)
        recoIds = self.getRecoAxisIds(name, keepaxes, mode)
        #print("recoIds", recoIds)
        transferIds = self.getTransferAxisIds(name, keepaxes, mode)
        #print("transferIds", transferIds)

        #CHECK
        reco = self.getTransferedReco(name, True,
                                            bins=bins, 
                                            keepaxes=keepaxes,
                                            mode=mode
                                      )[0].values(flow=True)
        #print(reco)
        print(reco.shape)
        summat = np.einsum(mat, transferIds, recoIds, optimize=True)
        #print(summat)
        print(summat.shape)
        maxdeviation = np.max(np.abs(summat - reco))
        print("maxdeviation", maxdeviation)
        #assert (np.all(np.isclose(summat, reco)))

        #NORMALIZE
        gen = self.getTransferedGen(name, True,
                                          bins=bins,
                                          keepaxes=keepaxes,
                                          mode=mode
                                    )[0].values(flow=True)
        invgen = np.where(gen > 0, 1./gen, 0)
        mat = np.einsum(mat, transferIds, invgen, genIds, transferIds, 
                        optimize=True)

        print("THE MATMUL TEST")
        print("mat.shape", mat.shape)
        print("transferIds", transferIds)
        print("gen.shape", gen.shape)
        print("genIds", genIds)
        print("recoIds", recoIds)
        forwardtest = np.einsum(mat, transferIds, gen, genIds, recoIds,
                                optimize=True)
        print("forwardtest.shape", forwardtest.shape)
        print("sumreco", np.sum(reco))
        print("sumforward", np.sum(forwardtest))
        print("sumgen", np.sum(gen))
        #CHECK
        #print(reco)
        #print(forwardtest)
        print(np.max(np.abs(forwardtest - reco)))
        #assert (np.all(np.isclose(forwardtest, reco)))
        return mat

    def getFactorizedTransfer(self, name, bins={}, keepaxes=['dRbin'], mode='proj'):
        mat = self.getTransferMat(name, bins, keepaxes, mode)

        F = np.sum(mat, axis=tuple(range(len(keepaxes))), keepdims=True)
        invF = np.where(F > 0, 1./F, 0)
        S = mat * invF

        return np.squeeze(F), S

    def getGenTemplate(self, name, bins, keepaxes, mode):
        gen, covgen = self.getTransferedGen(name, False, bins, keepaxes, mode)
        genM, covgenM = self.getTransferedGen(name, True, bins, keepaxes, mode)

        genvals = gen.values(flow=True)
        genMvals = genM.values(flow=True)
        covgenvals = covgen.values(flow=True)
        covgenMvals = covgenM.values(flow=True)

        #ratio, covratio = getratio(genMvals, genvals,
        #                           covgenMvals, covgenvals,
        #                           cov1x2 = None) #conservative choice
        ratio = np.nan_to_num(genMvals/genvals)

        if np.any((genMvals==0) & (genvals!=0)):
            print("WARNING: some bins are zero in pure GEN and nonzero in unmatched")

        covratio = np.zeros_like(covgenvals)

        return ratio, covratio
        
    def getRecoTemplate(self, name, bins, keepaxes, mode):
        reco, covreco = self.getTransferedReco(name, False, bins, keepaxes, mode)
        recoM, covrecoM = self.getTransferedReco(name, True, bins, keepaxes, mode)

        recovals = reco.values(flow=True)
        recoMvals = recoM.values(flow=True)
        covrecovals = covreco.values(flow=True)
        covrecoMvals = covrecoM.values(flow=True)

        #ratio, covratio = getratio(recovals, recoUMvals,
        #                           covrecovals, covrecoUMvals,
        #                           cov1x2 = None) #conservative choice
        if np.any((recoMvals==0) & (recovals!=0)):
            print("WARNING: some bins are zero in pure RECO and nonzero in unmatched")

        ratio = np.nan_to_num(recovals/recoMvals)
        covratio = np.zeros_like(covrecovals)

        return ratio, covratio

    def getCombinedTransferMatrix(self, name, bins=None, keepaxes=None, mode='proj'):
        UG, _ = self.getGenTemplate(name, bins, keepaxes, mode)
        mat = self.getTransferMat(name, bins, keepaxes, mode)
        UR, _ = self.getRecoTemplate(name, bins, keepaxes, mode)

        genIds = self.getGenAxisIds(name, keepaxes, mode)
        recoIds = self.getRecoAxisIds(name, keepaxes, mode)
        transferIds = self.getTransferAxisIds(name, keepaxes, mode)

        #print("ONTO COMBINED")
        #print("UG.shape", UG.shape)
        #print("genIndices", genIds)
        #print("mat.shape", mat.shape)
        #print("transferIndices", transferIds)
        #print("UR.shape", UR.shape)
        #print("recoIndices", recoIds)
        matrix = np.einsum(UR, recoIds, mat, transferIds, UG, genIds,
                           transferIds, optimize=True)

        #check
        recopure = self.getTransferedReco(name, True, bins, keepaxes, mode)[0].values(flow=True)
        reco = self.getTransferedReco(name, False, bins, keepaxes, mode)[0].values(flow=True)
        gen = self.getTransferedGen(name, False, bins, keepaxes, mode)[0].values(flow=True)

        forwardtest = np.einsum(matrix, transferIds, gen, genIds, 
                                recoIds, optimize=True)
        #assert (np.all((np.isclose(forwardtest, reco)) | (recopure==0)))
        bad = ~np.isclose(forwardtest, reco)
        if np.any(bad):
            print("WARNING: closure is not perfect")

        return matrix

    def forward(self, name, other=None, othername=None,
                useTemplates=False):
        if other is None:
            other = self
        if othername is None:
            othername = name

        if useTemplates:
            fullgen, covfullgen = other.getTransferedGen(name, False)
            fullgen = fullgen.values(flow=True)
            covfullgen = covfullgen.values(flow=True)
            UG, covUG = self.getGenTemplate(name)

            gen, covgen = getproduct(fullgen, UG,
                                     covfullgen, 
                                     covUG,
                                     cov1x2=None) # treat as indep
        else:
            gen, covgen = other.getTransferedGen(name, True)
            gen = gen.values(flow=True)
            covgen = covgen.values(flow=True)

        mat = self.getTransferMat(name)
        
        genIds = self.getGenAxisIds(name, mode)
        recoIds = self.getRecoAxisIds(name, mode)
        transferIds = self.getTransferAxisIds(name, mode)
        forward = np.einsum(mat, transferIds, gen, genIds, recoIds,
                            optimize=True)
        covforward = np.zeros_like(covgen)

        #CHECK
        if other is self and othername == name:
            reco, _ = self.getTransferedReco(name, True)
            reco = reco.values(flow=True)
            assert (np.all(np.isclose(forward, reco)))
            print("passed assert 1")

        if useTemplates:
            UR, covUR = self.getRecoTemplate(name)
            forward, covforward = getproduct(forward, 
                                             UR,
                                             covforward, 
                                             covUR,
                                             cov1x2=None) #treat as indep

            #CHECK
            if other is self and othername == name:
                reco, _ = self.getTransferedReco(name, False)
                reco = reco.values(flow=True)
                assert (np.all((np.isclose(forward, reco)) | (forward==0)))
                print("passed assert 2")
                bad = ~np.isclose(forward, reco)
                if np.sum(bad) !=0:
                    print("WARNING: some bins are zero in pure and nonzero in unmatched")

        #turn these into histograms
        #this almost works..... but what if there are skipped indices?
        Hforward, HcovForward = self.getTransferedReco(name)
        Hforward = Hforward.copy().reset() + forward
        HcovForward = HcovForward.copy().reset() + covforward
        return Hforward, HcovForward

    def getForwardValsErrs(self, name, bins,
                           other=None, othername=None,
                           useTemplates=False):
        Hforward, HcovForward = self.forward(name, other, othername, 
                                             useTemplates=useTemplates)
        vals, cov = self.project(Hforward, HcovForward, bins)
        return vals, np.sqrt(diagonal(cov))

    def dumpUnfoldingMatrices(self, name):
        mat = self.getTransferMat(name)
        gen, covgen = self.getTransferedGen(name, False)
        reco, covreco = self.getTransferedReco(name, False)

        basepath = os.path.join(self.path, 'unfolding', name)
        os.makedirs(basepath, exist_ok=True)

        matpath = os.path.join(basepath, 'M.npy') 
        genpath = os.path.join(basepath, 'gen.npy')
        recopath = os.path.join(basepath, 'reco.npy')
        covgenpath = os.path.join(basepath, 'covgen.npy')
        covrecopath = os.path.join(basepath, 'covreco.npy')

        np.save(matpath, mat)
        np.save(genpath, gen.values(flow=True))
        np.save(recopath, reco.values(flow=True))
        np.save(covgenpath, covgen.values(flow=True))
        np.save(covrecopath, covreco.values(flow=True))

        Htrans = self.getTransfer(name, 'proj')
        READMEpath = os.path.join(basepath, 'README.txt')
        self.writeUnfoldingDocs(name, READMEpath,
                           reco, covreco, Htrans)

    def writeUnfoldingDocs(self, name, READMEpath, 
                           Hreco, HcovReco, Htrans):
        config = self.Hdict['config']

        with open(READMEpath, 'w') as f:
            f.write("This directory contains the transfer matrix\n")
            f.write("and gen and reco distributions in numpy format:\n")
            f.write("M.npy: transfer matrix\n")
            f.write("gen.npy: gen distribution\n")
            f.write("reco.npy: reco distribution\n")
            f.write("covgen.npy: covariance matrix of gen distribution\n")
            f.write("covreco.npy: covariance matrix of reco distribution\n\n")
            f.write("the shape of the gen and reco matrices are the same:\n")
            f.write(str(Hreco.axes.name) + "\n\n")
            f.write("the shape of the covgen and covreco matrices are the same:\n")
            f.write(str(HcovReco.axes.name) + "\n\n")
            f.write("the shape of the transfer matrix is:\n")
            f.write(str(Htrans.axes.name) + "\n\n")
            f.write("Note that the transfer matrix is diagonal in the following quantites:\n")
            for axis in Hreco.axes.name:
                if not config['skipTrans'][axis] and config['diagTrans'][axis]:
                    f.write(axis + ", ")
            f.write("\n\n")
            f.write("NB these shapes are not compatible with naive matrix multiplication\n")
            f.write("With numpy einsum, the foward-folding can be written\n")
            genIds = self.getGenAxisIds(name, 'proj')
            recoIds = self.getRecoAxisIds(name, 'proj')
            transferIds = self.getTransferAxisIds(name, 'proj')
            f.write("np.einsum(M, %s, gen, %s, %s, optimize=True)"%(
                transferIds, genIds, recoIds))
