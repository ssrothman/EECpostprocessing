import hist
import os
import awkward as ak
import numpy as np
import pickle
import numbers
from .EECstats import *

def project_out(H, axes_to_remove):
    axes = list(H.axes.name)
    for ax in axes_to_remove:
        axes.remove(ax)
    return H.project(*axes)

def projectCovAsym(Hcov, bins1, bins2):
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
        elif axisname != 'dRbin':
            Hcov = project_out(Hcov, [axis])
    #make sure axes are in correct order
    Hcov = Hcov.project('dRbin_1', 'dRbin_2')
    return Hcov

def project(H, Hcov, bins):
    #Confirmed to handle covariances correctly :)
    names = H.axes.name
    for axis in names:
        if axis in bins:
            H = H[{axis : bins[axis]}]
            Hcov = Hcov[{axis+'_1':bins[axis], axis+'_2':bins[axis]}]
        elif axis != 'dRbin':
            H = project_out(H, [axis])
            Hcov = project_out(Hcov, [axis+'_1', axis+'_2'])

    values = H.values(flow=True)
    covariance = Hcov.values(flow=True)

    return values, covariance

class EEChistReader:
    def __init__(self, path):
        self.path = path
        self.Hdict = {}
        self.load()

    def load(self):
        pklpath = os.path.join(self.path, 'hists.pkl')
        if os.path.exists(pklpath):
            with open(pklpath, 'rb') as f:
                self.Hdict = pickle.load(f)
        else:
            raise FileNotFoundError(f'File {pklpath} not found')

    @property
    def projToCov(self):
        return {
            'Hreco' : 'HcovReco',
            'Hgen' : 'HcovGen',
            'HrecoUNMATCH' : 'HcovRecoUNMATCH',
            'HgenUNMATCH' : 'HcovGenUNMATCH',
            'HrecoPure' : 'HcovRecoPure',
            'HgenPure' : 'HcovGenPure'
        }

    def getProj(self, name, key):
        return self.Hdict[name][key], \
               self.Hdict[name][self.projToCov[key]]

    def getProjValsErrs(self, name, key, bins,
                        density=False):
        #should have correct statistical treatment 
        Hproj, Hcov = self.getProj(name, key)
        vals, cov = project(Hproj, Hcov, bins)
        vals, cov = maybe_density(vals, cov, density)

        return vals, np.sqrt(diagonal(cov))

    def getRelationValsErrs(self, name, key, bins, density,
                            other, oname, okey, obins, odensity,
                            mode='ratio'):
        if key == 'factor':
            vals, _ = self.getFactorizedTransfer(name, bins)
            cov = np.zeros((*vals.shape, *vals.shape))

            oval, _ = other.getFactorizedTransfer(oname, obins)
            ocov = np.zeros((*oval.shape, *oval.shape))

            N = 1
            oN = 1
        else:
            Hproj, Hcov = self.getProj(name, key)
            vals, cov = project(Hproj, Hcov, bins)
            vals, cov, N = maybe_density(vals, cov, density, return_N=True)

            oHproj, oHcov = other.getProj(oname, okey)
            oval, ocov = project(oHproj, oHcov, obins)
            oval, ocov, oN = maybe_density(oval, ocov, odensity, return_N=True)

        if self is not other or key != okey or name != oname or key=='factor':
            #treat them as independent
            cov1x2 = None
        else:
            _, cov1x2 = self.getProj(name, key)
            cov1x2 = projectCovAsym(cov1x2, bins, obins)
            cov1x2 = cov1x2.values(flow=True)
            #cov1x2 = None

        cov1x2 = maybe_density_cross(vals, oval, density, odensity,
                                     N, oN, cov1x2)
        ans, covans = applyRelation(vals, cov, oval, ocov, cov1x2, mode)
        return ans, np.sqrt(diagonal(covans))

    def getTransfer(self, name):
        return self.Hdict[name]['Htrans']

    def getTransferedGen(self, name, pure=True):
        if pure:
            gen, covgen = self.getProj(name, 'HgenPure')
        else:
            gen, covgen = self.getProj(name, 'Hgen')
        names = gen.axes.name
        for axis in names:
            if self.Hdict[name]['config']['skipTrans'][axis]:
                gen = project_out(gen, [axis])
                covgen = project_out(covgen, [axis+'_1', axis+'_2'])
        return gen, covgen

    def getTransferedReco(self, name, pure=True):
        if pure:
            reco, covreco = self.getProj(name, 'HrecoPure')
        else:
            reco, covreco = self.getProj(name, 'Hreco')
        names = reco.axes.name
        for axis in names:
            if self.Hdict[name]['config']['skipTrans'][axis]:
                reco = project_out(reco, [axis])
                covreco = project_out(covreco, [axis+'_1', axis+'_2'])
        return reco, covreco
    
    def getAxisIds(self, name):
        gen = {}
        reco = {}
        diag = {}
        Htrans = self.getTransfer(name)
        for i, axis in enumerate(Htrans.axes.name):
            if 'Gen' in axis:
                gen[axis[:-4]] = i
            elif 'Reco' in axis:
                reco[axis[:-5]] = i+10
            else:
                diag[axis] = i+20

        return gen, reco, diag

    def getRecoAxisIds(self, name):
        genIds, recoIds, diagIds = self.getAxisIds(name)
        Hreco = self.getTransferedReco(name)[0]
        ans = []
        for axis in Hreco.axes.name:
            if axis in recoIds:
                ans.append(recoIds[axis])
            else:
                ans.append(diagIds[axis])
        return ans

    def getGenAxisIds(self, name):
        genIds, recoIds, diagIds = self.getAxisIds(name)
        Hgen = self.getTransferedGen(name)[0]
        ans = []
        for axis in Hgen.axes.name:
            if axis in genIds:
                ans.append(genIds[axis])
            else:
                ans.append(diagIds[axis])
        return ans
    
    def getTransferAxisIds(self, name):
        genIds, recoIds, diagIds = self.getAxisIds(name)
        Htrans = self.getTransfer(name)
        ans = []
        for axis in Htrans.axes.name:
            if 'Gen' in axis:
                ans.append(genIds[axis[:-4]])
            elif 'Reco' in axis:
                ans.append(recoIds[axis[:-5]])
            else:
                ans.append(diagIds[axis])
        return ans

    def getTransferIndex(self, name, key):
        Htrans = self.getTransfer(name)
        return Htrans.axes.name.index(key)

    def getTransferMat(self, name):
        mat = self.getTransfer(name).values(flow=True)

        genIds = self.getGenAxisIds(name)
        recoIds = self.getRecoAxisIds(name)
        transferIds = self.getTransferAxisIds(name)

        #CHECK
        reco = self.getTransferedReco(name)[0].values(flow=True)
        summat = np.einsum(mat, transferIds, recoIds, optimize=True)
        assert (np.all(np.isclose(summat, reco)))

        #NORMALIZE
        gen = self.getTransferedGen(name)[0].values(flow=True)
        invgen = np.where(gen > 0, 1./gen, 0)
        mat = np.einsum(mat, transferIds, invgen, genIds, transferIds, 
                        optimize=True)

        forwardtest = np.einsum(mat, transferIds, gen, genIds, recoIds,
                                optimize=True)
        return mat

    def getFactorizedTransfer(self, name, bins={}, keepaxis='dRbin'):
        mat = self.getTransferMat(name)

        config = self.Hdict[name]['config']

        indexing = [slice(None)] * len(mat.shape)
        for key, val in bins.items():
            idx = val
            if self.Hdict[name]['Hreco'].axes[key].traits.underflow:
                idx += 1
            idxslice = slice(idx, idx+1)
            if config['skipTrans'][key]:
                continue
            elif config['diagTrans'][key]:
                indexing[self.getTransferIndex(name, key)] = idxslice
            else:
                indexing[self.getTransferIndex(name, key+"_Reco")] = idxslice
                indexing[self.getTransferIndex(name, key+"_Gen")] = idxslice

        keep = []
        for i, axis in enumerate(self.getTransfer(name).axes.name):
            if keepaxis in axis:
                keep.append(i)

        mat = mat[tuple(indexing)]
        mat = np.einsum(mat, list(range(len(mat.shape))), keep, optimize=True)
 
        F = np.sum(mat, axis=(0))
        invF = np.where(F > 0, 1./F, 0)
        S = np.einsum('ai,i->ai', mat, invF, optimize=True)

        return F, S

    def getGenTemplate(self, name):
        gen, covgen = self.getTransferedGen(name, False)
        genM, covgenM = self.getTransferedGen(name, True)

        genvals = gen.values(flow=True)
        genMvals = genM.values(flow=True)
        covgenvals = covgen.values(flow=True)
        covgenMvals = covgenM.values(flow=True)

        #ratio, covratio = getratio(genMvals, genvals,
        #                           covgenMvals, covgenvals,
        #                           cov1x2 = None) #conservative choice
        ratio = np.nan_to_num(genMvals/genvals)
        covratio = np.zeros_like(covgenvals)

        return ratio, covratio
        
    def getRecoTemplate(self, name):
        reco, covreco = self.getTransferedReco(name, False)
        recoM, covrecoM = self.getTransferedReco(name, True)

        recovals = reco.values(flow=True)
        recoMvals = recoM.values(flow=True)
        covrecovals = covreco.values(flow=True)
        covrecoMvals = covrecoM.values(flow=True)

        #ratio, covratio = getratio(recovals, recoUMvals,
        #                           covrecovals, covrecoUMvals,
        #                           cov1x2 = None) #conservative choice
        ratio = np.nan_to_num(recovals/recoMvals)
        covratio = np.zeros_like(covrecovals)

        return ratio, covratio

    def getCombinedTransferMatrix(self, name):
        UG, _ = self.getGenTemplate(name)
        mat = self.getTransferMat(name)
        UR, _ = self.getRecoTemplate(name)

        genIds = self.getGenAxisIds(name)
        recoIds = self.getRecoAxisIds(name)
        transferIds = self.getTransferAxisIds(name)

        matrix = np.einsum(UR, recoIds, mat, transferIds, UG, genIds,
                           transferIds, optimize=True)

        #check
        recopure = self.getTransferedReco(name, True)[0].values(flow=True)
        reco = self.getTransferedReco(name, False)[0].values(flow=True)
        gen = self.getTransferedGen(name, False)[0].values(flow=True)

        forwardtest = np.einsum(matrix, transferIds, gen, genIds, 
                                recoIds, optimize=True)
        assert (np.all((np.isclose(forwardtest, reco)) | (recopure==0)))
        bad = ~np.isclose(forwardtest, reco)
        if np.any(bad):
            print("WARNING: some bins are zero in pure and nonzero in unmatched")

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
        
        genIds = self.getGenAxisIds(name)
        recoIds = self.getRecoAxisIds(name)
        transferIds = self.getTransferAxisIds(name)
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
        vals, cov = project(Hforward, HcovForward, bins)
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

        Htrans = self.getTransfer(name)
        READMEpath = os.path.join(basepath, 'README.txt')
        self.writeUnfoldingDocs(name, READMEpath,
                           reco, covreco, Htrans)

    def writeUnfoldingDocs(self, name, READMEpath, 
                           Hreco, HcovReco, Htrans):
        config = self.Hdict[name]['config']

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
            genIds = self.getGenAxisIds(name)
            recoIds = self.getRecoAxisIds(name)
            transferIds = self.getTransferAxisIds(name)
            f.write("np.einsum(M, %s, gen, %s, %s, optimize=True)"%(
                transferIds, genIds, recoIds))
