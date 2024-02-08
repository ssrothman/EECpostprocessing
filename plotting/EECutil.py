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
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.Hdict = pickle.load(f)
        else:
            raise FileNotFoundError(f'File {self.path} not found')

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

        return vals, np.sqrt(np.einsum('ii->i', cov, optimize=True))

    def getRelationValsErrs(self, name, key, bins, density,
                            other, oname, okey, obins, odensity,
                            mode='ratio'):
        if key == 'factor':
            vals, _ = self.getFactorizedTransfer(name, bins)
            cov = np.zeros((*vals.shape, *vals.shape))

            oval, _ = other.getFactorizedTransfer(oname, obins)
            ocov = np.zeros((*oval.shape, *oval.shape))
        else:
            Hproj, Hcov = self.getProj(name, key)
            vals, cov = project(Hproj, Hcov, bins)
            vals, cov = maybe_density(vals, cov, density)

            oHproj, oHcov = other.getProj(oname, okey)
            oval, ocov = project(oHproj, oHcov, obins)
            oval, ocov = maybe_density(oval, ocov, odensity)

        if self is not other or key != okey or name != oname or key=='factor':
            #treat them as independent
            cov1x2 = None
        else:
            raise NotImplementedError("Need to implement proper covariances")

        ans, covans = applyRelation(vals, cov, oval, ocov, cov1x2, mode)
        return ans, np.sqrt(np.einsum('ii->i', covans, optimize=True))

    def getTransfer(self, name):
        return self.Hdict[name]['Htrans']

    def getTransferedGen(self, name):
        gen = self.getProj(name, 'HgenPure')[0]
        names = gen.axes.name
        for axis in names:
            if self.Hdict[name]['config']['skipTrans'][axis]:
                gen = project_out(gen, [axis])
        return gen

    def getTransferedReco(self, name):
        reco = self.getProj(name, 'HrecoPure')[0]
        names = reco.axes.name
        for axis in names:
            if self.Hdict[name]['config']['skipTrans'][axis]:
                reco = project_out(reco, [axis])
        return reco
    
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
        Hreco = self.getTransferedReco(name)
        ans = []
        for axis in Hreco.axes.name:
            if axis in recoIds:
                ans.append(recoIds[axis])
            else:
                ans.append(diagIds[axis])
        return ans

    def getGenAxisIds(self, name):
        genIds, recoIds, diagIds = self.getAxisIds(name)
        Hgen = self.getTransferedGen(name)
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
        reco = self.getTransferedReco(name).values(flow=True)
        summat = np.einsum(mat, transferIds, recoIds, optimize=True)
        assert (np.all(np.isclose(summat, reco)))

        #NORMALIZE
        gen = self.getTransferedGen(name).values(flow=True)
        invgen = np.where(gen > 0, 1./gen, 0)
        mat = np.einsum(mat, transferIds, invgen, genIds, transferIds, 
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
        gen, covgen = self.getProj(name, 'Hgen')
        genUM, covgenUM = self.getProj(name, 'HgenUNMATCH')

        genvals = gen.values(flow=True)
        genUMvals = genUM.values(flow=True)

        ratio = np.where(genvals > 0., genUMvals/genvals, 0)

        #covgenvals = covgen.values(flow=True)
        #covgenUMvals = covgenUM.values(flow=True)
        #note the correction for the covariance due to the correlation
        #covratio = ratio*ratio * (covgenvals/(genvals*genvals)
        #                          + covgenUMvals/(genUMvals*genUMvals)
        #                          - 2*covgenvals/(genvals*genUMvals))
        #covratio = np.where((genvals > 0.) & (genUMvals > 0), covratio, 0)
        
        return ratio, np.zeros((*ratio.shape, *ratio.shape))
        
    def applyGenTemplateToFull(self, name, gen, covgen):
        ratio, covratio = self.getGenTemplate(name)
        genvals, covgenvals = gen.values(flow=True), covgen.values(flow=True)

        genUM = genvals * ratio

        #covgenUM = covgen * ratio*ratio + gen*gen*covratio

        return genUM, np.zeros((*genUM.shape, *genUM.shape))

    def getRecoTemplate(self, name):
        reco, covreco = self.getProj(name, "Hreco")
        recoUM, covrecoUM = self.getProj(name, "HrecoUNMATCH")

        recovals = reco.values(flow=True)
        recoUMvals = recoUM.values(flow=True)

        ratio = np.where(recovals > 0., recoUMvals/recovals, 0)

        return ratio, np.zeros((*ratio.shape, *ratio.shape))

    def applyRecoTemplateToPure(self, name, reco, covreco):
        ratio, covratio = self.getRecoTemplate(name)
        if type(reco) is hist.Hist:
            recovals = reco.values(flow=True)
        else:
            recovals = reco
        if type(covreco) is hist.Hist:
            covrecovals = covreco.values(flow=True)
        else:
            covrecovals = covreco

        recoUM = recovals * ratio / (1-ratio)
        recoUM = np.where(ratio < 1, recoUM, 0)

        return recoUM, np.zeros((*recoUM.shape, *recoUM.shape))

    def forward(self, name, other=None, othername=None,
                useTemplates=False):
        if other is None:
            other = self
        if othername is None:
            othername = name

        if useTemplates:
            fullgen, covfullgen = other.getProj(othername, 'Hgen')
            genUM, covgenUM = self.applyGenTemplateToFull(
                    name, fullgen, covfullgen)
            gen = fullgen - genUM
            covgen = covfullgen - covgenUM #this isn't quite right
                                           #since it assumes perfect correl
                                           #but actually the template is indep
                                           #doesn't matter for now
                                           #seeing as the template
                                           #returns 0 covariance anyway
        else:
            gen, covgen = other.getProj(othername, 'HgenPure')

        gen = gen.values(flow=True)
        covgen = covgen.values(flow=True)

        mat = self.getTransferMat(name)
        
        Nax = len(mat.shape)//2
        if Nax==3:
            forward = np.einsum('abcijk,ijk->abc', mat, gen, optimize=True)
        elif Nax==2:
            forward = np.einsum('abij,ij->ab', mat, gen, optimize=True)

        #have to instantiate the transposed matrix
        #in contiguous memory
        #otherwise the einsum will do some crazy inefficient stuff
        #and take a million years
        if Nax==3:
            transmat = np.ascontiguousarray(np.einsum('abcijk->ijkabc', mat), optimize=True)
        elif Nax==2:
            transmat = np.ascontiguousarray(np.einsum('abij->ijab', mat), optimize=True)

        #even still, have to do it in two steps
        if Nax==3:
            step1 = np.einsum('abcdef, defghi -> abcghi', mat, covgen, optimize=True)
            covforward = np.einsum('abcghi, ghijkl -> abcjkl', step1, transmat, optimize=True)
        elif Nax==2:
            step1 = np.einsum('abde, degh -> abgh', mat, covgen, optimize=True)
            covforward = np.einsum('abgh, ghjk -> abjk', step1, transmat, optimize=True)

        #CHECK
        if other is self and othername == name:
            reco, _ = self.getProj(name, 'HrecoPure')
            reco = reco.values(flow=True)
            assert (np.all(np.isclose(forward, reco)))
            print("passed assert 1")

        if useTemplates:
            recoUM, covrecoUM = self.applyRecoTemplateToPure(
                    name, forward, covforward)
            forward = forward + recoUM
            covforward = covforward + covrecoUM #this isn't right,
                                                #but it doesn't matter
                                                #because the template
                                                #returns 0 covariance anyway
            #CHECK
            if other is self and othername == name:
                reco, _ = self.getProj(name, 'Hreco')
                reco = reco.values(flow=True)
                assert (np.all(np.isclose(forward, reco)))
                print("passed assert 2")

        #turn these into histograms
        Hforward = self.Hdict[name]['Hreco'].copy().reset() + forward
        HcovForward = self.Hdict[name]['HcovReco'].copy().reset() + covforward
        return Hforward, HcovForward

    def getForwardValsErrs(self, name, ptbin, pubin, 
                           other=None, othername=None,
                           useTemplates=False):
        Hforward, HcovForward = self.forward(name, other, othername, 
                                             useTemplates=useTemplates)
        return project(Hforward, HcovForward, ptbin, pubin)
