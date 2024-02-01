import hist
import os
import awkward as ak
import numpy as np
import pickle

def project_out(H, axes_to_remove):
    axes = list(H.axes.name)
    for ax in axes_to_remove:
        axes.remove(ax)
    return H.project(*axes)

def project(H, Hcov, ptbin, pubin):
    if ptbin is not None:
        H = H[{'pt':ptbin}]
        Hcov = Hcov[{'pt_1':ptbin, 'pt_2':ptbin}]
    else:
        H = project_out(H, ['pt'])
        Hcov = project_out(Hcov, ['pt_1', 'pt_2'])

    if pubin is not None:
        H = H[{'nPU':pubin}]
        Hcov = Hcov[{'nPU_1':pubin, 'nPU_2':pubin}]
    else:
        H = project_out(H, ['nPU'])
        Hcov = project_out(Hcov, ['nPU_1', 'nPU_2'])

    values = H.values(flow=True)
    covariance = Hcov.values(flow=True)
    errs = np.sqrt(np.einsum('ii->i', covariance))

    return values, errs



class EEChistReader:
    def __init__(self, path):
        self.path = path
        self.Hdict = {}
        self.load()
        self.fillPure()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.Hdict = pickle.load(f)
        else:
            raise FileNotFoundError(f'File {self.path} not found')

    def fillPure(self):
        for name in self.Hdict:
            self.Hdict[name]['HrecoPure'] = self.Hdict[name]['Hreco'] - self.Hdict[name]['HrecoUNMATCH']
            self.Hdict[name]['HgenPure'] = self.Hdict[name]['Hgen'] - self.Hdict[name]['HgenUNMATCH']
            #note the correction for the covariance due to the correlations
            self.Hdict[name]['HcovRecoPure'] = self.Hdict[name]['HcovReco'] - self.Hdict[name]['HcovRecoUNMATCH']
            self.Hdict[name]['HcovGenPure'] = self.Hdict[name]['HcovGen'] - self.Hdict[name]['HcovGenUNMATCH']

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

    def getProjValsErrs(self, name, key,
                        ptbin, pubin):
        Hproj, Hcov = self.getProj(name, key)
        return project(Hproj, Hcov, ptbin, pubin)

    def getTransfer(self, name):
        return self.Hdict[name]['Htrans']

    def getTransferMat(self, name):
        mat = self.getTransfer(name).values(flow=True)

        #CHECK
        reco = self.getProj(name, 'HrecoPure')[0].values(flow=True)
        summat = np.sum(mat, axis=(3,4,5))
        assert (np.all(np.isclose(summat, reco)))

        #NORMALIZE
        gen = self.getProj(name, 'HgenPure')[0].values(flow=True)
        invgen = np.where(gen > 0, 1./gen, 0)
        mat = np.nan_to_num(np.einsum('abcijk,ijk->abcijk', mat, invgen))
        return mat

    def getFactorizedTransfer(self, name, pubin, ptbin):
        mat = self.getTransferMat(name)

        mat = mat[ptbin, :, pubin, ptbin, :, pubin]
        print(mat.shape)

        F = np.sum(mat, axis=(0))
        invF = np.where(F > 0, 1./F, 0)
        S = np.einsum('ai,i->ai', mat, invF)

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
        
        forward = np.einsum('abcijk,ijk->abc', mat, gen)

        #have to instantiate the transposed matrix
        #in contiguous memory
        #otherwise the einsum will do some crazy inefficient stuff
        #and take a million years
        transmat = np.ascontiguousarray(np.einsum('abcijk->ijkabc', mat))

        #even still, have to do it in two steps
        step1 = np.einsum('abcdef, defghi -> abcghi', mat, covgen)
        covforward = np.einsum('abcghi, ghijkl -> abcjkl', step1, transmat)

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
