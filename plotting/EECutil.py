import hist
import os
import awkward as ak
import numpy as np
import pickle

wtaxis = hist.axis.Regular(25, 1e-6, 1, transform=hist.axis.transform.log)

class EEC(object):
    def __init__(self, path, includeInefficiencies=False):
        self.includeInefficiencies = includeInefficiencies
        self.path = path
        self.covnames = {
            'Hreco' : 'HcovReco',
            'Hgen' : 'HcovGen',
            'HrecoPUjets' : 'HcovRecoPUjets',
            'HrecoUNMATCH' : 'HcovRecoUNMATCH',
            'HgenUNMATCH' : 'HcovGenUNMATCH',
            'HrecoPure' : 'HcovRecoPure',
            'HgenPure' : 'HcovGenPure',
        }
        with open(path, 'rb') as f:
            self.Hdict = pickle.load(f)

        for name in self.Hdict.keys():
            self.Hdict[name]['HrecoPure'] = self.Hdict[name]['Hreco'] - \
                                                self.Hdict[name]['HrecoPUjets'] - \
                                                self.Hdict[name]['HrecoUNMATCH']
            self.Hdict[name]['HcovRecoPure'] = self.Hdict[name]['HcovReco'] - \
                                                self.Hdict[name]['HcovRecoPUjets'] - \
                                                self.Hdict[name]['HcovRecoUNMATCH']
            self.Hdict[name]['HgenPure'] = self.Hdict[name]['Hgen'] - \
                                                self.Hdict[name]['HgenUNMATCH']
            self.Hdict[name]['HcovGenPure'] = self.Hdict[name]['HcovGen'] - \
                                                self.Hdict[name]['HcovGenUNMATCH']

    def getDRtransfer(self, name, etabin, pubin):
        S = self.Hdict[name]['HtransSG'].values(flow=True)
        if etabin is None:
            S = np.sum(S, axis=0)
        elif etabin>=0:
            S = S[etabin+1, :, :, :, :]

        if etabin is not None and etabin<0:
            print(S.shape)
            print(np.sum(S, axis=(1,2)).shape)
            sumS = 1/np.sum(S, axis=(1,2))
            S = np.einsum('eijkl,ekl->eijkl', S, sumS)
            S = np.nan_to_num(S)
            print(S.shape)
            print('made')
        else:
            S = np.nan_to_num(S/np.sum(S, axis=(0,1)))
        return S

    def getFullTransfer(self, name):
        S = self.getDRtransfer(name, -1, None)
        factors = self.Hdict[name]['HtransFR']#.project('ptReco', 'dRbinReco')
        factors = factors.values(flow=True)
        print(factors.shape)
        print(S.shape)
        return np.einsum('eij,eijkl->eijkl', factors, S)

    def applyPUjetsTemplate(self, name, ontopof):
        #if other is None:
        #    other = self
        #if othername is None:
        #    othername = name

        PUjets = self.Hdict[name]['HrecoPUjets'].project('pt', 'dRbin').values(flow=True)
        total = self.Hdict[name]['HrecoPure'].project('pt', 'dRbin').values(flow=True)
        template = np.nan_to_num(PUjets/total)

        #othertotal = other.Hdict[name]['Hreco'].project("ptReco", 'dRbinReco').values(flow=True)
        return template * ontopof

    def applyContaminationTemplate(self, name, ontopof):
        #if other is None:
        #    other = self
        #if othername is None:
        #    othername = name

        UNMATCH = self.Hdict[name]['HrecoUNMATCH'].project('pt', 'dRbin').values(flow=True)
        total = self.Hdict[name]['HrecoPure'].project('pt', 'dRbin').values(flow=True)
        template = np.nan_to_num(UNMATCH/total)

        #othertotal = other.Hdict[name]['Hreco'].project("ptReco", 'dRbinReco').values(flow=True)
        return template * ontopof
        
    def applyInefficiencyTemplate(self, name, ontopof):
        #if other is None:
        #    other = self
        #if othername is None:
        #    othername = name

        ineff = self.Hdict[name]['HgenUNMATCH'].project('pt', 'dRbin').values(flow=True)
        total = self.Hdict[name]['Hgen'].project('pt', 'dRbin').values(flow=True)
        template = np.nan_to_num(ineff/total)

        #othertotal = other.Hdict[name]['Hgen'].project("pt", 'dRbin').values(flow=True)
        return template * ontopof

    def forward(self, name, other=None, othername=None, doTemplates=True):
        #if other is None:
        #    other = self
        #if othername is None:
        #    othername = name

        SdR = self.Hdict[name]['HtransSG'].project("ptReco", 'dRbinReco',
                                                   'ptGen', 'dRbinGen')
        SdR = SdR.values(flow=True)
        SdR = np.nan_to_num(SdR/np.sum(SdR, axis=(0,1)))

        F = self.Hdict[name]['HtransFR'].project('ptReco', 'dRbinReco')
        F = F.values(flow=True)

        T = np.einsum('ijkl,ij->ijkl', SdR, F)
        
        if doTemplates:
            Gtot = other.Hdict[othername]['Hgen'].project('pt','dRbin').values(flow=True)
            ineff = self.applyInefficiencyTemplate(name, Gtot)
            G = Gtot-ineff
        else:
            G=other.Hdict[othername]['HgenPure'].project('pt','dRbin').values(flow=True)

        vals = np.einsum('ijkl,kl->ij', T, G)
        if doTemplates:
            contamination = self.applyContaminationTemplate(name, vals)
            #PUjets = self.applyPUjetsTemplate(name, vals)
            #contamination = other.Hdict[othername]['HrecoUNMATCH'].project('pt', 'dRbin').values(flow=True)
            PUjets = other.Hdict[othername]['HrecoPUjets'].project('pt', 'dRbin').values(flow=True)
            vals = vals + contamination + PUjets
        return vals[None,:,:,None], np.zeros((12,52,12,52))

        S = self.getFullTransfer(name)

        oG = other.Hdict[othername]['HgenPure'].values(flow=True)
        oGerr = other.Hdict[othername]['HcovGenPure'].values(flow=True)
        
        print("vals")
        print(S.shape, oG.shape)
        vals = np.einsum('qijkl,eklp->eijp', S, oG)/4
        return vals, np.zeros((12,52,12,52))
        print(vals.shape)
        print('vars1')
        vars1 = np.einsum('eijkl,eklmn->eijmn', S, oGerr)
        print(vars1.shape)
        print('S2')
        S2 = np.einsum('eijkl->eklij', S)
        print(S2.shape)
        print("vars2")
        vars2 = np.einsum('eijkl,eklmn->eijmn', vars1, S2)
        print(vars2.shape)

        return vals, vars2

    def getFactors(self, name, ptbin, etabin, pubin, wrt):
        Hfact = self.Hdict[name]['HtransFR']
        if ptbin is not None:
            Hfact = Hfact[{'ptReco' : ptbin}]

        if etabin is not None:
            Hfact = Hfact[{'eta' : etabin}]

        if wrt == 'dR':
            Hfact = Hfact.project('dRbinReco')
        elif wrt == 'EECwt':
            Hfact = Hfact.project('EECwtReco')
        else:
            raise ValueError('wrt must be dR or EECwt')

        vals = Hfact.values(flow=True)
        errs = np.sqrt(Hfact.variances(flow=True))

        return vals, errs

    def getValsErrs(self, name, key, ptbin, etabin, pubin):
        vals = self.Hdict[name][key].values(flow=True)
        cov = self.Hdict[name][self.covnames[key]].values(flow=True)

        return self.project(vals, cov, ptbin, etabin, pubin)

    def getForwardValsErrs(self, name, other=None, othername=None, 
                           ptbin=None, etabin=None, pubin=None,
                           doTemplates=True):
        vals, cov = self.forward(name, other, othername, doTemplates=doTemplates)

        return self.project(vals, cov, ptbin, etabin, pubin)

    def project(self, vals, cov, ptbin, etabin, pubin):
        if etabin is not None:
            vals = vals[etabin+1, :, :, :]
            cov = np.zeros_like(cov)
            print("Warning: forgot to bin cov in eta, so it's bogus")
        else:
            vals = np.sum(vals, axis=0)

        if pubin is not None:
            vals = vals[:, :, pubin+1]
            cov = np.zeros_like(cov)
            print("Warning: forgot to bin cov in pu, so it's bogus")
        else:
            vals = np.sum(vals, axis=2)

        if ptbin is not None:
            vals = vals[ptbin+1, :]
            cov = cov[ptbin+1, :, ptbin+1, :]
        else:
            vals = np.sum(vals, axis=0)
            cov = np.sum(cov, axis=(0,2))

        return vals, np.sqrt(np.diag(cov))
