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

    def getDRtransfer(self, name):
        S = self.Hdict[name]['HtransSG'].values(flow=True)
        S = np.nan_to_num(S/np.sum(S, axis=(0,1)))
        return S

    def getFullTransfer(self, name):
        S = self.getDRtransfer(name)
        factors = self.Hdict[name]['HtransFR'].project('ptReco', 'dRbinReco')
        factors = factors.values(flow=True)
        return np.einsum('ij,ijkl->ijkl', factors, S)

    def forward(self, name, other=None, othername=None):
        if other is None:
            other = self
        if othername is None:
            othername = name

        S = self.getFullTransfer(name)

        oG = other.Hdict[othername]['HgenPure'].values(flow=True)
        oGerr = other.Hdict[othername]['HcovGenPure'].values(flow=True)
        
        print("vals")
        vals = np.einsum('ijkl,kl->ij', S, oG)
        print(vals.shape)
        print('vars1')
        vars1 = np.einsum('ijkl,klmn->ijmn', S, oGerr)
        print(vars1.shape)
        print('S2')
        S2 = np.einsum('ijkl->klij', S)
        print(S2.shape)
        print("vars2")
        vars2 = np.einsum('ijkl,klmn->ijmn', vars1, S2)
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
                           ptbin=None, etabin=None, pubin=None):
        vals, cov = self.forward(name, other, othername)

        return self.project(vals, cov, ptbin, etabin, pubin)

    def project(self, vals, cov, ptbin, etabin, pubin):
        if etabin is not None:
            vals = vals[etabin+1, :, :, :]
            print("Warning: forgot to bin cov in eta, so it's bogus")
        else:
            vals = np.sum(vals, axis=0)

        if pubin is not None:
            vals = = vals[:, :, pubin+1]
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

class EEC_binwt(object):
    def __init__(self, path, includeInefficiencies=True):
        self.path = path
        self.Hdict = self.setup_Hdict(path)
        self.includeInefficiencies = includeInefficiencies

    def getWeights(self, name, key, ptbin, dRbin):
        vals = self.Hdict[name][key]

        if ptbin is not None:
            vals = vals[ptbin+1, :, :]
        else:
            vals =  np.sum(vals, axis=0)

        if dRbin is not None:
            vals = vals[dRbin, :]
        else:
            vals = np.sum(vals, axis=0)

        return vals

    def getValsErrs(self, name, key, ptbin):
        if key == 'forward':
            vals = self.forwardTransfer(name)
        else:
            vals = self.Hdict[name][key]
        
        if ptbin is not None:
            vals = vals[ptbin+1, :, :]
        else:
            vals =  np.sum(vals, axis=0)

        wts = wtaxis.centers
        vals = np.sum(vals[:,1:-1] * wts[None,:], axis=1)

        return vals, np.zeros_like(vals)

    def forwardTransfer(self, name, genmat=None):
        trans = self.getRawTransfer(name)
        if genmat is None:
            if self.includeInefficiencies:
                genmat = self.Hdict[name]['Hgen']
            else:
                genmat = self.Hdict[name]['HgenPure']
        return np.einsum('ijklmn,lmn->ijk', trans, genmat)

    def getRawTransfer(self, name):
        trans = self.Hdict[name]['Htrans']
        if self.includeInefficiencies:
            gen = self.Hdict[name]['Hgen']
        else:
            gen = self.Hdict[name]['HgenPure']

        transValue = trans.copy()
        invgen = 1/gen
        invgen[gen==0] = 0
        transValue = np.einsum('ijklmn,lmn->ijklmn',transValue, invgen)

        return transValue

    def getTransfer(self, name, ptbin):
        trans = self.Hdict[name]['Htrans']
        if self.includeInefficiencies:
            gen = self.Hdict[name]['Hgen']
        else:
            gen = self.Hdict[name]['HgenPure']

        if ptbin is not None:
            gen = gen[ptbin+1, :, :]
            trans = trans[ptbin+1, :, :, ptbin+1, :, :]
        else:
            gen =  np.sum(gen, axis=0)
            trans = np.sum(trans, axis=(0,3))

        transValue = trans.copy()

        for i in range(trans.shape[2]):
            for j in range(trans.shape[3]):
                if gen[i,j] > 0:
                    transValue[:,:,i,j]/=gen[i,j]
                elif np.sum(trans[:,:,i,j] > 0):
                    transValue[:,:,i,j]=0
                    print("bad", i, j)

        return transValue

    @staticmethod
    def sliceTransfer(trans, otherbin, which):
        if which == 'wt':
            if otherbin is not None:
                trans = trans[otherbin,:,otherbin,:]
            else:
                trans = np.sum(trans, axis=(0, 2))
        else:
            if otherbin is not None:
                trans = trans[:, otherbin, :, otherbin]
            else:
                trans = np.sum(trans, axis=(1, 3))
        return trans

    def getSlicedTransfer(self, name, ptbin, otherbin, which):
        return self.sliceTransfer(self.getTransfer(name, ptbin), otherbin, which)

    @staticmethod
    def load_proj(path):
        values = np.memmap(path, dtype=np.float64,
                           mode='c', shape=(12, 52, 27))
        return values

    @staticmethod
    def load_transfer(path):
        values = np.memmap(path, dtype=np.float64,
                           mode='c', shape=(12,52,27,12,52,27))
        return values

    @staticmethod
    def setup_Hdict(basepath):
        Hdict = {}
        for name in os.listdir(basepath):
            Hdict[name ] = {}
            for key in os.listdir(os.path.join(basepath, name)):
                if 'trans' in key:
                    Hdict[name][key[:-4]]=EEC_binwt.load_transfer(
                            os.path.join(basepath,name,key))
                else:
                    Hdict[name][key[:-4]]=EEC_binwt.load_proj(
                            os.path.join(basepath,name,key))
        return Hdict
