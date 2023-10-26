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

    def getValsErrs(self, name, key, ptbin):
        vals = self.Hdict[name][key]
        cov = self.Hdict[name][self.covnames[key]]

        if ptbin is not None:
            vals = vals[{'pt':ptbin}]
            cov = cov[{'pt1':ptbin, 'pt2':ptbin}]

        vals = vals.project('dRbin')
        cov = cov.project('dRbin1', 'dRbin2')
        
        vals = vals.values(flow=False)
        errs = np.sqrt(np.diag(cov.values(flow=False)))

        return vals, errs

    def makeTransfer(self, name, ptbin):
        if self.includeInefficiencies:
            Hgen = self.Hdict[name]['Hgen']
            HcovGen = self.Hdict[name]['HcovGen']
        else:
            Hgen = Hdict['HgenPure']
            HcovGen = Hdict['HcovGenPure']

        Hgen = Hgen[{'pt':ptbin}].project('dRbin')
        HcovGen = HcovGen[{'pt1':ptbin, 'pt2':ptbin}].project('dRbin1', 'dRbin2')

        Htrans = Hdict['Htrans']
        Htrans = Htrans[{'ptReco':ptbin, 'ptGen':ptbin}].project('dRbinReco','dRbinGen')
        
        transValue = Htrans.values(flow=False)
        genValue = Hgen.values(flow=False)

        target = np.sum(transValue, axis=1)

        transValue = transValue / genValue[None, :]

        return transValue

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
            print("branch1")
        else:
            gen = self.Hdict[name]['HgenPure']
            print('branch2')

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
