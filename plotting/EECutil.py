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
            self.Hdict[name]['HcovRecoPure'] = self.Hdict[name]['HcovReco'] - self.Hdict[name]['HcovRecoUNMATCH']
            self.Hdict[name]['HgenPure'] = self.Hdict[name]['Hgen'] - self.Hdict[name]['HgenUNMATCH']
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

        if ptbin is not None:
            Hproj = Hproj[{'pt':ptbin}]
            Hcov = Hcov[{'pt_1':ptbin, 'pt_2':ptbin}]
        else:
            Hproj = project_out(Hproj, ['pt'])
            Hcov = project_out(Hcov, ['pt_1', 'pt_2'])

        if pubin is not None:
            Hproj = Hproj[{'nPU':pubin}]
            Hcov = Hcov[{'nPU_1':pubin, 'nPU_2':pubin}]
        else:
            Hproj = project_out(Hproj, ['nPU'])
            Hcov = project_out(Hcov, ['nPU_1', 'nPU_2'])

        values = Hproj.values(flow=True)
        covariance = Hcov.values(flow=True)
        errs = np.sqrt(np.einsum('ii->i', covariance))

        return values, errs
