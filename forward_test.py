import pickle
import numpy as np

with open("Oct26_2023_EECcorr/EEC/hists.pkl", 'rb') as f:
    x = pickle.load(f)

name = 'FancyEEC'
transS = x[name]['HtransS']
transF = x[name]['HtransF']

Svals = transS.values(flow=True)
Sdenom = np.sum(Svals, axis=(0,1))
Svals = Svals/Sdenom
np.nan_to_num(Svals, copy=False)

Fvals = transF.values(flow=True)
Fvals = np.sum(Fvals, axis=2)

factors = transF.axes['factor'].centers
meanfactors = np.sum(Fvals[:,:,1:-1] * factors[None,None,:], axis=-1)/np.sum(Fvals[:,:,1:-1], axis=-1)
np.nan_to_num(meanfactors, copy=False)
print(np.mean(meanfactors))

print(meanfactors.shape)
print(Svals.shape)
transfer = Svals#np.einsum('ijkl,kl->ijkl', Svals, meanfactors)

x[name]['HgenPure'] = x[name]['Hgen'] - x[name]['HgenUNMATCH']
x[name]['HrecoPure'] = x[name]['Hreco'] - x[name]['HrecoUNMATCH'] - x[name]['HrecoPUjets']

gen = x[name]['HgenPure'].values(flow=True)
reco = x[name]['HrecoPure'].values(flow=True)


print()
print(gen.shape)
print(transfer.shape)
print(reco.shape)

transfered = np.einsum('ijkl,kl->ij', transfer, gen)
print()
print(np.max(transfered - reco))
print(np.min(transfered - reco))

meanfactors2 = reco/transfered
np.nan_to_num(meanfactors2, copy=False)

transfer2 = transfer*meanfactors2[:,:,None,None]
transfered2 = np.einsum('ijkl,kl->ij', transfer2, gen)
print()
print(np.max(transfered2 - reco))
print(np.min(transfered2 - reco))


print()
print(meanfactors2[1])
print(meanfactors[1])
print()
