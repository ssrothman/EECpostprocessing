import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pythia_base = '/data/submit/srothman/EEC/Apr24_2024/DYJetsToLL/EECproj/hists_file0to1825_tight/'
herwig_base = '/data/submit/srothman/EEC/Apr24_2024/DYJetsToLL_Herwig/EECproj/hists_file0to605_tight/'

T_pythia = np.load("/work/submit/srothman/EEC/EECunfold/tPrPgP_P/transfer.npy")
T_herwig = np.load("/work/submit/srothman/EEC/EECunfold/tHrHgH_H/transfer.npy")

R_pythia = np.load("%sreco.npy" % pythia_base)
Rp_pythia = np.load("%srecopure.npy" % pythia_base)

R_herwig = np.load("%sreco.npy" % herwig_base)
Rp_herwig = np.load("%srecopure.npy" % herwig_base)

G_pythia = np.load("%sgen.npy" % pythia_base)
Gp_pythia = np.load("%sgenpure.npy" % pythia_base)

G_herwig = np.load("%sgen.npy" % herwig_base)
Gp_herwig = np.load("%sgenpure.npy" % herwig_base)

Gtemplate_pythia = np.nan_to_num(Gp_pythia / G_pythia)
Gtemplate_herwig = np.nan_to_num(Gp_herwig / G_herwig)

Rtemplate_pythia = np.nan_to_num(R_pythia / Rp_pythia)
Rtemplate_herwig = np.nan_to_num(R_herwig / Rp_herwig)

import matplotlib.pyplot as plt

q = 2*6*28
print(Gtemplate_pythia.shape)
N = np.prod(Gtemplate_pythia.shape)
T_pythia = T_pythia.reshape((N, N))
T_herwig = T_herwig.reshape((N, N))

plt.pcolormesh(np.log(T_pythia[:q,:q]/T_herwig[:q,:q]), cmap='coolwarm',
               vmin=-1, vmax=1)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.pcolormesh(T_pythia[:q,:q], cmap='viridis', norm=LogNorm())
ax2.pcolormesh(T_herwig[:q,:q], cmap='viridis', norm=LogNorm())
plt.show()


#plt.plot(Gtemplate_herwig.ravel()/Gtemplate_pythia.ravel())
plt.plot(Gtemplate_pythia[0].ravel()/Gtemplate_herwig[0].ravel())
plt.ylim(0.5, 2.0)
plt.show()

plt.plot(Gtemplate_pythia.ravel(), label='Pythia')
plt.plot(Gtemplate_herwig.ravel(), label='Herwig')
plt.legend()
plt.show()

plt.show()
