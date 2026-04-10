from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack, load_prebinned_dataset, build_prebinned_dataset_stack, load_prebinned_root_histogram
import simonplot as splt
import uproot
from simonplot.util.common import setup_canvas, add_cms_legend, make_oneax, savefig
import matplotlib.pyplot as plt

MC = build_pq_dataset_stack(
    configsuite = 'BasicConfig',
    runtag = 'Mar_01_2026',
    dataset = 'allMC',
    objsyst = 'nominal',
    table = 'events',
    location = 'scratch-submit'
)
MC.compute_weight(59.54)

wt_nom = splt.variable.BasicVariable('wt_nominal')
wt_up = splt.variable.BasicVariable('wt_PUUp')
wt_dn = splt.variable.BasicVariable('wt_PUDown')

cut = splt.cut.AndCuts([
    splt.cut.EqualsCut('nMu', 2),
    splt.cut.EqualsCut('nEle', 0),
    splt.cut.LessThanCut('numMediumB', 2),
    splt.cut.TwoSidedCut('Zmass', 91.1876 - 15, 91.1876 + 15)
])

var = splt.variable.BasicVariable('nTrueInt')

binning = splt.binning.BasicBinning(200, -0.5, 199.5)

ax = binning.build_axis(var)

H = uproot.open('../CMSSW_10_6_26/src/SRothman/Analysis/production/crab/Lumis_Mar_01_2026/total/pileupHist.root:pileup').to_hist() # type: ignore 

Hnom = MC.fill_hist(var, cut, wt_nom, ax)
Hup = MC.fill_hist(var, cut, wt_up, ax)
Hdn = MC.fill_hist(var, cut, wt_dn, ax)

N = H.sum().value
Nnom = Hnom.sum().value
Nup = Hup.sum().value
Ndn = Hdn.sum().value

import numpy as np

centers = np.arange(200)
widths = np.ones_like(centers)

fig = setup_canvas()
ax_main = make_oneax(fig)
add_cms_legend(ax_main, True, lumi=59.54) # pyright: ignore[reportCallIssue, reportArgumentType]

ax_main.errorbar(
    centers, H.values()/N,
    yerr = np.sqrt(H.variances())/N,
    xerr = widths / 2,
    fmt='o',
    label = 'data',
    c = 'k',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hnom.values()/Nnom,
    yerr = np.sqrt(Hnom.variances())/Nnom,
    xerr = widths / 2,
    fmt='o',
    label = 'nominal',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hup.values()/Nup,
    yerr = np.sqrt(Hup.variances())/Nup,
    xerr = widths / 2,
    fmt='o',
    label = 'up',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hdn.values()/Ndn,
    yerr = np.sqrt(Hdn.variances())/Ndn,
    xerr = widths / 2,
    fmt='o',
    label = 'down',
    markersize=4, capsize=1, 
)

ax_main.legend()
ax_main.set_yscale('log')
ax_main.set_xlabel('nTrueInt')
ax_main.set_ylabel("Density")
savefig(fig,'ANplots/PU/nTrueInt_wdata')
plt.close(fig)

fig = setup_canvas()
ax_main = make_oneax(fig)
add_cms_legend(ax_main, True, lumi=59.54) # pyright: ignore[reportCallIssue, reportArgumentType]

ax_main.errorbar(
    centers[:80], H.values()[:80]/N,
    yerr = np.sqrt(H.variances()[:80])/N,
    xerr = widths[:80] / 2,
    fmt='o',
    label = 'data',
    c = 'k',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers[:80], Hnom.values()[:80]/Nnom,
    yerr = np.sqrt(Hnom.variances()[:80])/Nnom,
    xerr = widths[:80] / 2,
    fmt='o',
    label = 'nominal',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers[:80], Hup.values()[:80]/Nup,
    yerr = np.sqrt(Hup.variances()[:80])/Nup,
    xerr = widths[:80] / 2,
    fmt='o',
    label = 'up',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers[:80], Hdn.values()[:80]/Ndn,
    yerr = np.sqrt(Hdn.variances()[:80])/Ndn,
    xerr = widths[:80] / 2,
    fmt='o',
    label = 'down',
    markersize=4, capsize=1, 
)

ax_main.legend()
ax_main.set_yscale('log')
ax_main.set_xlabel('nTrueInt')
ax_main.set_ylabel("Density")
savefig(fig,'ANplots/PU/nTrueInt_wdata_narrow')
plt.close(fig)






DATA = build_pq_dataset_stack(
    configsuite = 'BasicConfig',
    runtag = 'Mar_01_2026',
    dataset = 'DATA',
    objsyst = 'DATA',
    table = 'events',
    location = 'scratch-submit',
    no_count = True
)
DATA.compute_weight(59.54)


var_rho = splt.variable.BasicVariable('rho')

binning = splt.binning.BasicBinning(100, 0, 80)
ax = binning.build_axis(var_rho)

Hrho_data = DATA.fill_hist(var_rho, cut, wt_nom, ax)
Hrho_nom = MC.fill_hist(var_rho, cut, wt_nom, ax)
Hrho_up = MC.fill_hist(var_rho, cut, wt_up, ax)
Hrho_dn = MC.fill_hist(var_rho, cut, wt_dn, ax)

N_data = Hrho_data.sum().value
N_nom = Hrho_nom.sum().value
N_up = Hrho_up.sum().value
N_dn = Hrho_dn.sum().value

centers = Hrho_data.axes[0].centers
widths = Hrho_data.axes[0].widths

fig = setup_canvas()
ax_main = make_oneax(fig)
add_cms_legend(ax_main, True, lumi=59.54) # pyright: ignore[reportCallIssue, reportArgumentType]

ax_main.errorbar(
    centers, Hrho_data.values()/N_data,
    yerr = np.sqrt(Hrho_data.variances())/N_data,
    xerr = widths / 2,
    fmt='o',
    label = 'data',
    c = 'k',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hrho_nom.values()/N_nom,
    yerr = np.sqrt(Hrho_nom.variances())/N_nom,
    xerr = widths / 2,
    fmt='o',
    label = 'nominal',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hrho_up.values()/N_up,
    yerr = np.sqrt(Hrho_up.variances())/N_up,
    xerr = widths / 2,
    fmt='o',
    label = 'up',
    markersize=4, capsize=1, 
)
ax_main.errorbar(
    centers, Hrho_dn.values()/N_dn,
    yerr = np.sqrt(Hrho_dn.variances())/N_dn,
    xerr = widths / 2,
    fmt='o',
    label = 'down',
    markersize=4, capsize=1, 
)

ax_main.legend()
ax_main.set_yscale('log')
ax_main.set_xlabel('rho')
ax_main.set_ylabel("Density")
savefig(fig,'ANplots/PU/rho_wdata')
plt.close(fig)
