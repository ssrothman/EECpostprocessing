print("start")
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
print("imported dataset logic")
import simonplot as splt
print("imported simonplot")
import numpy as np
import matplotlib.pyplot as plt
import hist
import simonplot.util.common as splt_common
import simonplot.util.histplot as splt_histplot
from simonplot.util.comparison import ComparisonHistStruct

def run_plots(MC_, DATA_, binlabels_, bincuts_, bincolors_):

    fig = splt_common.setup_canvas()
    ax = splt_common.make_oneax(fig)
    splt_common.add_cms_legend(ax, True, DATA_.lumi)


    for i, (cut, label, color) in enumerate(zip(bincuts_, binlabels_, bincolors_)):
        print(f"running for {label}")
        HMC = MC_.fill_hist(
            var,
            cut,
            wt,
            axis
        )
        HDATA = DATA_.fill_hist(
            var, 
            cut,
            wt,
            axis
        )

        thresh = HMC.axes[0].index(1.0) + 1 # track pT threshold above which we normalize

        MCintegral = np.sum(HMC.values(flow=True)[thresh:])
        dataintegral = np.sum(HDATA.values(flow=True)[thresh:])

        HMC = HMC * dataintegral / MCintegral

        Hratio = ComparisonHistStruct(
            HDATA,
            HMC,
            'ratio'
        )

        splt_histplot.simon_histplot(
            Hratio, 
            ax, 
            density=False,
            fillbetween=None,
            label = label,
            color = color
        )

    ax.set_xscale('log')
    plt.axhline(1.0, c='k', ls='--')
    plt.ylim(0.0, 2.0)
    ax.legend()
    ax.set_xlabel('track pT (GeV)')
    ax.set_ylabel('Data / MC')

    plt.savefig('test.png')
    plt.close(fig)

    print("done(?)")


print("About to build MC dataset")
MC = build_pq_dataset(
    configsuite='VetoConfig2',
    runtag='Mar_01_2026',
    dataset='Pythia_inclusive',
    objsyst='nominal',
    table='parts',
    location='xrootd-submit'
)

print("About to build DATA dataset")
DATA = build_pq_dataset(
    configsuite='VetoConfig2',
    runtag='Mar_01_2026',
    dataset='DATA_2018C',
    objsyst='DATA',
    table='parts',
    location='xrootd-submit'
)
print("computing weight")
MC.compute_weight(DATA.lumi)

var = splt.variable.BasicVariable('pt')
wt = splt.variable.BasicVariable('wt_nominal')

axis = hist.axis.Regular(
    100, 1e-1, 1e2,
    transform = hist.axis.transform.log
)

bincuts = [
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 0.0, 0.3),
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 0.3, 0.6),
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 0.6, 0.9),
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 0.9, 1.2),
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 1.2, 1.5),
    splt.cut.TwoSidedCut(splt.variable.AbsVariable('eta'), 1.5, 1.8),
    splt.cut.GreaterThanCut(splt.variable.AbsVariable('eta'), 1.8)
]
binlabels = [
    '$0.0 < |\\eta| < 0.3$',
    '$0.3 < |\\eta| < 0.6$',
    '$0.6 < |\\eta| < 0.9$',
    '$0.9 < |\\eta| < 1.2$',
    '$1.2 < |\\eta| < 1.5$',
    '$1.5 < |\\eta| < 1.8$',
    '$|\\eta| > 1.8$'
]
cmap = plt.cm.viridis
bincolors = [
    cmap(i/(len(binlabels)-1)) for i in range(len(binlabels))
]

#run_plots(MC, DATA, binlabels, bincuts, bincolors)


bincuts = [
    splt.cut.TwoSidedCut('NPV', 0, 20),
    splt.cut.TwoSidedCut('NPV', 20, 40),
    splt.cut.TwoSidedCut('NPV', 40, 60),
    splt.cut.GreaterThanCut('NPV', 60)
]
binlabels = [
    '$0 < N_{PV} < 20$',
    '$20 < N_{PV} < 40$',
    '$40 < N_{PV} < 60$',
    '$N_{PV} > 60$'
]
bincolors = [
    cmap(i/(len(binlabels)-1)) for i in range(len(binlabels))
]
run_plots(MC, DATA, binlabels, bincuts, bincolors)