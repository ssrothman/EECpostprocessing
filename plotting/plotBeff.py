import matplotlib.pyplot as plt
import numpy as np
from plotting.util import *


def plotSingleEfficiency(data, taglevel, flavor, eta, label, ax):
    Hproj = data['Beff'][{'genflav' : flavor, 'eta' : eta}].project('pt', 'btag_'+taglevel)

    values = Hproj.values(flow=True)
    variances = Hproj.variances(flow=True)

    Bpass = values[:, 1]
    Bfail = values[:, 0]

    eff = Bpass/(Bpass + Bfail)
    eff = np.nan_to_num(eff, nan=0, posinf=0, neginf=0)

    edges = np.asarray(Hproj.axes[0].edges.tolist() + [1e4])
    widths = np.diff(edges)
    centers = (edges[1:] + edges[:-1])/2

    print(centers)
    print(eff)

    ax.errorbar(centers, eff, 
                xerr = widths/2, 
                yerr=0, 
                fmt='o', label='%s'%(label))

    ax.set_xlabel('Jet $p_T$ [GeV]')
    ax.set_ylabel('B-tagging efficiency')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='black', linestyle='--')

def plotBarrelEndcap(data, taglevel, flavor, folder=None):
    fig, ax = setup_plain()

    add_cms_info(ax, True)

    text = 'deepJet %s WP\n'%taglevel
    if flavor == 0:
        text += 'udsg'
    elif flavor == 1:
        text += 'c'
    elif flavor == 2:
        text += 'b'

    text += ' jets'

    ax.text(0.05, 0.85, text, transform=ax.transAxes, 
            verticalalignment='top')

    plotSingleEfficiency(data, taglevel, flavor, 0, 'Barrel', ax)
    plotSingleEfficiency(data, taglevel, flavor, 1, 'Endcap', ax)

    ax.legend()

    if folder is not None:
        savefig(os.path.join(folder, 
                             'BarrelEndcap_%s_%s'%(taglevel, flavor)))

    plt.show()

def plotTaglevel(data, flavor, eta, folder):
    fig, ax = setup_plain()
    add_cms_info(ax, True)

    text = ''
    if eta == 0:
        text = 'Barrel\n'
    elif eta == 1:
        text = 'Endcap\n'

    if flavor == 0:
        text += ' udsg jets'
    elif flavor == 1:
        text += ' c jets'
    elif flavor == 2:
        text += ' b jets'

    ax.text(0.05, 0.85, text, transform=ax.transAxes,
            verticalalignment='top')

    plotSingleEfficiency(data, 'tight', flavor, eta, 'tight', ax)
    plotSingleEfficiency(data, 'medium', flavor, eta, 'medium', ax)
    plotSingleEfficiency(data, 'loose', flavor, eta, 'loose', ax)

    ax.legend()

    if folder is not None:
        savefig(os.path.join(folder, 'Taglevel_%s_%s'%(flavor, eta)))

    plt.show()
