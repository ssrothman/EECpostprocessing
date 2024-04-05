import matplotlib.pyplot as plt
from hist import rebin
import seaborn as sns
import numpy as np
import pickle
import matplotlib

def getprobs(vals):
    sums = np.sum(vals, axis=2)
    
    pl = np.nan_to_num(vals[:,:,0]/sums)
    pc = np.nan_to_num(vals[:,:,1]/sums)
    pb = np.nan_to_num(vals[:,:,2]/sums)

    return pl, pc, pb

def getdists(vals):
    pl = vals[:,:,0]
    pc = vals[:,:,1]
    pb = vals[:,:,2]

    pl = pl/np.sum(pl)
    pc = pc/np.sum(pc)
    pb = pb/np.sum(pb)

    return pl, pc, pb

def taggerdists(H):
    cmap = 'Reds'

    #norm = matplotlib.colors.Normalize(0, 1)
    norm = matplotlib.colors.LogNorm()
    getfun = getprobs
    #getfun = getdists

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = \
            plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)

    vals = H[{'pt' : 0}].project('CvL', 'CvB', 'genflav')[::rebin(5),::rebin(5),:].values()
    pl, pc, pb = getfun(vals)

    sns.heatmap(pl, annot=False, square=True, ax=ax0,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pc, annot=False, square=True, ax=ax1,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pb, annot=False, square=True, ax=ax2,
                norm=norm, cbar=False,
                cmap=cmap)


    vals = H[{'pt' : 1}].project('CvL', 'CvB', 'genflav')[::rebin(5),::rebin(5),:].values()
    pl, pc, pb = getfun(vals)

    sns.heatmap(pl, annot=False, square=True, ax=ax3,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pc, annot=False, square=True, ax=ax4,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pb, annot=False, square=True, ax=ax5,
                norm=norm, cbar=False,
                cmap=cmap)

    vals = H[{'pt' : 2}].project('CvL', 'CvB', 'genflav')[::rebin(5),::rebin(5),:].values()
    pl, pc, pb = getfun(vals)

    sns.heatmap(pl, annot=False, square=True, ax=ax6,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pc, annot=False, square=True, ax=ax7,
                norm=norm, cbar=False,
                cmap=cmap)
    sns.heatmap(pb, annot=False, square=True, ax=ax8,
                norm=norm, cbar=False,
                cmap=cmap)

    ax0.set_title("Light jets")
    ax1.set_title("Charm jets")
    ax2.set_title("Bottom jets")

    ax0.set_ylabel("CvL")
    ax0.set_xlabel("CvB")
    ax1.set_xlabel("CvB")
    ax2.set_xlabel("CvB")

    A = 0.1
    m = 3
    x = np.linspace(0, 1, 100)
    y = m*x + A
    ax0.plot(x*10, y*10, c='k')
    ax1.plot(x*10, y*10, c='k')
    ax2.plot(x*10, y*10, c='k')
    ax3.plot(x*10, y*10, c='k')
    ax4.plot(x*10, y*10, c='k')
    ax5.plot(x*10, y*10, c='k')
    ax6.plot(x*10, y*10, c='k')
    ax7.plot(x*10, y*10, c='k')
    ax8.plot(x*10, y*10, c='k')
    B = 0.7
    x2 = np.linspace((B-A)/m, 1, 100)
    y2 = np.ones_like(x2)*B
    ax0.plot(x2*10, y2*10, c='k')
    ax1.plot(x2*10, y2*10, c='k')
    ax2.plot(x2*10, y2*10, c='k')
    ax3.plot(x2*10, y2*10, c='k')
    ax4.plot(x2*10, y2*10, c='k')
    ax5.plot(x2*10, y2*10, c='k')
    ax6.plot(x2*10, y2*10, c='k')
    ax7.plot(x2*10, y2*10, c='k')
    ax8.plot(x2*10, y2*10, c='k')


    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax0.get_children()[0], cax=cbar_ax)

    fig.suptitle("Flavor fractions in each discriminator bin")

    plt.show()


def confusion3d(H):
    vals = H.project("B", 'CvL', 'CvB', 'genflav')[::rebin(2), ::rebin(2), ::rebin(2), :].values()
    pl1, pc1, pb1 = getprobs(np.sum(vals, axis=0))
    pl2, pc2, pb2 = getprobs(np.sum(vals, axis=1))
    pl3, pc3, pb3 = getprobs(np.sum(vals, axis=2))

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) =\
            plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
                                        
    sns.heatmap(pl1, annot=False, square=True, ax=ax0,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pc1, annot=False, square=True, ax=ax1,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pb1, annot=False, square=True, ax=ax2,
                vmin=0, vmax=1, cbar=False)

    ax0.set_ylabel("CvL")
    ax0.set_xlabel("CvB")
    ax1.set_xlabel("CvB")
    ax2.set_xlabel("CvB")

    ax0.set_title("Light jet probability")
    ax1.set_title("Charm jet probability")
    ax2.set_title("Bottom jet probability")

    sns.heatmap(pl2, annot=False, square=True, ax=ax3,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pc2, annot=False, square=True, ax=ax4,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pb2, annot=False, square=True, ax=ax5,
                vmin=0, vmax=1, cbar=False)

    ax3.set_ylabel("B")
    ax3.set_xlabel("CvB")
    ax4.set_xlabel("CvB")
    ax5.set_xlabel("CvB")
    
    sns.heatmap(pl3, annot=False, square=True, ax=ax6,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pc3, annot=False, square=True, ax=ax7,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pb3, annot=False, square=True, ax=ax8,
                vmin=0, vmax=1, cbar=False)

    ax6.set_ylabel("B")
    ax6.set_xlabel("CvL")
    ax7.set_xlabel("CvL")
    ax8.set_xlabel("CvL")


    plt.tight_layout()

    plt.show()

def confusion(H):
    vals = H.project("ctag", 'btag', 'genflav').values()


    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), 
                                        sharex=True, sharey=True)
                                        
    sns.heatmap(pl, annot=True, square=True, ax=ax0,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pc, annot=True, square=True, ax=ax1,
                vmin=0, vmax=1, cbar=False)
    sns.heatmap(pb, annot=True, square=True, ax=ax2,
                vmin=0, vmax=1, cbar=False)

    ax0.set_ylabel("ctag")
    ax0.set_xlabel("btag")
    ax1.set_xlabel("btag")
    ax2.set_xlabel("btag")

    ax0.set_title("Light jet probability")
    ax1.set_title("Charm jet probability")
    ax2.set_title("Bottom jet probability")
    
    fig.suptitle("Jet flavor confusion given tight tags")

    plt.show()
