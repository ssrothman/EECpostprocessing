from coffea.nanoevents import NanoEventsFactory
import json
from RecursiveNamespace import RecursiveNamespace

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def figtee(res4, Rbin, rbin):
    fig, ax = plt.subplots(figsize=(20,20))

    plt.title("Tee", fontsize=40)

    edges_RL = np.linspace(0, 0.8, 9)
    edges_r = np.linspace(0, 1, 20)
    edges_phi = np.linspace(0, 0.5*np.pi, 20)
    centers_r = 0.5*(edges_r[1:] + edges_r[:-1])
    centers_phi = 0.5*(edges_phi[1:] + edges_phi[:-1])

    normclass = LogNorm
    cmap = 'viridis'

    Rmin = edges_RL[rbin-1]
    Rmax = edges_RL[rbin]

    rmin = edges_r[Rbin-1]
    rmax = edges_r[Rbin]

    text = '$%.1f < R < %.1f$\n$%.1f < r < %.1f$' % (Rmin, Rmax, rmin, rmax)
    ax.text(0.91, 0.98, text, fontsize=25, ha='center', va='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='ghostwhite', edgecolor='k', 
                      linewidth=3))

    area = (edges_r[1:,None]**2 - edges_r[:-1,None]**2) * (edges_phi[None,1:] - edges_phi[None,:-1])/2
    #area = np.ones_like(DATA[0,0,1:-(missing_r_bins+1),1:-1])
    normalized = res4[Rbin][1:-1,1:-1]/area

    dres4 = np.sqrt(res4)/200
    dnormalized = dres4[Rbin][1:-1,1:-1]/area

    angular_average = np.mean(normalized, axis=1, keepdims=True)
    ratio = normalized/angular_average
    dratio = dnormalized/angular_average

    width = edges_phi[1] - edges_phi[0]

    big_phi = np.concatenate([centers_phi, 
                              (np.pi-centers_phi)[::-1], 
                              np.pi+centers_phi, 
                              (2*np.pi-centers_phi)[::-1]])
    big_ratio = np.concatenate([ratio[rbin],
                                (ratio[rbin])[::-1],
                                ratio[rbin],
                                (ratio[rbin])[::-1]])
    big_dy = np.concatenate([dratio[rbin],
                             dratio[rbin][::-1],
                             dratio[rbin],
                             dratio[rbin][::-1]])

    ax.errorbar(big_phi, big_ratio, xerr=width/2, yerr=big_dy, fmt='o')

    def cos2phi(x, a, b):
        return a + b*np.cos(2*x)

    def linear(x, a, b):
        return a + b*x

    def cosphi(x, a, b):
        return a + b*np.cos(x)

    def cossqphi(x, a, b):
        return a + b*np.cos(x)**2

    from scipy.optimize import curve_fit

    ax.axhline(1, color='k', linestyle='--')

    popt_cos, pcov_cos = curve_fit(cos2phi, big_phi, big_ratio, sigma=big_dy)
    x = np.linspace(0, 2*np.pi, 100)
    cos_label = '$%0.2g + %0.2g \cos(2\phi)$' % (popt_cos[0], popt_cos[1])
    ax.plot(x, cos2phi(x, *popt_cos), label=cos_label)

    ax.legend(fontsize=20, loc='upper left')
    
    chisq_cos = np.sum((cos2phi(big_phi, *popt_cos) - big_ratio)**2/big_dy**2)
    chisq_linear = np.sum((1-big_ratio)**2/big_dy**2)

    print("Cosine fit: ", chisq_cos)
    print("Linear fit: ", chisq_linear)
    print("Linear fit / Cosine fit: ", chisq_linear/chisq_cos)

    plt.show()

def do(res4, shapenum, show=True, prefix=None):
    titles = [None, "Dipole", "Tee", None]


    SHAPE = res4.shape
    DATA = ak.to_numpy(res4)

    missing_r_bins = 0
    
    nbin_RL = SHAPE[0]
    print(nbin_RL)
    nbin_r = SHAPE[1]
    nbin_phi = SHAPE[2]

    edges_RL = np.linspace(0, 0.8, 9)
    print(edges_RL)
    edges_r = np.linspace(0, 1, nbin_r - 1)
    if(missing_r_bins>0):
        edges_r = edges_r[:-missing_r_bins]
    edges_phi = np.linspace(0, 0.5*np.pi, nbin_phi - 1)
    centers_r = 0.5*(edges_r[1:] + edges_r[:-1])
    centers_phi = 0.5*(edges_phi[1:] + edges_phi[:-1])
    print(SHAPE)
    print(edges_r)
    print(edges_phi)

    normclass = LogNorm
    #normclass = Normalize
    cmap = 'viridis'
    
    for drbin in range(1, 9):

        fig, ax = plt.subplots(figsize=(20,20), 
                               subplot_kw={'projection': 'polar'})

        plt.title(titles[shapenum], fontsize=40)

        Rmin = edges_RL[drbin-1]
        Rmax = edges_RL[drbin]
        text = '$%.1f < R < %.1f$' % (Rmin, Rmax)
        ax.text(0.91, 0.98, text, fontsize=25, ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', edgecolor='k', 
                          linewidth=3))

        area = (edges_r[1:,None]**2 - edges_r[:-1,None]**2) * (edges_phi[None,1:] - edges_phi[None,:-1])/2
        #area = np.ones_like(DATA[0,0,1:-(missing_r_bins+1),1:-1])

        pc = ax.pcolormesh(edges_phi, edges_r, 
                           DATA[drbin][1:-(missing_r_bins+1), 1:-1]/area,
                           cmap=cmap, norm=normclass())
        pc2 = ax.pcolormesh(np.pi-edges_phi, edges_r, 
                            DATA[drbin][1:-(missing_r_bins+1), 1:-1]/area,
                            cmap=cmap, norm=normclass())
        pc3 = ax.pcolormesh(np.pi+edges_phi, edges_r, 
                           DATA[drbin][1:-(missing_r_bins+1), 1:-1]/area,
                           cmap=cmap, norm=normclass())
        pc4 = ax.pcolormesh(2*np.pi-edges_phi, edges_r, 
                            DATA[drbin][1:-(missing_r_bins+1), 1:-1]/area,
                            cmap=cmap, norm=normclass())
        fig.colorbar(pc)

        if prefix is not None:
            plt.savefig("%s_%d.png"%(prefix, drbin), format='png',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    for drbin in range(1,9):
        angular_average = np.mean(DATA[drbin][1:-(missing_r_bins+1), 1:-1], axis=1, keepdims=True)
        ratio = DATA[drbin][1:-(missing_r_bins+1), 1:-1]/angular_average
        fig, ax = plt.subplots(figsize=(20,20), 
                               subplot_kw={'projection': 'polar'})
        plt.title("%s ratio to angular average" % titles[shapenum], fontsize=40)

        Rmin = edges_RL[drbin-1]
        Rmax = edges_RL[drbin]
        text = '$%.1f < R < %.1f$' % (Rmin, Rmax)
        ax.text(0.91, 0.98, text, fontsize=25, ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', edgecolor='k', 
                          linewidth=3))

        normclass = Normalize
        pc = ax.pcolormesh(edges_phi, edges_r, 
                           ratio,
                           cmap=cmap, norm=normclass())
        pc2 = ax.pcolormesh(np.pi-edges_phi, edges_r, 
                            ratio,
                            cmap=cmap, norm=normclass())
        pc3 = ax.pcolormesh(np.pi+edges_phi, edges_r,
                            ratio,
                            cmap=cmap, norm=normclass())
        pc4 = ax.pcolormesh(2*np.pi-edges_phi, edges_r,
                            ratio,
                            cmap=cmap, norm=normclass())
        fig.colorbar(pc)
        if prefix is not None:
            plt.savefig("%s_ratio2d_%d.png"%(prefix, drbin), format='png',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    for drbin in range(1,9):
        angular_average = np.mean(DATA[drbin][1:-(missing_r_bins+1), 1:-1], axis=1, keepdims=True)
        ratio = DATA[drbin][1:-(missing_r_bins+1), 1:-1]/angular_average

        fig, ax = plt.subplots(figsize=(20,10))
        plt.title(titles[shapenum], fontsize=40)

        Rmin = edges_RL[drbin-1]
        Rmax = edges_RL[drbin]
        text = '$%.1f < R < %.1f$' % (Rmin, Rmax)
        ax.text(0.91, 0.98, text, fontsize=25, ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', edgecolor='k', 
                          linewidth=3))
        
        for r in [0, 4, 9, 14, 18]:
            left_r = edges_r[r]
            right_r = edges_r[r+1]

            big_centers = np.concatenate([centers_phi, 
                                          (np.pi-centers_phi)[::-1], 
                                          np.pi+centers_phi,
                                          (2*np.pi-centers_phi)[::-1]])
            big_ratio = np.concatenate([ratio[r, :], 
                                        ratio[r, ::-1], 
                                        ratio[r, :], 
                                        ratio[r, ::-1]])

            plt.plot(big_centers, big_ratio, label="%.2f < r < %.2f" % (left_r, right_r))
        
        plt.legend(loc ='upper left')
        if prefix is not None:
            plt.savefig("%s_ratio1d_%d.png"%(prefix, drbin), format='png',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    fig, ax = plt.subplots(figsize=(20,10))
    plt.title(titles[shapenum], fontsize=40)

    angular_average = np.mean(DATA[1:-(missing_r_bins+1), 1:-1], axis=(2), keepdims=False)
    N = np.mean(angular_average, axis=1, keepdims=True)
    angular_average /= N
    for R in range(1,9):
        Rmin = edges_RL[R-1]
        Rmax = edges_RL[R]

        plt.plot(centers_r, angular_average[R-1, :], 
                 label='%.1f < R < %.1f' % (Rmin, Rmax))
    plt.show()
