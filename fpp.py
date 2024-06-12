from coffea.nanoevents import NanoEventsFactory
import json
from RecursiveNamespace import RecursiveNamespace

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def do3d(res4):
    titles = [None, "Dipole", "Tee", None]

    SHAPE = res4.shape
    DATA = ak.to_numpy(res4)

    missing_r_bins = 0

    nbin_r = SHAPE[2]
    nbin_phi = SHAPE[3]

    edges_r = np.linspace(0, 1, nbin_r - 1)
    if(missing_r_bins>0):
        edges_r = edges_r[:-missing_r_bins]
    edges_phi = np.linspace(0, 0.5*np.pi, nbin_phi - 1)
    centers_r = 0.5*(edges_r[1:] + edges_r[:-1])
    centers_phi = 0.5*(edges_phi[1:] + edges_phi[:-1])
    print(SHAPE)
    print(edges_r)
    print(edges_phi)

    R, PHI = np.meshgrid(centers_r, centers_phi, indexing='ij')

    X1 = R*np.cos(PHI)
    Y1 = R*np.sin(PHI)

    X2 = R*np.cos(np.pi - PHI)
    Y2 = R*np.sin(np.pi - PHI)

    X3 = R*np.cos(np.pi + PHI)
    Y3 = R*np.sin(np.pi + PHI)

    X4 = R*np.cos(2*np.pi - PHI)
    Y4 = R*np.sin(2*np.pi - PHI)

    X = np.concatenate([X1, X2, X3, X4], axis=1)
    Y = np.concatenate([Y1, Y2, Y3, Y4], axis=1)

    #plt.plot(X, Y, 'o')
    #plt.show()

    for shapenum in [1, 2]:
        normclass = LogNorm
        #normclass = Normalize
        cmap = 'viridis'

        Z = DATA[shapenum,2][1:-(missing_r_bins+1),1:-1]
        print(Z.shape)
        print(edges_phi.shape)
        print(edges_r.shape)
        ZC = np.concatenate([Z, Z, Z, Z], axis=1)

        fig, ax = plt.subplots(figsize=(20,20),
                               subplot_kw={'projection': '3d'})
        plt.title(titles[shapenum], fontsize=40)

        vmax = np.max((Z.ravel()))
        vmin = 0
        c = matplotlib.cm.viridis((Z.ravel())/vmax - vmin)

        dR = edges_r[1:] - edges_r[:-1]
        dPHI = edges_phi[1:] - edges_phi[:-1]

        dR, dPHI = np.meshgrid(dR, dPHI, indexing='ij')

        ax.bar3d(R.ravel(), PHI.ravel(), 0, 
                 dR.ravel(), dPHI.ravel(), Z.ravel(), 
                 color=c)

        plt.xlabel("r")
        plt.ylabel("phi")

        plt.show()

        fig, ax = plt.subplots(figsize=(20,20),
                               subplot_kw={'projection': 'polar'})
        plt.title(titles[shapenum], fontsize=40)
        pc = ax.pcolormesh(edges_phi, edges_r, Z,
                           cmap=cmap, norm=normclass())
        plt.show()

        fig, ax = plt.subplots(figsize=(20,20), 
                               subplot_kw={'projection': '3d'})

        plt.title(titles[shapenum], fontsize=40)

        pc = ax.plot_surface(X1, Y1, Z, cmap=cmap, norm=normclass())
        _  = ax.plot_surface(X2, Y2, Z, cmap=cmap, norm=normclass())
        _  = ax.plot_surface(X3, Y3, Z, cmap=cmap, norm=normclass())
        _  = ax.plot_surface(X4, Y4, Z, cmap=cmap, norm=normclass())

        #area = np.ones_like(DATA[0,0,1:-(missing_r_bins+1),1:-1])

        #pc = ax.pcolormesh(edges_phi, edges_r, 
        #                   DATA[shapenum, 2][1:-(missing_r_bins+1), 1:-1]/area,
        #                   cmap=cmap, norm=normclass())
        #pc2 = ax.pcolormesh(np.pi-edges_phi, edges_r, 
        #                    DATA[shapenum, 2][1:-(missing_r_bins+1), 1:-1]/area,
        #                    cmap=cmap, norm=normclass())
        #pc3 = ax.pcolormesh(np.pi+edges_phi, edges_r, 
        #                   DATA[shapenum, 2][1:-(missing_r_bins+1), 1:-1]/area,
        #                   cmap=cmap, norm=normclass())
        #pc4 = ax.pcolormesh(2*np.pi-edges_phi, edges_r, 
        #                    DATA[shapenum, 2][1:-(missing_r_bins+1), 1:-1]/area,
        #                    cmap=cmap, norm=normclass())
        fig.colorbar(pc)

        plt.show()




def do(res4, drbin):
    titles = [None, "Dipole", "Tee", None]


    SHAPE = res4.shape
    DATA = ak.to_numpy(res4)

    missing_r_bins = 0

    nbin_r = SHAPE[2]
    nbin_phi = SHAPE[3]

    edges_r = np.linspace(0, 1, nbin_r - 1)
    if(missing_r_bins>0):
        edges_r = edges_r[:-missing_r_bins]
    edges_phi = np.linspace(0, 0.5*np.pi, nbin_phi - 1)
    centers_r = 0.5*(edges_r[1:] + edges_r[:-1])
    centers_phi = 0.5*(edges_phi[1:] + edges_phi[:-1])
    print(SHAPE)
    print(edges_r)
    print(edges_phi)

    for shapenum in [1, 2]:
        fig, ax = plt.subplots(figsize=(20,20), 
                               subplot_kw={'projection': 'polar'})

        plt.title(titles[shapenum], fontsize=40)

        normclass = LogNorm
        #normclass = Normalize
        cmap = 'viridis'

        area = (edges_r[1:,None]**2 - edges_r[:-1,None]**2) * (edges_phi[None,1:] - edges_phi[None,:-1])/2
        #area = np.ones_like(DATA[0,0,1:-(missing_r_bins+1),1:-1])

        pc = ax.pcolormesh(edges_phi, edges_r, 
                           DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1]/area,
                           cmap=cmap, norm=normclass())
        pc2 = ax.pcolormesh(np.pi-edges_phi, edges_r, 
                            DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1]/area,
                            cmap=cmap, norm=normclass())
        pc3 = ax.pcolormesh(np.pi+edges_phi, edges_r, 
                           DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1]/area,
                           cmap=cmap, norm=normclass())
        pc4 = ax.pcolormesh(2*np.pi-edges_phi, edges_r, 
                            DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1]/area,
                            cmap=cmap, norm=normclass())
        fig.colorbar(pc)

        plt.show()

        angular_average = np.mean(DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1], axis=1, keepdims=True)
        ratio = DATA[shapenum, drbin][1:-(missing_r_bins+1), 1:-1]/angular_average
        fig, ax = plt.subplots(figsize=(20,20), 
                               subplot_kw={'projection': 'polar'})
        plt.title("%s ratio to angular average" % titles[shapenum], fontsize=40)
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
        plt.show()

        fig, ax = plt.subplots(figsize=(10,20))
        plt.title(titles[shapenum], fontsize=40)
        
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
        
        

        plt.legend()
        plt.show()
