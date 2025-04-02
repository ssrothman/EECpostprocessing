import numpy as np
import hist
import awkward as ak

def basichist_from_df(df, bootstrap):
    maxR = np.max(df['R'])
    maxr = np.max(df['r'])
    maxc = np.max(df['c'])

    minR = np.min(df['R'])
    minr = np.min(df['r'])
    minc = np.min(df['c'])

    H = hist.Hist(
        hist.axis.Integer(minR, maxR+1,
                          name="R", label = '$R$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr, maxr+1,
                          name="r", label = '$r$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc, maxc+1,
                          name="c", label = '$c$',
                          underflow=False, overflow=False),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt', label='$p_{T}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Integer(0, bootstrap+1,
                          name='bootstrap', label='bootstrap',
                          underflow=False, overflow=False),
        storage=hist.storage.Double()
    )

    row_per_evt = ak.run_lengths(df['eventhash'])
    
    rng = np.random.default_rng(0)
    boots = rng.poisson(1.0, (len(row_per_evt), bootstrap))
    boots = np.repeat(boots, row_per_evt, axis=0)

    H.fill(
        R=df['R'],
        r=df['r'],
        c=df['c'],
        pt=df['pt'],
        bootstrap = 0,
        weight=df['evtwt']*df['wt'],
    )

    for i in range(bootstrap):
        H.fill(
            R=df['R'],
            r=df['r'],
            c=df['c'],
            pt=df['pt'],
            bootstrap = i+1,
            weight=df['evtwt']*df['wt']*boots[:,i],
        )

    return H

def transferhist_from_df(df):
    maxR_reco = np.max(df['R_reco'])
    maxR_gen = np.max(df['R_gen'])
    maxr_reco = np.max(df['r_reco'])
    maxr_gen = np.max(df['r_gen'])
    maxc_reco = np.max(df['c_reco'])
    maxc_gen = np.max(df['c_gen'])

    minR_reco = np.min(df['R_reco'])
    minR_gen = np.min(df['R_gen'])
    minr_reco = np.min(df['r_reco'])
    minr_gen = np.min(df['r_gen'])
    minc_reco = np.min(df['c_reco'])
    minc_gen = np.min(df['c_gen'])

    H = hist.Hist(
        hist.axis.Integer(minR_reco, maxR_reco+1,
                          name="R_reco", label = '$R_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minR_gen, maxR_gen+1,
                          name="R_gen", label = '$R_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr_reco, maxr_reco+1,
                          name="r_reco", label = '$r_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minr_gen, maxr_gen+1,
                          name="r_gen", label = '$r_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc_reco, maxc_reco+1,
                          name="c_reco", label = '$c_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(minc_gen, maxc_gen+1,
                          name="c_gen", label = '$c_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt_reco', label='$p_{T,reco}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Variable([200, 400, 800, 1600],
                           name='pt_gen', label='$p_{T,gen}$ [GeV]',
                           underflow=True, overflow=True),
        storage=hist.storage.Double()
    )

    H.fill(
        R_reco=df['R_reco'],
        R_gen=df['R_gen'],
        r_reco=df['r_reco'],
        r_gen=df['r_gen'],
        c_reco=df['c_reco'],
        c_gen=df['c_gen'],
        pt_reco=df['pt_reco'],
        pt_gen=df['pt_gen'],
        weight=df['evtwt']*df['wt_reco'],
    )

    return H
