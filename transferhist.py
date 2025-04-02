import numpy as np
import hist
import awkward as ak

def transferhist_from_df(df):

    maxR_reco = np.max(df['R_reco'])
    maxR_gen = np.max(df['R_gen'])
    maxr_reco = np.max(df['r_reco'])
    maxr_gen = np.max(df['r_gen'])
    maxc_reco = np.max(df['c_reco'])
    maxc_gen = np.max(df['c_gen'])

    H = hist.Hist(
        hist.axis.Integer(0, maxR_reco+2,
                          name="R_reco", label = '$R_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(0, maxR_gen+2,
                          name="R_gen", label = '$R_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(0, maxr_reco+2,
                          name="r_reco", label = '$r_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(0, maxr_gen+2,
                          name="r_gen", label = '$r_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(0, maxc_reco+2,
                          name="c_reco", label = '$c_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Integer(0, maxc_gen+2,
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
