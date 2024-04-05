import numpy as np
import awkward as ak
import pandas as pd

class TrainingData:
    def __init__(self):
        pass

    def binAll(self, readers, jetMask, evtMask, wt):
        jets = readers.rRecoJet._x.Jet

        pt = jets.pt
        B = jets.btagDeepB
        CvL = jets.btagDeepCvL
        CvB = jets.btagDeepCvB
        hadronFlavour = jets.hadronFlavour
        eta = np.abs(jets.eta)

        pt = ak.to_numpy(ak.flatten(pt, axis=None))
        B = ak.to_numpy(ak.flatten(B, axis=None))
        CvL = ak.to_numpy(ak.flatten(CvL, axis=None))
        CvB = ak.to_numpy(ak.flatten(CvB, axis=None))
        hadronFlavour = ak.to_numpy(ak.flatten(hadronFlavour, axis=None))
        eta = ak.to_numpy(ak.flatten(eta, axis=None))

        selection = (B != -1) & (eta < 2.0)

        return pd.DataFrame({'pt'             :            pt[selection], 
                             'eta'            :           eta[selection],
                             'B'              :             B[selection], 
                             'CvL'            :           CvL[selection], 
                             'CvB'            :           CvB[selection], 
                             'hadronFlavour'  : hadronFlavour[selection]})
