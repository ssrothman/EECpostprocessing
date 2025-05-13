import hist
import awkward as ak
import numpy as np
from .util import squash
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os.path

class KinematicsBinner:
    def __init__(self, config, *args, **kwargs):
        self.config = config

    def eventLevel(self, readers, evtMask, jetMask, wtVars, outpath):
        thevals = {}

        if self.isMC:
            if readers.LHE is not None:
                thevals['genHT'] = readers.LHE.HT[evtMask]
            thevals['nTrueInt'] = ak.values_astype(readers.nTrueInt[evtMask], np.int32)

        if hasattr(readers, '_rho'):
            thevals['rho'] = readers.rho[evtMask]

        if hasattr(readers, 'METpt'):
            thevals['MET'] = readers.METpt[evtMask]

        thevals['numLooseB'] = ak.sum(readers.rRecoJet.jets.passLooseB[jetMask], axis=-1)[evtMask]
        thevals['numMediumB'] = ak.sum(readers.rRecoJet.jets.passMediumB[jetMask], axis=-1)[evtMask]
        thevals['numTightB'] = ak.sum(readers.rRecoJet.jets.passTightB[jetMask], axis=-1)[evtMask]

        thevals['numJets'] = ak.num(readers.rRecoJet.jets[jetMask], axis=-1)[evtMask]

        thevals['Zpt'] = readers.rMu.Zs.pt[evtMask]
        thevals['Zy'] = readers.rMu.Zs.rapidity[evtMask]
        thevals['Zmass'] = readers.rMu.Zs.mass[evtMask]

        thevals['leadMuPt'] = readers.rMu.muons[:,0].pt[evtMask]
        thevals['leadMuEta'] = readers.rMu.muons[:,0].eta[evtMask]
        thevals['leadMuCharge'] = readers.rMu.muons[:,0].charge[evtMask]
        
        #if np.min(thevals['leadMuPt']) < 0:
        #    print("Uhhhhhhh negative lead pT??")
        #    print(thevals['leadMuPt'])
        #    print("max:", np.max(readers.rMu.muons[:,0].pt[evtMask]))
        #    print("min:", np.min(readers.rMu.muons[:,0].pt[evtMask]))
        #    argmin = np.argmin(readers.rMu.muons[:,0].pt[evtMask])
        #    print("argmin:", argmin)
        #    print("The bad muon is:")
        #    print("\tpt:", readers.rMu.muons[:,0].pt[evtMask][argmin])
        #    print("\teta:", readers.rMu.muons[:,0].eta[evtMask][argmin])
        #    print("\tphi:", readers.rMu.muons[:,0].phi[evtMask][argmin])
        #    print("\tcharge:", readers.rMu.muons[:,0].charge[evtMask][argmin])
        #    print("\trawpt:", readers.rMu.muons[:,0].rawPt[evtMask][argmin])
        #    print('\tRoccoR:', readers.rMu.muons[:,0].RoccoR[evtMask][argmin])
        #    print()

        thevals['subMuPt'] = readers.rMu.muons[:,1].pt[evtMask]
        thevals['subMuEta'] = readers.rMu.muons[:,1].eta[evtMask]
        thevals['subMuCharge'] = readers.rMu.muons[:,1].charge[evtMask]

        for variation in wtVars:
            thewt = wtVars[variation]
            thevals[variation] = thewt[evtMask]

        for key in thevals.keys():
            thevals[key] = squash(thevals[key])

        table = pa.Table.from_pandas(pd.DataFrame(thevals),
                                     preserve_index=False)

        filekey = readers.rMu._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        os.makedirs(outpath, exist_ok=True)
        destination = os.path.join(outpath, filekey + '.parquet')
        pq.write_table(table, destination)

        return ak.sum(thevals['evtwt_nominal'])

    def jetLevel(self, readers, jetMask, wtVars, outpath):
        thevals = {}

        thevals['pt'] = readers.rRecoJet.jets.corrpt[jetMask]
        thevals['eta'] = readers.rRecoJet.jets.eta[jetMask]
        thevals['phi'] = readers.rRecoJet.jets.phi[jetMask]

        thevals['passLooseB'] = readers.rRecoJet.jets.passLooseB[jetMask]
        thevals['passMediumB'] = readers.rRecoJet.jets.passMediumB[jetMask]
        thevals['passTightB'] = readers.rRecoJet.jets.passTightB[jetMask]

        if self.isMC:
            pflav = readers.rRecoJet.jets.partonFlavour[jetMask]
            hflav = readers.rRecoJet.jets.hadronFlavour[jetMask]
            thevals['flav'] = ak.where((pflav == 21) & (hflav == 0), 21, hflav)

        thevals['nConstituents'] = readers.rRecoJet.jets.nConstituents[jetMask]
        thevals['nPassingParts'] = readers.rRecoJet.simonjets.nPart[jetMask]
        
        if hasattr(readers.rRecoJet.simonjets, 'CHSpt'):
            thevals['nCHS'] = readers.rRecoJet.simonjets.nCHS[jetMask]
            thevals['CHSpt'] = readers.rRecoJet.simonjets.CHSpt[jetMask]
            thevals['CHSeta'] = readers.rRecoJet.simonjets.CHSeta[jetMask]
            thevals['CHSphi'] = readers.rRecoJet.simonjets.CHSphi[jetMask]

        if self.isMC:
            thevals['matchPt'] = readers.rRecoJet.simonjets.jetMatchPt[jetMask]
            thevals['matchEta'] = readers.rRecoJet.simonjets.jetMatchEta[jetMask]
            thevals['matchPhi'] = readers.rRecoJet.simonjets.jetMatchPhi[jetMask]
            thevals['matched'] = readers.rRecoJet.simonjets.jetMatched[jetMask]
    
        for variation in wtVars:
            thewt = wtVars[variation]
            thewt_b, _ = ak.broadcast_arrays(thewt, thevals['pt'])
            thevals[variation] = thewt_b

        for key in thevals.keys():
            thevals[key] = squash(thevals[key])

        table = pa.Table.from_pandas(pd.DataFrame(thevals),
                                     preserve_index=False)
        filekey = readers.rRecoJet._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        os.makedirs(outpath, exist_ok=True)
        destination = os.path.join(outpath, filekey + '.parquet')
        pq.write_table(table, destination)

        return ak.sum(thevals['evtwt_nominal'])

    def CHSJetLevel(self, readers, evtMask, wtVars, outpath):
        if not hasattr(readers.rRecoJet.simonjets, 'CHSpt'):
            return 0

        thevals = {}

        thevals['pt'] = readers.rRecoJet.matchedCHSjets.pt[evtMask]
        thevals['eta'] = readers.rRecoJet.matchedCHSjets.eta[evtMask]
        thevals['phi'] = readers.rRecoJet.matchedCHSjets.phi[evtMask]
        
        thevals['passLooseB'] = readers.rRecoJet.matchedCHSjets.passLooseB[evtMask]
        thevals['passMediumB'] = readers.rRecoJet.matchedCHSjets.passMediumB[evtMask]
        thevals['passTightB'] = readers.rRecoJet.matchedCHSjets.passTightB[evtMask]
    
        if self.isMC:
            pflav = readers.rRecoJet.matchedCHSjets.partonFlavour[evtMask]
            hflav = readers.rRecoJet.matchedCHSjets.hadronFlavour[evtMask]
            thevals['flav'] = ak.where((pflav == 21) & (hflav == 0), 21, hflav)

        for variation in wtVars:
            thewt = wtVars[variation][evtMask]
            thewt_b, _ = ak.broadcast_arrays(thewt, thevals['pt'])
            thevals[variation] = thewt_b

        for key in thevals.keys():
            thevals[key] = squash(thevals[key])

        table = pa.Table.from_pandas(pd.DataFrame(thevals),
                                     preserve_index=False)

        filekey = readers.rRecoJet._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        os.makedirs(outpath, exist_ok=True)
        destination = os.path.join(outpath, filekey + '.parquet')
        pq.write_table(table, destination)

        return ak.sum(thevals['evtwt_nominal'])

    def particleLevel(self, readers, mask, wtVars, outpath):
        thevals = {}

        thevals['pt'] = readers.rRecoJet.parts.pt[mask]
        thevals['eta'] = readers.rRecoJet.parts.eta[mask]
        thevals['phi'] = readers.rRecoJet.parts.phi[mask]
        thevals['pdgid'] = readers.rRecoJet.parts.pdgid[mask]
        thevals['charge'] = readers.rRecoJet.parts.charge[mask]

        thevals['dxy'] = readers.rRecoJet.parts.dxy[mask]
        thevals['dz'] = readers.rRecoJet.parts.dz[mask]
        thevals['puppiWeight'] = readers.rRecoJet.parts.puppiWeight[mask]
        thevals['fromPV'] = readers.rRecoJet.parts.fromPV[mask]

        if self.isMC:
            thevals['matchPt'] = readers.rRecoJet.parts.matchPt[mask]
            thevals['matchEta'] = readers.rRecoJet.parts.matchEta[mask]
            thevals['matchPhi'] = readers.rRecoJet.parts.matchPhi[mask]
            thevals['matchCharge'] = readers.rRecoJet.parts.matchCharge[mask]
            thevals['nMatches'] = readers.rRecoJet.parts.nMatches[mask]
            thevals['matchTypes'] = readers.rRecoJet.parts.matchTypes[mask]

        thevals['jetPt'] = readers.rRecoJet.jets.corrpt[mask]
        thevals['jetEta'] = readers.rRecoJet.jets.eta[mask]
        thevals['jetPhi'] = readers.rRecoJet.jets.phi[mask]

        thevals['jetPt'], thevals['jetEta'], thevals['jetPhi'], _ = ak.broadcast_arrays(
            thevals['jetPt'], thevals['jetEta'], thevals['jetPhi'],
            thevals['pt']
        )

        for variation in wtVars:
            thewt = wtVars[variation]
            thewt_b, _ = ak.broadcast_arrays(thewt, thevals['pt'])
            thevals[variation] = thewt_b

        for key in thevals.keys():
            thevals[key] = squash(thevals[key])

        table = pa.Table.from_pandas(pd.DataFrame(thevals),
                                     preserve_index=False)
        filekey = readers.rRecoJet._x.behavior['__events_factory__']._partition_key
        filekey = filekey.replace('/','_')
        os.makedirs(outpath, exist_ok=True)
        destination = os.path.join(outpath, filekey + '.parquet')
        pq.write_table(table, destination)

        return ak.sum(thevals['evtwt_nominal'])

    def binAll(self, readers, mask, evtMask, wtVars, basepath):
        Hevt = self.eventLevel(
            readers, evtMask, mask, wtVars,
            os.path.join(basepath, 'event')
        )
        Hjet = self.jetLevel(
            readers, mask, wtVars,
            os.path.join(basepath, 'jet')
        )
        HjetCHS = self.CHSJetLevel(
            readers, evtMask, wtVars,
            os.path.join(basepath, 'jetCHS')
        )
        Hpart = self.particleLevel(
            readers, mask, wtVars,
            os.path.join(basepath, 'part')
        )

        return {
            'Hevt': Hevt,
            'Hjet': Hjet,
            'HjetCHS': HjetCHS,
            'Hpart': Hpart,
        }
