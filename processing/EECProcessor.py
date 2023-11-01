import awkward as ak
import numpy as np
from coffea import processor

import reading.reader as reader

import selections.masks as masks
import selections.weights as weights

import binning.binEEC as binEEC
import binning.binEEC_binwt as binEEC_binwt

from Locker import Locker
import pickle

import os

def write_mmaps(Hdict, basepath):
    for name in Hdict.keys():
        os.makedirs(os.path.join(basepath, name), exist_ok=True)
        for key in Hdict[name]:
            fname = os.path.join(basepath, name, key+'.npy')
            newarr = Hdict[name][key]
            if os.path.exists(fname):
                the_mmap = np.memmap(fname, dtype=newarr.dtype, mode='r+',
                                     shape=newarr.shape)
                the_mmap[:] += newarr[:]
            else:
                the_mmap = np.memmap(fname, dtype=newarr.dtype, mode='w+',
                                     shape=newarr.shape)
                the_mmap[:] = newarr[:]

def recursive_merge(old, new):
    #operation is in place
    if type(old) is not dict:
        return new+old
    else:
        for key in old.keys():
            new[key] = recursive_merge(old[key], new[key])
    return new

def hist_to_numpy(hist):
    return hist.values(flow=True)

def recursive_hist_to_numpy(hist):
    if type(hist) is not dict:
        return hist_to_numpy(hist)
    else:
        for key in hist.keys():
            hist[key] = recursive_hist_to_numpy(hist[key])
    return hist

class EECProcessor(processor.ProcessorABC):
    def __init__(self, names, matchNames, nDR, binwt, ineff):
        self.names = names
        self.matchNames = matchNames
        self.nDR = nDR
        self.binwt = binwt
        self.ineff = ineff

    def postprocess(self, accumulator):
        pass
    
    def process_from_fname(self, fname):
        from coffea.nanoevents import NanoEventsFactory
        events = NanoEventsFactory.from_root(fname).events()
        return self.process(events)

    def locking_merge_on_disk(self, fname, destination):
        # a bit of a hack because I can't get things to work reasonably

        new = self.process_from_fname(fname)
        print("--"*20)
        print("CHECK DIFFERENCE")
        print()
        transreco = new['EEC']['Htrans'].project('ptReco', 'dRbinReco', 'EECwtReco')
        reco = new['EEC']['HrecoPure']
        print(np.max(transreco.values(flow=True) - reco.values(flow=True)))
        print()
        print('--'*20)
        new = recursive_hist_to_numpy(new)

        print("REQUESTING LOCK")
        with Locker():
            print("OBTAINED LOCK")
            write_mmaps(new, destination)
            print("RELEASED LOCK")

    def process(self, events):
        print("top of process")
        #setup inputs
        jets = reader.jetreader(events, 'selectedPatJetsAK4PFPuppi', "SimonJets")
        muons = reader.muonreader(events, "Muon")
        HLT = events.HLT

        #evtSel = masks.getEventSelection(muons, HLT)
        #jetSel = masks.getJetSelection(jets, muons, evtSel)

        #jetMask = jetSel.all(*jetSel.names)
        import numpy as np
        #jetMask = np.ones(len(HLT), dtype=bool)
        jetMask = np.abs(jets.simonjets.eta) < 2.0
        jetMask = jetMask & (jets.simonjets.pt > 100)

        #evtWeight = weights.getEventWeight(events)
        #weight = evtWeight.weight()
        weight = np.ones(len(HLT))

        #return outputs
        result = {}
        print("just before loop")
        
        binner = binEEC_binwt if self.binwt else binEEC
    
        for name, matchName in zip(self.names, self.matchNames):
            print("doing", name, matchName)
            result[name] = binner.doAll(
                    events, '%sTransfer'%name, 'Reco%s'%name, 'Gen%s'%name,
                    '%sParticles'%matchName, '%sGenParticles'%matchName,
                    self.nDR, weight, jetMask, includeInefficiency=self.ineff)

        return result
