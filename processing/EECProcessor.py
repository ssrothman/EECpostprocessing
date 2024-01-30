import awkward as ak
import numpy as np
from coffea import processor

from reading.allreader import AllReaders

import selections.masks as masks
import selections.weights as weights

from Locker import Locker
import pickle

import os

from binning.binEEC import binAll

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
    def __init__(self, config, statsplit=False):
        self.config = config
        self.statsplit = statsplit

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
        #setup inputs
        readers = []
        for i in range(len(self.config.EECnames)):
            readers.append(AllReaders(events, self.config, i))

        evtSel = masks.getEventSelection(
                readers[0].rMu, readers[0].HLT, self.config)
        jetSel = masks.getJetSelection(
                readers[0].rRecoJet, readers[0].rMu, 
                evtSel, self.config.jetSelection)

        jetMask = jetSel.all(*jetSel.names)

        evtWeight = weights.getEventWeight(events)
        weight = evtWeight.weight()

        #return outputs
        result = {}
        if self.statsplit:
            for i in range(len(readers)):
                EECname = self.config.EECnames[i]
                result[EECname+"1"] = binAll(
                        readers[i], self.config.nDR,
                        jetMask & (events.event%2==0), weight)
                result[EECname+"2"] = binAll(
                        readers[i], self.config.nDR,
                        jetMask & (events.event%2==1), weight)
        else:
            for i in range(len(readers)):
                EECname = self.config.EECnames[i]
                result[EECname] = binAll(
                        readers[i], self.config.nDR,
                        jetMask, weight)

        return result
