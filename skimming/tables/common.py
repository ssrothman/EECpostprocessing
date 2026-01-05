from pdb import run
import awkward as ak
import numpy as np
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import Weights
from typing import Any
import pyarrow as pa

def add_common_vars(thevals : dict[str, Any], 
                    objs: AllObjects,
                    evtmask : ak.Array):
    if objs.isMC and hasattr(objs, 'LHE'):
        thevals['genHT'] = objs.LHE.HT[evtmask]
        thevals['genVpt'] = objs.LHE.Vpt[evtmask]

    thevals['Zpt'] = objs.Muons.Zs.pt[evtmask]
    thevals['Zmass'] = objs.Muons.Zs.mass[evtmask]
    thevals['Zy'] = objs.Muons.Zs.rapidity[evtmask]
    thevals['MET'] = objs.MET.pt[evtmask]
    thevals['numLooseB'] = ak.sum(objs.AK4Jets.jets.passLooseB[evtmask], axis=-1)
    thevals['numMediumB'] = ak.sum(objs.AK4Jets.jets.passMediumB[evtmask], axis=-1)
    thevals['numTightB'] = ak.sum(objs.AK4Jets.jets.passTightB[evtmask], axis=-1)

def add_weight_variations(thevals : dict[str, Any],
                          weights : Weights,
                          evtmask : ak.Array):
    thevals['wt_nominal'] = weights.weight()[evtmask] # pyright: ignore[reportOptionalSubscript]
    for variation in weights.variations:
        thevals[variation] = weights.weight(variation)[evtmask] # pyright: ignore[reportOptionalSubscript]

def broadcast_all(thevals : dict[str, ak.Array],
                  shape_ref : ak.Array):
    for key in thevals:
        thevals[key] = ak.broadcast_arrays(thevals[key], shape_ref)[0]

def to_pa_table(thevals : dict[str, ak.Array]) -> pa.Table:
    for key in thevals:
        thevals[key] = pa.array(ak.to_numpy(ak.flatten(thevals[key], axis=None))) # pyright: ignore[reportArgumentType]
    return pa.Table.from_pydict(thevals)

def add_event_id(thevals : dict[str, ak.Array],
                 event : ak.Array,
                 lumi : ak.Array,
                 run : ak.Array,
                 evtmask : ak.Array):
    
    e = ak.to_numpy(event[evtmask]).astype(np.uint64)
    l = ak.to_numpy(lumi[evtmask]).astype(np.uint64)
    r = ak.to_numpy(run[evtmask]).astype(np.uint64)

    id = ((r << 48) + (l << 32) + e).astype(np.uint64)

    thevals['event_id'] = id

