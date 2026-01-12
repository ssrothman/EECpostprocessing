

import os
from unfolding.histogram import Histogram
from unfolding.specs import dsspec, detectormodelspec
from unfolding.detectormodel import DetectorModel

def setup_unfolding_workspace(
        where : str,
        datacfg : dsspec,
        wtsyst : str,
        objsyst : str,
        MCcfg : dsspec,
        modelcfg : detectormodelspec,
        what : str,
) -> None:
    
    reco = Histogram.from_dataset(
        datacfg,
        what + '_totalReco',
        wtsyst,
        objsyst
    )

    if datacfg['isMC']:
        gen = Histogram.from_dataset(
            datacfg,
            what + '_totalGen',
            wtsyst,
            objsyst
        )
    else:
        gen = None

    mcgen = Histogram.from_dataset(
        MCcfg,
        what + '_totalGen',
        wtsyst,
        objsyst
    )

    reco.dump_to_disk(os.path.join(where, 'reco'))        
    mcgen.dump_to_disk(os.path.join(where, 'mcgen'))
    if gen is not None:
        gen.dump_to_disk(os.path.join(where, 'gen'))


    dm = DetectorModel.from_dataset(MCcfg, modelcfg)
    dm.dump_to_disk(os.path.join(where, 'detectormodel'))