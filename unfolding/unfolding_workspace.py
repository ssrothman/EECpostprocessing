

import os
from unfolding.histogram import Histogram
from unfolding.specs import dsspec

def setup_unfolding_workspace(
        where : str,
        datacfg : dsspec,
        wtsyst : str,
        objsyst : str,
        MCcfg : dsspec,
        what : str,
) -> None:
    
    reco = Histogram.from_dataset(
        datacfg,
        what + '_reco',
        wtsyst,
        objsyst
    )

    if datacfg['isMC']:
        gen = Histogram.from_dataset(
            datacfg,
            what + '_gen',
            wtsyst,
            objsyst
        )
    else:
        gen = None

    mcgen = Histogram.from_dataset(
        MCcfg,
        what + '_gen',
        wtsyst,
        objsyst
    )

    reco.dump_to_disk(os.path.join(where, 'reco'))        
    mcgen.dump_to_disk(os.path.join(where, 'mcgen'))
    if gen is not None:
        gen.dump_to_disk(os.path.join(where, 'gen'))