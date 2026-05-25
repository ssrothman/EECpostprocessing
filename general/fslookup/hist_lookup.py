import os
from typing import List
from general.fslookup.skim_path import lookup_skim_path

def get_hist_path(
        location : str, 
        config_suite : str,  
        runtag : str, 
        dataset : str,
        objsyst : str,
        evtwt : str,
        table : str,
        cov : bool,
        statN : int,
        statK : int | List[int],
        fname_suffix : str | None = None
    ):

    fs, skimpath = lookup_skim_path(
        location,
        config_suite,
        runtag,
        dataset,
        objsyst,
        table
    )

    thepath = os.path.join(
        os.path.dirname(skimpath),
        '%s_BINNED' % table
    )

    if cov:
        thepath += '_covmat'

    thepath += '_%s' % evtwt

    if statN > 0:
        if isinstance(statK, int):
            statK = [statK]
        statK_str = '+'.join(str(k) for k in statK)
        thepath += '_%dstat%s' % (statN, statK_str)

    if fname_suffix is not None:
        thepath += '_%s' % fname_suffix

    return fs, thepath + '.npy'

def get_hist_bincfg_path(
        location, 
        config_suite, 
        runtag, 
        dataset,
        objsyst,
        table
    ):

    fs, skimpath = lookup_skim_path(
        location,
        config_suite,
        runtag,
        dataset,
        objsyst,
        table
    )

    thepath = os.path.join(
        os.path.dirname(skimpath),
        '%s_bincfg.json' % table
    )

    return fs, thepath