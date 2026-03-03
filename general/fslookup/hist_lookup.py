import os
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
        statK : int
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
        thepath += '_%dstat%d' % (statN, statK)

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