import json
import os

from typing import Sequence
from simonpy.dictmerge import merge_dict

def load_config(suite : str):
    topdir = os.path.dirname(__file__)

    suitepath = os.path.join(
        topdir, 'suites', '%s.json'%suite
    )
    with open(suitepath) as f:
        suitecfg = json.load(f)

    config = {
        'configsuite_name' : suitecfg['configsuite_name'],
        'configsuite_comments' : suitecfg['configsuite_comments'],
    }
    for cfgfile in suitecfg['configs']:
        thepath = os.path.join(
            topdir, cfgfile
        )
        with open(thepath) as f:
            cfgpart = json.load(f)
            config = merge_dict(
                config, cfgpart,
                allow_new_keys=True,
            )

    return config