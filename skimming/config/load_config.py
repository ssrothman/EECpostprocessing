import json
import os

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
            config.update(cfgpart)
            
    return config