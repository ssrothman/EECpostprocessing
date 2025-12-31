import json
import os

from typing import Sequence

def merge_dict(original : dict, update : dict, allow_new_keys : bool, replace_dict : Sequence[str] = []) -> dict:
    '''
    Recursively merges two dictionaries, overwriting values in the original dict with those from the update dict.
    
    :param original: Original dictionary to be updated
    :type original: dict
    :param update: Dictionary with updates to apply
    :type update: dict
    :param allow_new_keys: If False, raises an error if update contains keys not in original
    :type allow_new_keys: bool
    :param replace_dict: List of keys for which dict values should be replaced entirely rather than merged
    :type replace_dict: Sequence[str]
    :return: Merged dictionary
    :rtype: dict[Any, Any]
    '''
    result = original.copy()
    for key, value in update.items():
        if key in result and type(result[key]) == type(value):
            if isinstance(value, dict) and key not in replace_dict:
                result[key] = merge_dict(result[key], value, allow_new_keys, replace_dict)
            else:
                result[key] = value
        elif key in result:
            raise TypeError(f"Type mismatch for key %s: should be %s but got %s" % (key, type(result[key]), type(value)))
        elif allow_new_keys:
            result[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in original dictionary and new keys are not allowed.")
        
    return result

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