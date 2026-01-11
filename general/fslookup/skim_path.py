from fsspec_xrootd.xrootd import XRootDFileSystem
from fslookup.location_lookup import location_lookup
import os
from typing import Any

def lookup_skim_path(location : str,
                     configsuite : str,
                     runtag : str,
                     dataset : str,
                     objsyst : str,
                     table : str) -> tuple[Any, str]:

    fs, basepath = location_lookup(location)
    skimpath = os.path.join(
        basepath,
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )
    return fs, skimpath