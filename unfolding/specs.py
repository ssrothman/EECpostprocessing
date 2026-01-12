

from typing import TypedDict, List


class dsspec(TypedDict):
    location: str
    config_suite: str
    runtag: str
    dataset: str
    isMC: bool

class detectormodelspec(TypedDict):
    objsyst : List[str]
    wtsyst : List[str]