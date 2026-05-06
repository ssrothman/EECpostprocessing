

from typing import Protocol, TypedDict, List, overload

import torch
import numpy as np

from simonpy.AbitraryBinning import ArbitraryGenRecoBinning

class dsspec(TypedDict):
    location: str
    config_suite: str
    runtag: str
    dataset: str
    isMC: bool
    target_lumi : float
    isStack : bool
    statN : int
    statK : int
    what : str

class whichsystspec(TypedDict):
    wtsyst : str
    objsyst : str

class systspec(TypedDict):
    name : str
    isobjsyst : bool
    onesided : bool
    varytransfer : bool
    label : str | None

class detectormodelspec(TypedDict):
    systematics : List[systspec]
    dset : dsspec
    what : str

class histspec(TypedDict):
    dset : dsspec
    hist : whichsystspec

class unfoldingworkspacespec(TypedDict):
    data : histspec
    model : detectormodelspec

class DetectorModelProtocol(Protocol):
    @property
    def nSyst(self) -> int:
        ...

    @property
    def nGen(self) -> int:
        ...

    @property
    def nReco(self) -> int:
        ...

    @property
    def device(self) -> str:
        ...

    @property
    def binning(self) -> ArbitraryGenRecoBinning:
        ...

    @property
    def nuisance_names(self) -> list[str]:
        ...

    @overload
    def forward(self, beta : torch.Tensor, theta : torch.Tensor) -> torch.Tensor:
        ...
    @overload
    def forward(self, beta : np.ndarray, theta : np.ndarray) -> np.ndarray:
        ...

    @classmethod
    def from_disk(cls, path : str) -> "DetectorModelProtocol":
        ...

    def dump_to_disk(self, where : str):
        ...

    def to_torch(self) -> "DetectorModelProtocol":
        ...

    def to_numpy(self, *args, **kwargs) -> "DetectorModelProtocol":
        ...

    def to(self, device : str, *args, **kwargs) -> "DetectorModelProtocol":
        ...

    def detach(self) -> "DetectorModelProtocol":
        ...