from typing import NotRequired, Sequence, TypedDict


class dset_spec(TypedDict):
    config : str
    runtag : str
    dataset : str
    objsyst : str
    wtsyst : str
    table : str
    location : str
    isstack : bool

class reweighting_spec(TypedDict):
    name : str
    
    #datasets
    num : dset_spec
    denom : dset_spec 

    # plot characteristics
    logx : bool
    logy : bool
    xlabel : str
    ylabel : str

    # variable to reweight w.r.t.
    variable : str
    bins : Sequence[float]
    variable_pcstr : str

    # [optional] second dimension with coarse bins
    second_variable : NotRequired[str]
    second_bins : NotRequired[Sequence[float]]
    second_variable_pcstr : NotRequired[str]

    # cut
    cut : Sequence[str]

    # spline smoothing values
    trial_smoothings : Sequence[float]
    final_smoothing : float