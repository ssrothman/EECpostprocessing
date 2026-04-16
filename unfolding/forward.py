from simonpy.stats_v2 import multivariate_gaussian_rvs
from unfolding.histogram import Histogram
from unfolding.specs import DetectorModelProtocol
import numpy as np
import torch
from tqdm import tqdm

def _ensure_device(array, device : str):
    if device == 'numpy':
        if isinstance(array, torch.Tensor):
            array = array.numpy(force=True)
        else:
            pass #already numpy
    else:
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        else:
            pass #already torch

        array.to(device)

    return array


def forward_values(gen : Histogram, nuisances : np.ndarray | torch.Tensor, model : DetectorModelProtocol):
    gen.to(model.device)
    nuisances = _ensure_device(nuisances, model.device)
    
    return model.forward(gen.values, nuisances) # type: ignore

def bootstrap_forward_cov(gen : Histogram, nuisances : np.ndarray | torch.Tensor, model : DetectorModelProtocol,
                          nboot : int, seed : int | None):
    gen.to(model.device)
    nuisances = _ensure_device(nuisances, model.device)

    vals0 = gen.values
    nuisances0 = nuisances

    # now draw Nboot samples
    '''
    Outline:
    2. Use the multivariate_rvs function to draw samples of the gen values
    3. Use a standard normal distribution to draw samples of the nuisances
    4. Forward each sample through the model to get a distribution of forward values
        (can this be batched efficiently?)
    5. Compute the covariance of the forward values across the samples
    6. Return the covariance matrix :)
    '''

    print("Bootstrapping forward covariance...")

    if seed is not None:
        np.random.seed(seed)

    # check that gen.L is reasonable
    test = gen.L @ gen.L.T
    print("gen.L @ gen.L.T:", test.sum())
    print("gen.cov:", gen.covmat.sum())

    vals_boot = multivariate_gaussian_rvs(vals0, gen.L, nboot) # shape (nboot, ngen)
    I = np.eye(nuisances0.shape[0])
    nuisances_boot = multivariate_gaussian_rvs(nuisances0, I, nboot) # shape (nboot, nnuis)

    print("Sampled gen values and nuisances")
    print(vals_boot.shape)
    print(nuisances_boot.shape)

    fwd_nominal = forward_values(gen, nuisances, model)

    fwd = np.empty((nboot, fwd_nominal.shape[0]), dtype=np.float32)
    for i in tqdm(range(nboot)):
        fwd[i, :] = model.forward(vals_boot[i], nuisances_boot[i])

    if isinstance(fwd, torch.Tensor):
        fwd = fwd.cpu().numpy()

    print("Ran batched forward??")
    print(fwd.shape)

    fwd_diff = fwd - fwd_nominal[None, :] # shape (nboot, nfwd)
    cov_fwd = fwd_diff.T @ fwd_diff / (nboot - 1) # shape (nfwd, nfwd)
    print(cov_fwd.shape)
    print("cov_fwd.sum():", cov_fwd.sum())
    return cov_fwd

def hessian_forward_cov(gen : Histogram, nuisances : np.ndarray | torch.Tensor, model : DetectorModelProtocol):
    gen.to(model.device)
    nuisances = _ensure_device(nuisances, model.device)

    '''
    Musings:
    
    Can we use torch autograd to compute the Jacobian of the forward function with respect to the gen values?
    If so, then we can compute the forward covariance as J @ cov_gen @ J^T, where J is the Jacobian and cov_gen is the covariance of the gen values.

    If this is practical, it would be potentially more efficient and more robust than the bootstrap approach
    '''
    raise NotImplementedError("Hessian-based forward covariance is not implemented yet (if ever)")
    return np.zeros((gen.values.shape[0], gen.values.shape[0]))

def forward_hist(gen : Histogram, nuisances : np.ndarray | torch.Tensor, model : DetectorModelProtocol,
                 nboot : int, seed : int) -> Histogram:
    
    print("Forward hist")
    covmat = bootstrap_forward_cov(gen, nuisances, model, nboot, seed)
    values = forward_values(gen, nuisances, model)

    return Histogram(values, covmat, gen.binning)
