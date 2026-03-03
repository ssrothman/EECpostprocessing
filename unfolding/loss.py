from typing import overload
from unfolding.histogram import Histogram
from unfolding.specs import DetectorModelProtocol
import numpy as np
import torch

class Loss:
    def __init__(self, 
                 reco : Histogram, 
                 genbaseline : np.ndarray | torch.Tensor,
                 model : DetectorModelProtocol,
                 negativePenalty : float = 0) -> None:
        self.reco = reco
        self.model = model
        self.negativePenalty = negativePenalty
        self.genbaseline = genbaseline

        if self.genbaseline.shape != (self.model.nGen,):
            raise ValueError(
                f"genbaseline must have shape ({self.model.nGen},), "
                f"but has shape {self.genbaseline.shape}"
            )

        if reco.device == 'numpy':
            if isinstance(genbaseline, torch.Tensor):
                self.genbaseline = genbaseline.numpy(force=True)
        elif reco.device != 'numpy':
            if not isinstance(genbaseline, torch.Tensor):
                self.genbaseline = torch.from_numpy(genbaseline)

            #assertion to make pyright happy 
            assert(isinstance(self.genbaseline, torch.Tensor))
            
            self.genbaseline = self.genbaseline.to(reco.device) 
        
    def to_numpy(self, *args, **kwargs) -> "Loss":
        self.reco.to_numpy(*args, **kwargs)
        self.model.to_numpy(*args, **kwargs)
        if isinstance(self.genbaseline, torch.Tensor):
            self.genbaseline = self.genbaseline.numpy(*args, **kwargs)

        return self

    def to_torch(self) -> "Loss":
        self.reco.to_torch()
        self.model.to_torch()
        if isinstance(self.genbaseline, np.ndarray):
            self.genbaseline = torch.from_numpy(self.genbaseline)
       
        self.genbaseline = self.genbaseline.to(self.reco.device)

        return self
    
    def to(self, device : str, *args, **kwargs) -> "Loss":
        self.reco.to(device, *args, **kwargs)
        self.model.to(device, *args, **kwargs)

        if isinstance(self.genbaseline, np.ndarray):
            if device == 'numpy':
                pass
            else: 
                self.genbaseline = torch.from_numpy(self.genbaseline)
                self.genbaseline = self.genbaseline.to(device, *args, **kwargs)
        elif isinstance(self.genbaseline, torch.Tensor):
            if device == 'numpy':
                self.genbaseline = self.genbaseline.numpy(*args, **kwargs)
            else:
                self.genbaseline = self.genbaseline.to(device, *args, **kwargs)
        
        return self
    
    def detach(self) -> "Loss":
        self.reco.detach()
        self.model.detach()
        if isinstance(self.genbaseline, torch.Tensor):
            self.genbaseline = self.genbaseline.detach()

        return self

    @overload
    def __call__(self, gen : np.ndarray,
                 theta : np.ndarray | None = None) -> np.float64:
        ...
    @overload
    def __call__(self, gen : torch.Tensor,
                 theta : torch.Tensor | None = None) -> torch.Tensor:
        ...

    def __call__(self, gen : np.ndarray | torch.Tensor,
                 theta : np.ndarray | torch.Tensor | None = None) -> np.float64 | torch.Tensor:
        '''
        If theta is None, gen is assumed to actually be
        a vector of the form (gen, theta) concatenated together.
        '''
        if theta is None:
            theta = gen[self.model.nGen:]
            gen = gen[:self.model.nGen]

        gen = gen * self.genbaseline

        if theta.shape != (self.model.nSyst,):
            raise ValueError(
                f"theta must have shape ({self.model.nSyst},), "
                f"but has shape {theta.shape}"
            )
        if gen.shape != (self.model.nGen,):
            raise ValueError(
                f"gen must have shape ({self.model.nGen},), "
                f"but has shape {gen.shape}"
            )

        if isinstance(gen, np.ndarray):
            '''
            The torch and numpy interfaces are similar enough
            that we can just branch here based on the type of gen.

            and then do all the computation in a unified way
            '''
            if not isinstance(theta, np.ndarray):
                raise TypeError("gen and theta must be of the same type")
            if self.reco.device != 'numpy':
                raise TypeError("reco must be in numpy mode when gen is a numpy array")
            if self.model.device != 'numpy':
                raise TypeError("model must be in numpy mode when gen is a numpy array")
            
            thepkg = np
        else:
            if not isinstance(gen, torch.Tensor):
                raise TypeError("gen must be either a numpy array or a torch tensor")
            if not isinstance(theta, torch.Tensor):                
                raise TypeError("gen and theta must be of the same type")
            if self.reco.device == 'numpy':
                raise TypeError("reco must be in torch mode when gen is a torch tensor")
            if self.model.device != self.reco.device:
                print("model device:", self.model.device)
                print("reco device:", self.reco.device)
                raise TypeError("model and reco must be on the same device when gen is a torch tensor")
            if str(gen.device) != self.model.device:
                print("gen device:", gen.device)
                print("model device:", self.model.device)
                raise TypeError("gen and model must be on the same device when gen is a torch tensor")
            if str(theta.device) != self.model.device:
                print("theta device:", theta.device)
                print("model device:", self.model.device)
                raise TypeError("theta and model must be on the same device when gen is a torch tensor")
            
            thepkg = torch

        #the checks above guarentee that everything is the right type
        #even though pyright can't figure it out

        pred = self.model.forward(gen, theta) # type: ignore
        err = pred - self.reco.values

        err_term = thepkg.linalg.multi_dot(
            (err, self.reco.invcov, err)
        )

        cstr_term = thepkg.sum(thepkg.square(theta)) # type: ignore

        result = err_term + cstr_term
    
        '''
        Apply penalty for negative values in the gen histogram
        to discourage unphysical solutions.

        Magnitude of the penalty is controlled by `self.negativePenalty`.

        The penalty is of the form:
            penalty = sum_{i | gen_i < 0} (-gen_i)
        so that the derivative with respect to gen_i is:
            d(penalty)/d(gen_i) = -1  if gen_i < 0
                                = 0   otherwise
        This encourages negative gen_i values to increase towards zero.
        '''
        if self.negativePenalty > 0:            
            negative_parts = thepkg.clip(gen, None, 0.0) # type: ignore
            penalty = -self.negativePenalty * thepkg.sum(negative_parts) # type: ignore
            result = result + penalty

        return result