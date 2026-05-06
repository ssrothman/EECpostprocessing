
import os

from typing import Mapping, Tuple
from unfolding.loss import Loss
import numpy as np
import torch
from time import time
import torchmin
import re
import json

class StatusCallback:
    def __init__(self, lossfunc, cpt_interval, cpt_start, cpt_path):
        self.lossfunc = lossfunc
        self.counter = 0
        self.cpt_interval = cpt_interval
        self.cpt_path = cpt_path
        self.cpt_start = cpt_start
        if cpt_path is not None:
            import os
            os.makedirs(cpt_path, exist_ok=True)

    def __call__(self, x):
        import os

        print("LOSS:", self.lossfunc(x).item())
        self.counter += 1
        if self.cpt_path is not None and self.counter % self.cpt_interval == 0:
            print("\tcheckpointing...")
            icpt = self.counter // self.cpt_interval
            icpt += self.cpt_start
            ipath = os.path.join(self.cpt_path, 'cpt_%03d.npy' % icpt)
            np.save(ipath, x.cpu().detach().numpy())

def res_to_npy(res): 
    for key in res.keys():
        if type(res[key]) is torch.Tensor:
            res[key] = res[key].cpu().detach().numpy()

def compute_hessian(loss : Loss, 
                    x : torch.Tensor) -> torch.Tensor:
    
    print("Computing Hessian...")
    H = torch.autograd.functional.hessian(loss, x, vectorize=False)
    return 0.5 * (H + H.T)

def save_result(res : Mapping, where : str) -> None:
    os.makedirs(where, exist_ok=True)
    res_to_npy(res)

    arrays = []
    dicts = []
    extradict = {}
    for key in res.keys():
        if isinstance(res[key], np.ndarray):
            arrays.append(key)
        elif isinstance(res[key], dict):
            dicts.append(key)
        else:
            extradict[key] = res[key]

    
    for key in arrays:
        with open(os.path.join(where, f"{key}.npy"), 'wb') as f:
            np.save(f, res[key])
    
    for key in dicts:
        with open(os.path.join(where, f"{key}.json"), 'w') as f:
            json.dump(res[key], f, indent=4)
    
    with open(os.path.join(where, "extradata.json"), 'w') as f:
        json.dump(extradict, f, indent=4)

class Minimizer:
    def __init__(self, cfg : dict) -> None:
        self.cfg = cfg

        if 'cpt_start' not in self.cfg:
            self.cfg['cpt_start'] = 0

    def save(self, where : str) -> None:
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, 'minimizer_cfg.json'), 'w') as f:
            json.dump(self.cfg, f, indent=4)

    @classmethod
    def continue_from(cls, where : str) -> Tuple["Minimizer", np.ndarray]:
        with open(os.path.join(where, 'minimizer_cfg.json'), 'r') as f:
            cfg = json.load(f)
        
        if os.path.exists(os.path.join(where, 'result')):
            print("Using `result` directory as the last checkpoint.")
            last_xval = np.load(
                os.path.join(where, 'result', 'x.npy')
            )
        else:
            last_xval = None

        if os.path.exists(os.path.join(where, 'checkpoints')):
            allfiles = os.listdir(os.path.join(where, 'checkpoints'))
            cpts = list(filter(
                lambda x: x.startswith('cpt_') and x.endswith('.npy'),
                allfiles
            ))
            if len(cpts) > 0:
                # use regex to extract checkpoint numbers
                cpt_nums = []
                for cpt in cpts:
                    m = re.match(r'cpt_(\d+)\.npy', cpt)
                    if m:
                        cpt_nums.append(int(m.group(1)))
                    else:
                        raise ValueError(f"Invalid checkpoint file name: {cpt}")

                last_cpt = np.max(cpt_nums)
                if last_xval is None:
                    last_xval = np.load(
                        os.path.join(where, 'checkpoints', f'cpt_{last_cpt:03d}.npy')
                    )
                    print("Continuing from checkpoint number:", last_cpt)
                else:
                    print("Starting checkpoints at number:", last_cpt + 1)
                    
                cfg['cpt_start'] = last_cpt + 1

        if last_xval is None:
            raise ValueError(
                "No checkpoints or result found to continue from"
            )

        return cls(cfg), last_xval

    def __call__(self, loss : Loss, 
                 x0 : np.ndarray | torch.Tensor | None,
                 device : str):

        cfgpath = os.path.join(self.cfg['logpath'], 'minimizer_cfg.json')
        if os.path.exists(cfgpath):
            with open(cfgpath, 'r') as f:
                oldcfg = json.load(f)
            
            del oldcfg['cpt_start']
            comparecfg = self.cfg.copy()
            del comparecfg['cpt_start']

            if oldcfg != comparecfg:
                print("Error: Minimizer configuration differs from saved configuration:")
                print("Saved configuration:")
                print(json.dumps(oldcfg, indent=4))
                print("Current configuration:")
                print(json.dumps(comparecfg, indent=4))
                raise ValueError("Minimizer configuration mismatch.")
        else:
            self.save(self.cfg['logpath'])
        
        #default initial guess is all ones
        if x0 is None:
            x0 = np.ones(loss.model.nGen)
        
        x0path = os.path.join(self.cfg['logpath'], 'x0.npy')
        if os.path.exists(x0path):
            pass
        else:
            if isinstance(x0, torch.Tensor):
                np.save(x0path, x0.detach().cpu().numpy())
            else:
                np.save(x0path, x0)

        baselinepath = os.path.join(self.cfg['logpath'], 'baseline.npy')
        if os.path.exists(baselinepath):
            pass
        else:
            if isinstance(loss.genbaseline, torch.Tensor):
                np.save(baselinepath, loss.genbaseline.detach().cpu().numpy())
            else:
                np.save(baselinepath, loss.genbaseline)

        binningpath = os.path.join(self.cfg['logpath'], 'binning.json')
        if os.path.exists(binningpath):
            pass
        else:
            loss.model.binning.genbinning.dump_to_file(binningpath)

        nuisance_names_path = os.path.join(self.cfg['logpath'], 'nuisance_names.txt')
        if os.path.exists(nuisance_names_path):
            pass
        else:
            with open(nuisance_names_path, 'w') as f:
                for name in loss.model.nuisance_names:
                    f.write(name + '\n')

        #ensure everything is `torch.Tensor`s
        loss = loss.to_torch()
        if isinstance(x0, np.ndarray):
            x0 = torch.from_numpy(x0)
            
        #if initial guess only has gen parameters, 
        # add zeros for syst parameters
        if len(x0) == loss.model.nGen and loss.model.nSyst > 0:
            t0 = torch.zeros(loss.model.nSyst, dtype = x0.dtype, device = x0.device)
            x0 = torch.cat([x0, t0])
        
        # move to correct device
        x0 = x0.to(device)
        loss = loss.to(device)

        cpt_path = os.path.join(self.cfg['logpath'], 'checkpoints')

        try:
            print("Running minimization with method:", self.cfg['method'])
            l0 = loss(x0)
            print("\tInitial loss:", l0.item())
            t0 = time()
            res = torchmin.minimize(
                loss,
                x0=x0,
                method=self.cfg['method'],
                callback = StatusCallback(
                    loss,
                    cpt_interval = self.cfg['cpt_interval'],
                    cpt_path = cpt_path,
                    cpt_start = self.cfg['cpt_start']   
                ),
                options = self.cfg['method_options']
            )
            print("Minimization completed.")
            print("\tt =", time()-t0)
            print("\tSuccess: ", res.success)
            print("\tStatus: ", res.status)
            print("\tMessage: ", res.message)
            print("\tL = %g"%res.fun.cpu().detach().item())

        except Exception as e:
            print(f"Minimization failed with error: {e}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            return 

        res.hessian = compute_hessian(loss, res.x)

        save_result(res, os.path.join(self.cfg['logpath'], 'result'))

        return res