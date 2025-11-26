import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class EPD_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform",
        flow_shift=1.0,
        algorithm_type="noise_prediction",
        checkpoint=False,
        use_afs=True,
        **kwargs
    ):
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.checkpoint = checkpoint
        self.use_afs = use_afs
        
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        self.timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.K = K = 2
        self.r = nn.Parameter(torch.zeros(steps, K))
        self.tilde_sigma = nn.Parameter(torch.zeros(steps, K))
        self.s = nn.Parameter(torch.zeros(steps, K))
        self.l = nn.Parameter(torch.zeros(steps, K))

    def sample(self, x, model_fn, output_traj=False, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.timesteps.to(device=device, dtype=dtype)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        rhos = self.noise_schedule.marginal_rho(timesteps)
        
        y = x / alphas[0]
        trajs = [x,]
        for i in range(self.steps):
            if self.use_afs and i == 0:
                noise = y / ((1 + rhos[i]**2).sqrt())
            else:
                x_i = y * alphas[i]
                #noise = self.checkpoint_model_fn(x_i, t_i) if self.checkpoint else self.model_fn(x_i, t_i)
                noise = self.checkpoint_model_fn(x_i, timesteps[i]) if self.checkpoint else self.model_fn(x_i, timesteps[i])
                
            o = self.l[i, 0]
            y = y + (1 + o) * (rhos[i+1] - rhos[i]) * noise
            x = y * alphas[i+1]
            trajs.append(x)
        
        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs