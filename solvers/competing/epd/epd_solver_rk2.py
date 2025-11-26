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

    def eval_model(self, y, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        x = y * alpha
        return self.checkpoint_model_fn(x, t) if self.checkpoint else self.model_fn(x, t)

    def get_x(self, y, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        return y * alpha

    def get_y(self, x, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        return x / alpha

    def sample(self, x, model_fn, output_traj=False, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.timesteps.to(device=device, dtype=dtype)
        rhos = self.noise_schedule.marginal_rho(timesteps)
        
        y = self.get_y(x, rhos[0])
        trajs = [x,]
        for i in range(self.steps):
            if self.use_afs and i == 0:
                noise = y / ((1 + rhos[i]**2).sqrt())
            else:
                noise = self.checkpoint_model_fn(x, timesteps[i]) if self.checkpoint else self.model_fn(x, timesteps[i])
        
            # Get y_k
            r = 0.5
            #tau = (rhos[i]**r) * (rhos[i+1]**(1-r))
            tau = rhos[i]*r + rhos[i+1]*r
            y_k = y + (tau - rhos[i])*noise
            x_k = self.get_x(y_k, tau)

            # Eval
            eval_t = self.noise_schedule.inverse_rho(tau)
            noise = self.checkpoint_model_fn(x_k, eval_t) if self.checkpoint else self.model_fn(x_k, eval_t)
            
            y = y + (rhos[i+1] - rhos[i]) * noise
            x = self.get_x(y, rhos[i+1])
            trajs.append(x)
        
        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs