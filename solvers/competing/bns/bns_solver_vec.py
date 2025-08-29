import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class BNS_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        algorithm_type="dual_prediction",
        checkpoint=False,
        k=0.5,
        **kwargs
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.checkpoint = checkpoint
        self.k = k
        
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

        self.a = nn.Parameter(torch.ones(steps,))
        self.b = nn.Parameter(torch.ones(steps, steps))
        
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        step_dt = timesteps[:-1] - timesteps[1:]
        dalphas = self.noise_schedule.dalpha(timesteps, k=self.k)
        dsigmas = self.noise_schedule.dsigma(timesteps, k=self.k)

        x0 = x
        vs = []
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            xc, ec = self.checkpoint_model_fn(x, timesteps[i]) if self.checkpoint else self.model_fn(x, timesteps[i])
            vc = (dalphas[i]*xc + dsigmas[i]*ec) * step_dt[i]
            vs.append(vc)

            x = x0 * self.a[i]
            for j in range(0, i+1):
                x = x + vs[j] * self.b[i, j]
            
        outputs = {'samples': x}
        return outputs