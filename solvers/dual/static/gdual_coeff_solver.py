import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class GDual_Coeff_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        order=2,
        lower_order_final=True,
        eps=1e-8,
        algorithm_type="dual_prediction",    
    ):
        assert algorithm_type == 'dual_prediction'
        assert order <= 2
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps
        
        init_params = torch.zeros(steps, 3)
        self.params = nn.Parameter(init_params)            
        
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        timesteps = self.get_time_steps(skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=self.steps, device=device, shift=self.flow_shift)
        alphas = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
        sigmas = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

        data_pred, noise_pred = self.checkpoint_model_fn(x, timesteps[0])
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            c1, c2, c3 = self.params[i]
            x = c1*x + c2*data_pred + c3*noise_pred

            if i < self.steps - 1:
                data_pred, noise_pred = self.checkpoint_model_fn(x, timesteps[i + 1])
 
        return x