import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class GDual_Hyper_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def u(self, alpha, sigma, gamma):
        return alpha * sigma**(-gamma) if gamma >= 0 else alpha**(1 + gamma)

    def v(self, alpha, sigma, gamma):
        return sigma**(1 - gamma) if gamma >= 0 else sigma * alpha**gamma

    def B(self, delta, tau):
       return delta + tau * delta**2    

    def sample(self, x, steps, skip_type="time_uniform_flow", order=2, flow_shift=1.0, gamma=0.0, tau_x=0.0, tau_e=0.0, **kwargs):
        lower_order_final = True

        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            alphas = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            sigmas = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

            hist = [self.model_fn(x, timesteps[0])]
            noise = x
            for i in tqdm(range(steps), disable=os.getenv("TQDM", "False")):
                xc, ec = hist[i]
                x = alphas[i+1]*xc + sigmas[i+1]*ec

                if i < steps - 1:
                    hist.append(self.model_fn(x, timesteps[i + 1]))

        return x