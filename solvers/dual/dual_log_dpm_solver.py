import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class Dual_Log_DPM_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def u(self, alpha):
        return torch.log(alpha)

    def v(self, sigma):
        return torch.log(sigma)

    def get_grad(self, curr, prev, function, variables, i, order):
        uc, un = function(variables[i]), function(variables[i+1])
        delta_uc = un - uc

        X = curr
        if order > 1:
            up = function(variables[i-1])
            delta_up = uc - up
            r = delta_up / delta_uc
            X = X + (curr - prev) / (2*r)

        return X, delta_uc

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, order):
        X, delta_uc = self.get_grad(xc, xp, self.u, alphas, i, order)
        E, delta_vc = self.get_grad(ec, ep, self.v, sigmas, i, order)

        X_coeff = -alphas[i+1] * torch.expm1(-delta_uc)
        E_coeff = -sigmas[i+1] * torch.expm1(-delta_vc)

        next = sample + X_coeff*X + E_coeff*E
        return next

    def sample(self, x, steps, skip_type="time_uniform_flow", order=2, flow_shift=1.0, gamma=0.0, tau_x=0.0, tau_e=0.0, **kwargs):
        assert order <= 2
        lower_order_final = True

        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            alphas = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            sigmas = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

            hist = [self.model_fn(x, timesteps[0])]

            for i in tqdm(range(steps), disable=os.getenv("TQDM", "False")):
                p = min(i+1, steps - i, order) if lower_order_final else min(i+1, order)
                xp, ep = hist[i - 1] if i > 0 else (None, None)
                x = self.get_next_sample(x, hist[i][0], xp, hist[i][1], ep, i, alphas, sigmas, p)

                if i < steps - 1:
                    hist.append(self.model_fn(x, timesteps[i + 1]))

        return x