import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class Dual_Direct_DPM_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)

    def O(self, delta):
        return delta
        
    def get_grad(self, curr, prev, variables, i, order):
        uc, un = variables[i], variables[i+1]
        delta_uc = un - uc

        X = curr * delta_uc
        if order > 1:
            up = variables[i-1]
            delta_up = uc - up
            r = delta_up / delta_uc
            X = X + (curr - prev) / (2*r) * self.O(delta_uc)

        return X

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, order):
        X = self.get_grad(xc, xp, alphas, i, order)
        E = self.get_grad(ec, ep, sigmas, i, order)

        next = sample + X + E
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