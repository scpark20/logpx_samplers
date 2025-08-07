import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class GDual_Solver(Solver):
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

    def compute_delta_and_ratio(self, fn, alphas, sigmas, i, gamma):
        c, n, p = i, i + 1, i - 1
        val_c = fn(alphas[c], sigmas[c], gamma)
        val_n = fn(alphas[n], sigmas[n], gamma)
        delta_c = val_n - val_c

        if p >= 0:
            val_p = fn(alphas[p], sigmas[p], gamma)
            delta_p = val_c - val_p
            ratio = delta_p / (delta_c + 1e-8)
        else:
            ratio = None
        
        return delta_c, ratio

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, gamma, tau_x, tau_e, order):
        delta_u, r_u = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma)
        delta_v, r_v = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma)
        
        X = xc * delta_u
        E = ec * delta_v
        
        if r_u is not None and order == 2:
            X += (xc - xp) / (2 * r_u + 1e-8) * self.B(delta_u, tau_x)
            E += (ec - ep) / (2 * r_v + 1e-8) * self.B(delta_v, tau_e)
            
        sample_coeff, grad_coeff = (
            ((sigmas[i + 1] / sigmas[i]) ** gamma, sigmas[i + 1] ** gamma)
            if gamma >= 0
            else ((alphas[i + 1] / alphas[i]) ** (-gamma), alphas[i + 1] ** (-gamma))
        )

        return sample_coeff * sample + grad_coeff * (X + E)

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
                x = self.get_next_sample(x, hist[i][0], xp, hist[i][1], ep, i, alphas, sigmas, gamma, tau_x, tau_e, p)

                if i < steps - 1:
                    hist.append(self.model_fn(x, timesteps[i + 1]))

        return x