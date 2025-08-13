import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class GDual_LogAffine_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def u(self, alpha, sigma, gamma, tau):
        return self.L(alpha*sigma**(-gamma), tau) if gamma >= 0 else self.L(alpha**(1+gamma), tau)

    def v(self, alpha, sigma, gamma, tau):
        return self.L(sigma**(1-gamma), tau) if gamma >= 0 else self.L(sigma*alpha**gamma, tau)

    def L(self, y, tau):
        return torch.log(1+tau*y) / tau if tau > 0 else y

    def L_inv(self, y, tau):
        return (torch.exp(tau*y)-1) / tau if tau > 0 else y
        
    def O_delta_square(self, delta, kappa):
       return kappa * delta**2    

    def compute_delta_and_ratio(self, fn, alphas, sigmas, i, gamma, tau, eps=1e-8):
        c, n, p = i, i + 1, i - 1
        val_c = fn(alphas[c], sigmas[c], gamma, tau)
        val_n = fn(alphas[n], sigmas[n], gamma, tau)
        delta_c = val_n - val_c
        L_inv_c = self.L_inv(val_c, tau)
        L_inv_n = self.L_inv(val_n, tau)

        if p >= 0:
            val_p = fn(alphas[p], sigmas[p], gamma, tau)
            delta_p = val_c - val_p
            ratio = delta_p / (delta_c + eps)
        else:
            ratio = None
        
        return delta_c, ratio, L_inv_c, L_inv_n

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, order, eps=1e-8):
        delta_u, r_u, L_inv_uc, L_inv_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, L_inv_vc, L_inv_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)
        
        X = xc * (L_inv_un - L_inv_uc)
        E = ec * (L_inv_vn - L_inv_vc)
        
        if r_u is not None and order == 2:
            X += 0.5 * (xc - xp) / (r_u + eps) * (L_inv_uc**(1-tau_x)*delta_u + self.O_delta_square(delta_u, kappa_x))
            E += 0.5 * (ec - ep) / (r_v + eps) * (L_inv_vc**(1-tau_e)*delta_v + self.O_delta_square(delta_v, kappa_e))
            
        sample_coeff, grad_coeff = (
            ((sigmas[i + 1] / sigmas[i]) ** gamma, sigmas[i + 1] ** gamma)
            if gamma >= 0
            else ((alphas[i + 1] / alphas[i]) ** (-gamma), alphas[i + 1] ** (-gamma))
        )

        return sample_coeff * sample + grad_coeff * (X + E)

    def sample(self, x, steps, skip_type="time_uniform_flow", order=2, flow_shift=1.0, gamma=0.0, tau_x=0.0, tau_e=0.0, kappa_x=0.0, kappa_e=0.0, eps=1e-8, **kwargs):
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
                x = self.get_next_sample(x, hist[i][0], xp, hist[i][1], ep, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, p, eps)

                if i < steps - 1:
                    hist.append(self.model_fn(x, timesteps[i + 1]))

        return x