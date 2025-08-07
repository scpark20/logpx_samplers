import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class GDual_Box_Solver(Solver):
    def __init__(
        self,
        model_fn,
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
        super().__init__(model_fn, noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps

        # for gamma, tau_x, tau_e, kappa_x, kappa_e
        self.params = nn.Parameter(torch.zeros(steps, 5))
        
    def u(self, alpha, sigma, gamma, tau):
        return self.box(alpha*sigma**(-gamma), tau) if gamma >= 0 else self.box(alpha**(1+gamma), tau)

    def v(self, alpha, sigma, gamma, tau):
        return self.box(sigma**(1-gamma), tau) if gamma >= 0 else self.box(sigma*alpha**gamma, tau)

    def box(self, y, tau):
        return (y**tau-1) / tau if tau > 0 else torch.log(y)

    def box_inv(self, y, tau):
        return (1+tau*y)**(1/tau) if tau > 0 else torch.exp(y)
        
    def O_delta_square(self, delta, kappa):
       return kappa * delta**2    

    def compute_delta_and_ratio(self, fn, alphas, sigmas, i, gamma, tau, eps=1e-8):
        c, n, p = i, i + 1, i - 1
        val_c = fn(alphas[c], sigmas[c], gamma, tau)
        val_n = fn(alphas[n], sigmas[n], gamma, tau)
        delta_c = val_n - val_c
        box_inv_c = self.box_inv(val_c, tau)
        box_inv_n = self.box_inv(val_n, tau)

        if p >= 0:
            val_p = fn(alphas[p], sigmas[p], gamma, tau)
            delta_p = val_c - val_p
            ratio = delta_p / (delta_c + eps)
        else:
            ratio = None
        
        return delta_c, ratio, box_inv_c, box_inv_n

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, order, eps=1e-8):
        delta_u, r_u, box_inv_uc, box_inv_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, box_inv_vc, box_inv_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)
        
        X = xc * (box_inv_un - box_inv_uc)
        E = ec * (box_inv_vn - box_inv_vc)
        
        if r_u is not None and order == 2:
            X += 0.5 * (xc - xp) / (r_u + eps) * (box_inv_uc**(1-tau_x)*delta_u + self.O_delta_square(delta_u, kappa_x))
            E += 0.5 * (ec - ep) / (r_v + eps) * (box_inv_vc**(1-tau_e)*delta_v + self.O_delta_square(delta_v, kappa_e))
            
        sample_coeff, grad_coeff = (
            ((sigmas[i + 1] / sigmas[i]) ** gamma, sigmas[i + 1] ** gamma)
            if gamma >= 0
            else ((alphas[i + 1] / alphas[i]) ** (-gamma), alphas[i + 1] ** (-gamma))
        )

        return sample_coeff * sample + grad_coeff * (X + E)

    def sample(self, x, **kwargs):
        
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        timesteps = self.get_time_steps(skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=self.steps, device=device, shift=self.flow_shift)
        alphas = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
        sigmas = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

        hist = [self.checkpoint_model_fn(x, timesteps[0])]
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i+1, self.steps - i, self.order) if self.lower_order_final else min(i+1, self.order)
            gamma, tau_x, tau_e, kappa_x, kappa_e = self.params[i]
            xp, ep = hist[i - 1] if i > 0 else (None, None)
            
            x = self.get_next_sample(x, hist[i][0], xp, hist[i][1], ep, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, p, self.eps)

            if i < self.steps - 1:
                hist.append(self.checkpoint_model_fn(x, timesteps[i + 1]))

        return x