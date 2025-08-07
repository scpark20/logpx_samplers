import os
from tkinter import X
import torch
from tqdm import tqdm
import math
from ..solver import Solver

class GDual_LogDirect_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def u(self, alpha, sigma, gamma):
        # γ ≥ 0: ρ = α · σ^(–γ),   γ < 0: ρ = α^(1+γ)
        rho = alpha * sigma**(-gamma) if gamma >= 0 else alpha**(1 + gamma)
        # ρ ≥ 1 → log 변환, ρ < 1 → 그대로
        return (torch.log(rho), True) if rho >= 1 else (rho, False)

    def v(self, alpha, sigma, gamma):
        # γ ≥ 0: ρ = σ^(1–γ),   γ < 0: ρ = σ · α^γ
        rho = sigma**(1 - gamma) if gamma >= 0 else sigma * alpha**gamma
        # ρ ≥ 1 → log 변환, ρ < 1 → 그대로
        return (torch.log(rho), True) if rho >= 1 else (rho, False)

    def O(self, delta, tau):
       return delta + tau*delta**2    

    def get_next_sample(self, sample, xc, xp, ec, ep, i, alphas, sigmas, gamma, tau_x, tau_e, order):
        u_c, X_LOG = self.u(alphas[i], sigmas[i], gamma)
        u_n, _ = self.u(alphas[i+1], sigmas[i+1], gamma)
        delta_uc = u_n - u_c

        v_c, E_LOG = self.v(alphas[i], sigmas[i], gamma)
        v_n, _ = self.v(alphas[i+1], sigmas[i+1], gamma)
        delta_vc = v_n - v_c

        X = xc * (-torch.expm1(-delta_uc)) if X_LOG else xc * delta_uc
        E = ec * (-torch.expm1(-delta_vc)) if E_LOG else ec * delta_vc

        if order == 2:
            u_p, _ = self.u(alphas[i-1], sigmas[i-1], gamma)
            delta_up = u_c - u_p
            r = delta_up / delta_uc
            X = X + 0.5*(xc - xp)/r*self.O(delta_uc, tau_x)

            v_p, _ = self.v(alphas[i-1], sigmas[i-1], gamma)
            delta_vp = v_c - v_p
            r = delta_vp / delta_vc
            E = E + 0.5*(ec - ep)/r*self.O(delta_vc, tau_e)

        sample_coeff = (sigmas[i+1]/sigmas[i])**gamma if gamma >= 0 else (alphas[i+1]/alphas[i])**(-gamma)
        if gamma >= 0:
            X_coeff = alphas[i+1] if X_LOG else sigmas[i+1]**gamma
            E_coeff = sigmas[i+1] if E_LOG else sigmas[i+1]**gamma
        else:
            X_coeff = alphas[i+1] if X_LOG else alphas[i+1]**(-gamma)
            E_coeff = sigmas[i+1] if E_LOG else alphas[i+1]**(-gamma)
        
        return sample_coeff*sample + X_coeff*X + E_coeff*E

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