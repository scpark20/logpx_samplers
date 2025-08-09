import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

# tau_x, tau_e에 exp없애기

class GDual_PC_Box_Solver(Solver):
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
        param_dim=()
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

        # for gamma, tau_x, tau_e, kappa_x, kappa_e
        
        if len(param_dim) > 0:
            init_params = torch.zeros(steps, 2, 5, 1, *param_dim)
        else:
            init_params = torch.zeros(steps, 2, 5)
        # tau_x, tau_e    
        self.params = nn.Parameter(init_params)            
        
    def u(self, alpha, sigma, gamma, tau):
        y_pos = alpha * sigma.pow(-gamma)      # gamma >= 0
        y_neg = alpha.pow(1 + gamma)           # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.box(y, tau)

    def v(self, alpha, sigma, gamma, tau):
        y_pos = sigma.pow(1 - gamma)           # gamma >= 0
        y_neg = sigma * alpha.pow(gamma)       # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.box(y, tau)

    def box(self, y, tau, eps=1e-8):
        pow_branch = (y.pow(tau) - 1) / (tau + eps)
        log_branch = torch.log(y)
        return torch.where(tau > 0, pow_branch, log_branch)

    def box_inv(self, y, tau, eps=1e-8):
        exp_branch = (1 + tau * y).clamp_min(eps).pow(1 / (tau + eps))
        log_branch = torch.exp(y)
        return torch.where(tau > 0, exp_branch, log_branch)
        
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

    def get_next_sample(self, sample, xs, es, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, order, eps=1e-8, corrector=False):
        xn, xc, xp = xs
        en, ec, ep = es
        delta_u, r_u, box_inv_uc, box_inv_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, box_inv_vc, box_inv_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)
        
        X = xc * (box_inv_un - box_inv_uc)
        E = ec * (box_inv_vn - box_inv_vc)
        
        if order == 2:
            if corrector:
                X += 0.5 * (xn - xc) * (box_inv_uc**(1-tau_x)*delta_u + self.O_delta_square(delta_u, kappa_x))
                E += 0.5 * (en - ec) * (box_inv_vc**(1-tau_e)*delta_v + self.O_delta_square(delta_v, kappa_e))
            else:
                X += 0.5 * (xc - xp) / r_u.clamp_min(eps) * (box_inv_uc**(1-tau_x)*delta_u + self.O_delta_square(delta_u, kappa_x))
                E += 0.5 * (ec - ep) / r_v.clamp_min(eps) * (box_inv_vc**(1-tau_e)*delta_v + self.O_delta_square(delta_v, kappa_e))
            
            
        pos_sample_coeff = (sigmas[i + 1] / sigmas[i]) ** gamma
        neg_sample_coeff = (alphas[i + 1] / alphas[i]) ** (-gamma)
        pos_grad_coeff   = sigmas[i + 1] ** gamma
        neg_grad_coeff   = alphas[i + 1] ** (-gamma)
        sample_coeff = torch.where(gamma >= 0, pos_sample_coeff, neg_sample_coeff)
        grad_coeff   = torch.where(gamma >= 0, pos_grad_coeff,   neg_grad_coeff)

        return sample_coeff * sample + grad_coeff * (X + E)

    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        timesteps = self.get_time_steps(skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=self.steps, device=device, shift=self.flow_shift)
        alphas = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
        sigmas = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i+1, self.steps - i, self.order) if self.lower_order_final else min(i+1, self.order)

            # Predictor
            gamma, tau_x, tau_e, kappa_x, kappa_e = self.params[i][0]
            #tau_x, tau_e = torch.exp(tau_x), torch.exp(tau_e)
            x_pred = self.get_next_sample(x_corr, (xn, xc, xp), (en, ec, ep), i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, p, self.eps, corrector=False)

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])

            # Corrector
            gamma, tau_x, tau_e, kappa_x, kappa_e = self.params[i][1]
            #tau_x, tau_e = torch.exp(tau_x), torch.exp(tau_e)
            x_corr = self.get_next_sample(x_corr, (xn, xc, xp), (en, ec, ep), i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, 2, self.eps, corrector=True)

            xp = xc; ep = ec; xc = xn; ec = en
            
        return x_pred