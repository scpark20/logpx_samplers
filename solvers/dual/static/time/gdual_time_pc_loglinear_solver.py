import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ....solver import Solver

def lambertw_autograd(x: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    """
    Lambert W 함수의 뉴턴-랩슨 근사 (backprop 지원, GPU 호환)
    w * exp(w) = x 를 만족하는 w를 찾는다.
    """
    # 초기 추정값: log(x+1)  (x> -1/e 구간에서 안정적)
    w = torch.log1p(x).clamp(min=-10, max=10)  # 폭발 방지
    for _ in range(max_iter):
        ew = torch.exp(w)
        wew = w * ew
        num = wew - x
        den = ew * (w + 1) - (w + 2) * num / (2 * w + 2)
        w = w - num / den
    return w

class GDual_Time_PC_LogLinear_Solver(Solver):
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
        param_dim=(),
        O2_coeff=False,
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
        self.O2_coeff = O2_coeff

        # for gamma, tau_x, tau_e, kappa_x, kappa_e
        
        if len(param_dim) > 0:
            init_params = torch.zeros(steps, 2, 5, 1, *param_dim)
        else:
            init_params = torch.zeros(steps, 2, 5)
        self.params = nn.Parameter(init_params)

        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

    def learned_timesteps(self, device=None, dtype=None):
        """
        log_deltas (length = steps)  ->  timesteps (length = steps+1, strictly decreasing)
        anchors: t[0] = T, t[-1] = t_eps
        """
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype

        T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)

        # 1) 양수 간격 + 총합 고정: softmax로 비율을 만들고 전체 스팬에 맞춤
        w = F.softmax(self.log_deltas, dim=0)            # (S,)
        deltas = (T - t_eps) * w                         # (S,), sum(deltas) = T - t_eps

        # 2) 누적합으로 감소하는 시간축 복원
        c = torch.cumsum(deltas, dim=0)                  # (S,)
        ts = torch.cat([T[None], T - c], dim=0)          # (S+1,)
        return ts
        
    def u(self, alpha, sigma, gamma, tau):
        y_pos = alpha * sigma.pow(-gamma)      # gamma >= 0
        y_neg = alpha.pow(1 + gamma)           # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def v(self, alpha, sigma, gamma, tau):
        y_pos = sigma.pow(1 - gamma)           # gamma >= 0
        y_neg = sigma * alpha.pow(gamma)       # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def L(self, y, tau, eps=1e-8):
        return tau*torch.log(y) + (1-tau)*y

    def L_inv(self, x, tau, eps=1e-8):
        a = 1.0 - tau
        z = (a / tau) * torch.exp(x / tau)
        y = (tau / a) * lambertw_autograd(z).real
        return y.clamp_min(eps)
    
    def O_delta_square(self, delta, kappa, tau, L_inv_uc):
        if self.O2_coeff:
            coeff = 2/3 * (tau*L_inv_uc) / ((1-tau)*L_inv_uc + tau)**3
            return coeff * kappa * delta**2
        else:
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

    def get_next_sample(self, sample, xs, es, i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, order, eps=1e-8, corrector=False):
        xn, xc, xp = xs
        en, ec, ep = es
        delta_u, r_u, L_inv_uc, L_inv_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, L_inv_vc, L_inv_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)
        
        X = xc * (L_inv_un - L_inv_uc)
        E = ec * (L_inv_vn - L_inv_vc)
        
        if order == 2:
            if corrector:
                X += 0.5 * (xn - xc) * (delta_u/((1-tau_x)+tau_x/L_inv_uc) + self.O_delta_square(delta_u, kappa_x, tau_x, L_inv_uc))
                E += 0.5 * (en - ec) * (delta_v/((1-tau_e)+tau_e/L_inv_vc) + self.O_delta_square(delta_v, kappa_e, tau_e, L_inv_vc))
            else:
                X += 0.5 * (xc - xp) * (delta_u/((1-tau_x)+tau_x/L_inv_uc) + self.O_delta_square(delta_u, kappa_x, tau_x, L_inv_uc)) / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * (delta_v/((1-tau_e)+tau_e/L_inv_vc) + self.O_delta_square(delta_v, kappa_e, tau_e, L_inv_vc)) / r_v.clamp_min(eps) 
            
        pos_sample_coeff = (sigmas[i + 1] / sigmas[i]) ** gamma
        neg_sample_coeff = (alphas[i + 1] / alphas[i]) ** (-gamma)
        pos_grad_coeff   = sigmas[i + 1] ** gamma
        neg_grad_coeff   = alphas[i + 1] ** (-gamma)
        sample_coeff = torch.where(gamma >= 0, pos_sample_coeff, neg_sample_coeff)
        grad_coeff   = torch.where(gamma >= 0, pos_grad_coeff,   neg_grad_coeff)

        return sample_coeff * sample + grad_coeff * (X + E)

    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        # noise_schedule이 텐서를 받아들일 수 있어야 자동미분이 유지됩니다.
        alphas = self.noise_schedule.marginal_alpha(timesteps)          # 벡터화된 구현 권장
        sigmas = self.noise_schedule.marginal_std(timesteps)
        
        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i+1, self.steps - i, self.order) if self.lower_order_final else min(i+1, self.order)

            # Predictor
            gamma, tau_x, tau_e, kappa_x, kappa_e = self.params[i][0]
            tau_x, tau_e = torch.sigmoid(tau_x), torch.sigmoid(tau_e)
            x_pred = self.get_next_sample(x_corr, (xn, xc, xp), (en, ec, ep), i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, p, self.eps, corrector=False)

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])

            # Corrector
            gamma, tau_x, tau_e, kappa_x, kappa_e = self.params[i][1]
            tau_x, tau_e = torch.sigmoid(tau_x), torch.sigmoid(tau_e)
            x_corr = self.get_next_sample(x_corr, (xn, xc, xp), (en, ec, ep), i, alphas, sigmas, gamma, tau_x, tau_e, kappa_x, kappa_e, 2, self.eps, corrector=True)

            xp = xc; ep = ec; xc = xn; ec = en
            
        return x_pred