import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class GDual_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        transform,
        param_extractor,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        order=2,
        lower_order_final=True,
        eps=1e-8,
        time_learning=True,
        use_corrector=True,
        exact_first=False
    ):
        assert order <= 2
        super().__init__(noise_schedule, 'dual_prediction')

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps
        self.time_learning = time_learning
        self.use_corrector = use_corrector
        self.param_extractor = param_extractor
        self.transform = transform
        self.exact_first = exact_first

        # learned timesteps (weights over (T - t_eps))
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

    def get_next_sample(
        self,
        sample,
        xs, es, i,
        log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params,
        order, corrector=False
    ):
        xn, xc, xp = xs
        en, ec, ep = es

        # logy & y
        logy_uc = self.transform.logy_u(log_alpha[i+0], log_sigma[i+0], params)
        logy_un = self.transform.logy_u(log_alpha[i+1], log_sigma[i+1], params)
        logy_vc = self.transform.logy_v(log_alpha[i+0], log_sigma[i+0], params)
        logy_vn = self.transform.logy_v(log_alpha[i+1], log_sigma[i+1], params)

        y_uc = torch.exp(logy_uc)
        y_un = torch.exp(logy_un)
        y_vc = torch.exp(logy_vc)
        y_vn = torch.exp(logy_vn)
        
        # ---- 여기부터 델타를 L(y_n) - L(y_c)로 계산 ----
        val_uc = self.transform.L_from_logy(logy_uc, params, x_side=True)
        val_un = self.transform.L_from_logy(logy_un, params, x_side=True)
        val_vc = self.transform.L_from_logy(logy_vc, params, x_side=False)
        val_vn = self.transform.L_from_logy(logy_vn, params, x_side=False)
        delta_u = val_un - val_uc
        delta_v = val_vn - val_vc
        
        first_and_second_u = self.transform.g_from_y(y_uc, params)*delta_u + \
                            self.transform.O_delta_square(delta_u, params)
        first_and_second_v = self.transform.g_from_y(y_vc, params, x_side=False)*delta_v + \
                            self.transform.O_delta_square(delta_v, params, x_side=False)

        if order == 1:
            if self.exact_first:
                X   = xc * (y_un - y_uc)
                E   = ec * (y_vn - y_vc)
            else:
                X = xc * first_and_second_u
                E = ec * first_and_second_v

        elif order == 2:
            # base (exact Δy terms)
            X   = xc * (y_un - y_uc)
            E   = ec * (y_vn - y_vc)

            if corrector:
                X = X + (xn - xc)/2 * first_and_second_u
                E = E + (en - ec)/2 * first_and_second_v
            else:
                logy_up = self.transform.logy_u(log_alpha[i-1], log_sigma[i-1], params)
                logy_vp = self.transform.logy_v(log_alpha[i-1], log_sigma[i-1], params)
                val_up  = self.transform.L_from_logy(logy_up, params, x_side=True)
                val_vp  = self.transform.L_from_logy(logy_vp, params, x_side=False)
                r_u = (val_uc - val_up) / delta_u.clamp_min(self.eps)
                r_v = (val_vc - val_vp) / delta_v.clamp_min(self.eps)

                X = X + (xc - xp)/(2*r_u) * first_and_second_u
                E = E + (ec - ep)/(2*r_v) * first_and_second_v
            
        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params)

        return sample_coeff * sample + grad_coeff * (X + E)

    # ---------- sampling ----------
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        # noise schedule (벡터화)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)

        # 로그/비율 선계산
        log_alpha, log_sigma = torch.log(alphas), torch.log(sigmas)
        log_alpha_ratio, log_sigma_ratio = log_alpha[1:] - log_alpha[:-1], log_sigma[1:] - log_sigma[:-1]
        
        # 초기 상태
        x_pred = x_corr = x
        xn, en, xp, ep = None, None, None, None

        # 첫 모델 평가 + 파라미터
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])
        params, hidden = self.param_extractor({'x':xc, 'e':ec, 't': timesteps[0:2], 'h': None})

        use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")   # ❗
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # Predictor
            x_pred = self.get_next_sample(
                x_corr, (xn, xc, xp), (en, ec, ep), i,
                log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params[:, 0], p_order, corrector=False
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
                params, hidden = self.param_extractor({'x':xn, 'e':en, 't': timesteps[i+1:i+3], 'h': hidden})
            else:
                break

            # Corrector
            if self.use_corrector:
                x_corr = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params[:, 1], 2, corrector=True
                )
            else:
                x_corr = x_pred

            # shift
            xp, ep = xc, ec; xc, ec = xn, en

        return x_pred