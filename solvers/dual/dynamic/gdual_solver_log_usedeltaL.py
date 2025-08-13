import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from contextlib import nullcontext

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
        eps=1e-2,
        time_learning=True,
        use_corrector=True,
        use_deltaL_1 = True,
        use_deltaL_2 = False,
        train_mode=False
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
        self.use_deltaL_1 = use_deltaL_1
        self.use_deltaL_2 = use_deltaL_2
        self.train_mode = train_mode

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
        xn, xc, xp = xs                # x_{i+1}, x_i, x_{i-1}
        en, ec, ep = es                # e_{i+1}, e_i, e_{i-1}
        p = self.transform.unpack(params)
        
        # log y (x/e 측)
        log_yc_x = self.transform.log_y(log_alpha[i],   log_sigma[i],   p, side='x')
        log_yn_x = self.transform.log_y(log_alpha[i+1], log_sigma[i+1], p, side='x')
        log_yc_e = self.transform.log_y(log_alpha[i],   log_sigma[i],   p, side='e')
        log_yn_e = self.transform.log_y(log_alpha[i+1], log_sigma[i+1], p, side='e')

        yc_x = torch.exp(log_yc_x)
        yn_x = torch.exp(log_yn_x)
        yc_e = torch.exp(log_yc_e)
        yn_e = torch.exp(log_yn_e)

        deltay_x = yn_x - yc_x
        deltay_e = yn_e - yc_e

        # Δu
        delta_ux = self.transform.L(log_yn_x, yn_x, p, side='x') - self.transform.L(log_yc_x, yc_x, p, side='x')
        delta_ue = self.transform.L(log_yn_e, yn_e, p, side='e') - self.transform.L(log_yc_e, yc_e, p, side='e')
        first_and_second_x = self.transform.g(yc_x, p, side='x') * delta_ux + self.transform.O2(delta_ux, p, side='x')
        first_and_second_e = self.transform.g(yc_e, p, side='e') * delta_ue + self.transform.O2(delta_ue, p, side='e')

        if self.use_deltaL_1:
            coeff1_x = deltay_x
            coeff1_e = deltay_e
        else:
            coeff1_x = first_and_second_x
            coeff1_e = first_and_second_e

        if self.use_deltaL_2:
            coeff2_x = deltay_x
            coeff2_e = deltay_e
        else:
            coeff2_x = first_and_second_x
            coeff2_e = first_and_second_e    

        if order == 1:
            X = xc * coeff1_x
            E = ec * coeff1_e
        
        elif order == 2:
            X = xc * deltay_x
            E = ec * deltay_e
            
            if corrector:
                X = X + 0.5 * (xn - xc) * coeff2_x
                E = E + 0.5 * (en - ec) * coeff2_e
            else:
                log_yp_x = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='x')
                log_yp_e = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='e')
                yp_x = torch.exp(log_yp_x)
                yp_e = torch.exp(log_yp_e)
                delta_ux_p = self.transform.L(log_yc_x, yc_x, p, side='x') - self.transform.L(log_yp_x, yp_x, p, side='x')
                delta_ue_p = self.transform.L(log_yc_e, yc_e, p, side='e') - self.transform.L(log_yp_e, yp_e, p, side='e')
                r_u = delta_ux_p / delta_ux
                r_v = delta_ue_p / delta_ue
                X = X + 0.5 * (xc - xp)/r_u * coeff2_x
                E = E + 0.5 * (ec - ep)/r_v * coeff2_e
                
        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(
            i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, p)
        out = sample_coeff * sample + grad_coeff * (X + E)
        return out

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

        context = nullcontext() if self.train_mode else torch.no_grad()
        with context:
            xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0]) if self.train_mode else self.model_fn(x_pred, timesteps[0])
            params, hidden = self.param_extractor({'x':xc, 'e':ec, 't': timesteps[0:2], 'h': None})
            
            use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
            for i in tqdm(range(self.steps), disable=not use_tqdm):
                p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

                # Predictor
                x_pred = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params[:, 0], p_order, corrector=False
                )
                
                if i < self.steps - 1:
                    xn, en = self.checkpoint_model_fn(x_pred, timesteps[i+1]) if self.train_mode else self.model_fn(x_pred, timesteps[i+1]) 
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
                xp, ep = xc, ec
                xc, ec = xn, en

        return x_pred
