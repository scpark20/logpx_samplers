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

        # debug switch (default: ON)
        self.debug_nans = os.getenv("DPM_DEBUG_NAN", "1") not in ("0", "False", "false", "")

        # learned timesteps (weights over (T - t_eps))
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

        self.big_val_thresh  = 1e4
        self.big_val_ratio   = 1e-3
        self.raise_on_big    = True


    def _nanlog(self, step, name, tensor):
        
        if not self.debug_nans: return
        if not isinstance(tensor, torch.Tensor): return

        t = tensor.detach()
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()

        finite_mask = torch.isfinite(t)
        n_fin = finite_mask.sum().item()

        fmin = fmax = fmean = maxabs = float('nan')
        n_big = 0
        ratio_big = 0.0

        if n_fin > 0:
            tf = t[finite_mask]
            abs_tf = tf.abs()
            fmin = tf.min().item()
            fmax = tf.max().item()
            fmean = tf.mean().item()
            maxabs = abs_tf.max().item()
            n_big = (abs_tf > self.big_val_thresh).sum().item()
            ratio_big = n_big / max(1, n_fin)

        # 출력 트리거: NaN/Inf가 있거나, 큰 값 비율이 기준 이상이거나, 최대절대값이 임계를 크게 넘을 때
        trigger = (n_nan > 0) or (n_inf > 0) or (n_big > 0 and (ratio_big >= self.big_val_ratio or maxabs > self.big_val_thresh*10))

        if trigger:
            print(
                f"[NaNCheck][step {step}] {name}: "
                f"NaN={n_nan}, Inf={n_inf}, "
                f"|x|>{self.big_val_thresh:g}: {n_big} ({ratio_big:.4%}), "
                f"max|x|={maxabs:.3e}, "
                f"finite[min={fmin:.3e}, max={fmax:.3e}, mean={fmean:.3e}], "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}",
                flush=True
            )
            # if self.raise_on_big and (n_nan or n_inf or n_big):
            #     raise RuntimeError(f"Abnormal values in {name} at step {step}")


    def get_next_sample(
        self,
        sample,
        xs, es, i,
        log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params,
        order, corrector=False
    ):
        xn, xc, xp = xs
        en, ec, ep = es
        eps = self.eps
        gamma, tau_x, tau_e, kappa_x, kappa_e = self.transform.unpack(params, eps)

        self._nanlog(i, "params_in", params)
        self._nanlog(i, "log_alpha", log_alpha)
        self._nanlog(i, "log_sigma", log_sigma)
        
        # logy & y
        logy_uc = self.transform.logy_u(log_alpha[i+0], log_sigma[i+0], gamma)
        logy_un = self.transform.logy_u(log_alpha[i+1], log_sigma[i+1], gamma)
        logy_vc = self.transform.logy_v(log_alpha[i+0], log_sigma[i+0], gamma)
        logy_vn = self.transform.logy_v(log_alpha[i+1], log_sigma[i+1], gamma)
        self._nanlog(i, "logy_uc", logy_uc)
        self._nanlog(i, "logy_un", logy_un)
        self._nanlog(i, "logy_vc", logy_vc)
        self._nanlog(i, "logy_vn", logy_vn)

        y_uc = torch.exp(logy_uc)
        y_un = torch.exp(logy_un)
        y_vc = torch.exp(logy_vc)
        y_vn = torch.exp(logy_vn)
        self._nanlog(i, "y_uc", y_uc)
        self._nanlog(i, "y_un", y_un)
        self._nanlog(i, "y_vc", y_vc)
        self._nanlog(i, "y_vn", y_vn)
        
        # deltas via L(y_n) - L(y_c)
        val_uc = self.transform.L_from_logy(logy_uc, tau_x)
        val_un = self.transform.L_from_logy(logy_un, tau_x)
        val_vc = self.transform.L_from_logy(logy_vc, tau_e)
        val_vn = self.transform.L_from_logy(logy_vn, tau_e)
        self._nanlog(i, "val_uc", val_uc)
        self._nanlog(i, "val_un", val_un)
        self._nanlog(i, "val_vc", val_vc)
        self._nanlog(i, "val_vn", val_vn)

        delta_u = val_un - val_uc
        delta_v = val_vn - val_vc
        self._nanlog(i, "delta_u", delta_u)
        self._nanlog(i, "delta_v", delta_v)
        
        first_and_second_u = self.transform.g_from_y(y_uc, tau_x)*delta_u + \
                             self.transform.O_delta_square(delta_u, kappa_x)
        first_and_second_v = self.transform.g_from_y(y_vc, tau_e)*delta_v + \
                             self.transform.O_delta_square(delta_v, kappa_e)
        
        if order == 1:
            if self.exact_first:
                X = xc * (y_un - y_uc)
                E = ec * (y_vn - y_vc)
            else:
                X = xc * first_and_second_u
                E = ec * first_and_second_v
            self._nanlog(i, "X_o1", X)
            self._nanlog(i, "E_o1", E)

        elif order == 2:
            X = xc * (y_un - y_uc)
            E = ec * (y_vn - y_vc)
            self._nanlog(i, "X_base_o2", X)
            self._nanlog(i, "E_base_o2", E)

            if corrector:
                X = X + 0.5 * (xn - xc) * first_and_second_u
                E = E + 0.5 * (en - ec) * first_and_second_v
                self._nanlog(i, "X_corr", X)
                self._nanlog(i, "E_corr", E)
            else:
                logy_up = self.transform.logy_u(log_alpha[i-1], log_sigma[i-1], gamma)
                logy_vp = self.transform.logy_v(log_alpha[i-1], log_sigma[i-1], gamma)
                self._nanlog(i, "logy_up", logy_up)
                self._nanlog(i, "logy_vp", logy_vp)

                val_up = self.transform.L_from_logy(logy_up, tau_x)
                val_vp = self.transform.L_from_logy(logy_vp, tau_e)
                self._nanlog(i, "val_up", val_up)
                self._nanlog(i, "val_vp", val_vp)

                def clamp(value):
                    clamped = torch.where(value>=0, value.clamp_min(eps), value.clamp_max(-eps))
                    value = value + (clamped - value).detach()
                    return value
                
                r_u = (val_uc - val_up) / clamp(delta_u)
                r_v = (val_vc - val_vp) / clamp(delta_v)
                self._nanlog(i, "r_u", r_u)
                self._nanlog(i, "r_v", r_v)
                
                X = X + 0.5*(xc - xp)/clamp(r_u) * first_and_second_u
                E = E + 0.5*(ec - ep)/clamp(r_v) * first_and_second_v
                self._nanlog(i, "X_predcorr", X)
                self._nanlog(i, "E_predcorr", E)
            
        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(
            i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, gamma
        )
        self._nanlog(i, "sample_coeff", sample_coeff)
        self._nanlog(i, "grad_coeff", grad_coeff)

        #gamma, _, _, _, _ = self.transform._unpack(params)
        #cond_near1 = (gamma - 1.0).abs() < eps      # γ ≈ 1
        #cond_near0 = gamma.abs() < eps              # γ ≈ 0

        # cond는 bool 텐서라 & 로 결합해야 함(괄호 필수)
        #grad = torch.where(cond_near1, X, torch.where(cond_near0, E, X + E))
        grad = X + E
        out = sample_coeff * sample + grad_coeff * grad
        self._nanlog(i, "out_step", out)
        return out

    # ---------- sampling ----------
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        self._nanlog(-1, "timesteps", timesteps)
        if not self.time_learning:
            timesteps = timesteps.detach()

        # noise schedule (벡터화)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)
        self._nanlog(-1, "alphas", alphas)
        self._nanlog(-1, "sigmas", sigmas)

        # 로그/비율 선계산
        log_alpha, log_sigma = torch.log(alphas), torch.log(sigmas)
        log_alpha_ratio, log_sigma_ratio = log_alpha[1:] - log_alpha[:-1], log_sigma[1:] - log_sigma[:-1]
        self._nanlog(-1, "log_alpha", log_alpha)
        self._nanlog(-1, "log_sigma", log_sigma)
        self._nanlog(-1, "log_alpha_ratio", log_alpha_ratio)
        self._nanlog(-1, "log_sigma_ratio", log_sigma_ratio)
        
        # 초기 상태
        x_pred = x_corr = x
        xn, en, xp, ep = None, None, None, None

        # 첫 모델 평가 + 파라미터
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])
        self._nanlog(0, "xc_init", xc)
        self._nanlog(0, "ec_init", ec)

        params, hidden = self.param_extractor({'x':xc, 'e':ec, 't': timesteps[0:2], 'h': None})
        self._nanlog(0, "params_init", params)

        use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # Predictor
            x_pred = self.get_next_sample(
                x_corr, (xn, xc, xp), (en, ec, ep), i,
                log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params[:, 0], p_order, corrector=False
            )
            self._nanlog(i, "x_pred", x_pred)

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
                self._nanlog(i, "xn", xn)
                self._nanlog(i, "en", en)
                params, hidden = self.param_extractor({'x':xn, 'e':en, 't': timesteps[i+1:i+3], 'h': hidden})
                self._nanlog(i, "params_next", params)
            else:
                break

            # Corrector
            if self.use_corrector:
                x_corr = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params[:, 1], 2, corrector=True
                )
                self._nanlog(i, "x_corr", x_corr)
            else:
                x_corr = x_pred

            # shift
            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
