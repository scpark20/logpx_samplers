import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from contextlib import nullcontext
from collections import deque   # NEW

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
        train_mode=False
    ):
        assert order <= 3
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
        self.train_mode = train_mode

        # learned timesteps (weights over (T - t_eps))
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

    def get_next_sample(
        self,
        sample,
        x_hist, e_hist,        # deque들 (오른쪽이 현재 i)
        xn, en,                # t_{i+1} 모델 출력 (corrector에서 사용)
        i,
        log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params,
        order,
        corrector=False
    ):
        xc = x_hist[-1]
        xp = x_hist[-2] if len(x_hist) >= 2 else None
        xpp= x_hist[-3] if len(x_hist) >= 3 else None
        ec = e_hist[-1]
        ep = e_hist[-2] if len(e_hist) >= 2 else None
        epp= e_hist[-3] if len(e_hist) >= 3 else None

        p   = self.transform.unpack(params)
        eps = self.eps

        # y 및 Δy
        log_yc_x = self.transform.log_y(log_alpha[i],   log_sigma[i],   p, side='x')
        log_yn_x = self.transform.log_y(log_alpha[i+1], log_sigma[i+1], p, side='x')
        log_yc_e = self.transform.log_y(log_alpha[i],   log_sigma[i],   p, side='e')
        log_yn_e = self.transform.log_y(log_alpha[i+1], log_sigma[i+1], p, side='e')

        yc_x, yn_x = torch.exp(log_yc_x), torch.exp(log_yn_x)
        yc_e, yn_e = torch.exp(log_yc_e), torch.exp(log_yn_e)

        deltay_x = yn_x - yc_x
        deltay_e = yn_e - yc_e

        # u, v 격자 (Δu_i = u_{i+1}-u_i, Δu_{i-1} = u_i-u_{i-1})
        ui_x = self.transform.L(log_yc_x, yc_x, p, side='x')
        un_x = self.transform.L(log_yn_x, yn_x, p, side='x')
        ui_e = self.transform.L(log_yc_e, yc_e, p, side='e')
        un_e = self.transform.L(log_yn_e, yn_e, p, side='e')
        delta_ux = un_x - ui_x
        delta_ue = un_e - ui_e

        if order == 1:
            X = xc * deltay_x
            E = ec * deltay_e

        elif order == 2:
            X = xc * deltay_x
            E = ec * deltay_e
            if corrector:
                # Heun(2M): xn/en 필요
                assert (xn is not None) and (en is not None)
                X = X + 0.5 * (xn - xc) * deltay_x
                E = E + 0.5 * (en - ec) * deltay_e
            else:
                # 뒤쪽 2점 predictor
                assert xp is not None and ep is not None
                log_yp_x = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='x')
                yp_x     = torch.exp(log_yp_x)
                up_x     = self.transform.L(log_yp_x, yp_x, p, side='x')
                r_u = torch.clamp((ui_x - up_x) / (delta_ux + eps), min=-1e6, max=1e6)

                log_yp_e = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='e')
                yp_e     = torch.exp(log_yp_e)
                up_e     = self.transform.L(log_yp_e, yp_e, p, side='e')
                r_v = torch.clamp((ui_e - up_e) / (delta_ue + eps), min=-1e6, max=1e6)

                X = X + 0.5 * (xc - xp) / r_u * deltay_x
                E = E + 0.5 * (ec - ep) / r_v * deltay_e

        else:  # order == 3
            if not corrector:
                # ---------- 3차 predictor (좌 anchor: {u_i,u_{i-1},u_{i-2}}) ----------
                assert xp is not None and xpp is not None and ep is not None and epp is not None

                # x측
                log_yp_x  = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='x')
                yp_x      = torch.exp(log_yp_x)
                up_x      = self.transform.L(log_yp_x, yp_x, p, side='x')
                log_ypp_x = self.transform.log_y(log_alpha[i-2], log_sigma[i-2], p, side='x')
                ypp_x     = torch.exp(log_ypp_x)
                upp_x     = self.transform.L(log_ypp_x, ypp_x, p, side='x')

                Δu_i_1 = ui_x - up_x
                Δu_i_2 = up_x - upp_x
                ηx = torch.clamp(delta_ux / (Δu_i_1 + eps), min=-1e6, max=1e6)
                φx = torch.clamp(Δu_i_1 / (Δu_i_1 + Δu_i_2 + eps), 0., 1.)
                ψx = torch.clamp(1. - φx, eps, 1.)

                a_x = 0.5 * ηx * (1. + φx) + (ηx**2) * (φx / 3.)
                b_x = (φx**2 / ψx) * (0.5 * ηx + (ηx**2)/3.)

                # e측
                log_yp_e  = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='e')
                yp_e      = torch.exp(log_yp_e)
                up_e      = self.transform.L(log_yp_e, yp_e, p, side='e')
                log_ypp_e = self.transform.log_y(log_alpha[i-2], log_sigma[i-2], p, side='e')
                ypp_e     = torch.exp(log_ypp_e)
                upp_e     = self.transform.L(log_ypp_e, ypp_e, p, side='e')

                Δv_i_1 = ui_e - up_e
                Δv_i_2 = up_e - upp_e
                ηe = torch.clamp(delta_ue / (Δv_i_1 + eps), min=-1e6, max=1e6)
                φe = torch.clamp(Δv_i_1 / (Δv_i_1 + Δv_i_2 + eps), 0., 1.)
                ψe = torch.clamp(1. - φe, eps, 1.)

                a_e = 0.5 * ηe * (1. + φe) + (ηe**2) * (φe / 3.)
                b_e = (φe**2 / ψe) * (0.5 * ηe + (ηe**2)/3.)

                X = (xc + a_x * (xc - xp) - b_x * (xp - xpp)) * deltay_x
                E = (ec + a_e * (ec - ep) - b_e * (ep - epp)) * deltay_e

            else:
                # ---------- 3차 corrector (우 anchor: {u_{i+1},u_i,u_{i-1}}) ----------
                # xn/en, xp/ep 필요
                assert (xn is not None) and (xp is not None) and (en is not None) and (ep is not None)

                # x측 계수: s = Δu_i, h = Δu_{i-1}
                log_yp_x = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='x')
                yp_x     = torch.exp(log_yp_x)
                up_x     = self.transform.L(log_yp_x, yp_x, p, side='x')
                s_x = delta_ux
                h_x = torch.clamp(ui_x - up_x, min=-1e6, max=1e6)
                den_x = torch.clamp(s_x + h_x, min=eps)

                a_cx = - (2. * s_x + 3. * h_x) / (6. * den_x)
                b_cx =   (s_x * s_x)           / (6. * torch.clamp(h_x * den_x, min=eps))

                # e측 계수
                log_yp_e = self.transform.log_y(log_alpha[i-1], log_sigma[i-1], p, side='e')
                yp_e     = torch.exp(log_yp_e)
                up_e     = self.transform.L(log_yp_e, yp_e, p, side='e')
                s_e = delta_ue
                h_e = torch.clamp(ui_e - up_e, min=-1e6, max=1e6)
                den_e = torch.clamp(s_e + h_e, min=eps)

                a_ce = - (2. * s_e + 3. * h_e) / (6. * den_e)
                b_ce =   (s_e * s_e)           / (6. * torch.clamp(h_e * den_e, min=eps))

                # 브래킷 × Δy
                X = (xn + a_cx * (xn - xc) - b_cx * (xc - xp)) * deltay_x
                E = (en + a_ce * (en - ec) - b_ce * (ec - ep)) * deltay_e

        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(
            i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, p
        )
        return sample_coeff * sample + grad_coeff * (X + E)

    # ---------- sampling ----------
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)
        log_alpha, log_sigma = torch.log(alphas), torch.log(sigmas)
        log_alpha_ratio, log_sigma_ratio = log_alpha[1:] - log_alpha[:-1], log_sigma[1:] - log_sigma[:-1]

        x_pred = x_corr = x
        x_hist = deque(maxlen=3)   # NEW
        e_hist = deque(maxlen=3)   # NEW

        context = nullcontext() if self.train_mode else torch.no_grad()
        with context:
            xc, ec = ( self.checkpoint_model_fn(x_pred, timesteps[0])
                    if self.train_mode else self.model_fn(x_pred, timesteps[0]) )
            x_hist.append(xc)
            e_hist.append(ec)
            params, hidden = self.param_extractor({
                'x': xc, 'e': ec,
                't': timesteps[0:2],
                'log_alpha': log_alpha[0:2],
                'log_sigma': log_sigma[0:2],
                'h': None,
                'step': 0
            })

            use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
            for i in tqdm(range(self.steps), disable=not use_tqdm):
                p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

                # --- Predictor (xn/en 아직 없음) ---
                x_pred = self.get_next_sample(
                    x_corr, x_hist, e_hist, xn=None, en=None, i=i,
                    log_alpha=log_alpha, log_sigma=log_sigma,
                    log_alpha_ratio=log_alpha_ratio, log_sigma_ratio=log_sigma_ratio,
                    params=params[:, 0], order=p_order, corrector=False
                )

                if i < self.steps - 1:
                    # 다음 스텝 모델값
                    xn, en = ( self.checkpoint_model_fn(x_pred, timesteps[i+1])
                            if self.train_mode else self.model_fn(x_pred, timesteps[i+1]) )
                    params, hidden = self.param_extractor({
                            'x': xn, 'e': en,
                            't': timesteps[i+1:i+3],
                            'log_alpha': log_alpha[i+1:i+3],
                            'log_sigma': log_sigma[i+1:i+3],
                            'h': hidden,
                            'step': i+1
                        })
                else:
                    break

                # --- Corrector ---
                if self.use_corrector:
                    x_corr = self.get_next_sample(
                        x_corr, x_hist, e_hist, xn, en, i,
                        log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio,
                        params[:, 1], min(p_order+1, 3), corrector=True
                    )
                else:
                    x_corr = x_pred

                # 히스토리 업데이트 (가장 오른쪽이 현재 시점으로 유지)
                x_hist.append(xn)
                e_hist.append(en)

        return x_pred

