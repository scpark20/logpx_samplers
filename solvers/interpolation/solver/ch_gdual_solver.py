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
        pred_order=2,
        corr_order=2, # 제대로 만듬
        lower_order_final=True,
        eps=1e-2,
        time_learning=True,
        use_corrector=True,
        train_mode=False
    ):
        super().__init__(noise_schedule, 'dual_prediction')

        self.steps = steps
        self.skip_type = skip_type
        self.pred_order = pred_order
        self.corr_order = corr_order
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

    @staticmethod
    def _apply_coeffs_to_window(coeffs, win):
        """
        coeffs: (B, C, M)
        win   : (B, M, C, H, W)
        return: (B, C, H, W)
        """
        w = coeffs.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B,M,C,1,1)
        return (w * win).sum(dim=1)

    def get_next_sample(self, i, sample, xs, es, alphas, sigmas, ar, sr, params, order, corrector=False):
        # params: (B, 5, C)  → unpack: dict of (B,C)
        p = self.transform.unpack(params)

        j = 2 if corrector else 1
        a_win = torch.flip(alphas[i - order + j:i + 2], dims=[0])   # (Mwin,)
        s_win = torch.flip(sigmas[i - order + j:i + 2], dims=[0])   # (Mwin,)

        # y, L: (B, C, Mwin)
        yx = self.transform.y(a_win, s_win, p, side='x')
        ye = self.transform.y(a_win, s_win, p, side='e')
        ux = self.transform.L(yx, p, side='x')  # (B,C,Mwin)
        ue = self.transform.L(ye, p, side='e')  # (B,C,Mwin)

        # predictor: us = ux[...,1:], corrector: us = ux (둘 다 (B,C,M))
        uc_x = ux[..., 1]                      # (B,C)
        un_x = ux[..., 0]                      # (B,C)
        us_x = ux if corrector else ux[..., 1:]  # (B,C,M)

        uc_e = ue[..., 1]
        un_e = ue[..., 0]
        us_e = ue if corrector else ue[..., 1:]

        coeffs_x = self.transform.get_coefficients(uc_x, un_x, us_x, p, side='x')  # (B,C,M)
        coeffs_e = self.transform.get_coefficients(uc_e, un_e, us_e, p, side='e')  # (B,C,M)

        # 창: (B, M, C, H, W)
        xs_win = torch.stack(xs[i - order + j:i + j][::-1], dim=1)
        es_win = torch.stack(es[i - order + j:i + j][::-1], dim=1)

        X = self._apply_coeffs_to_window(coeffs_x, xs_win)  # (B,C,H,W)
        E = self._apply_coeffs_to_window(coeffs_e, es_win)  # (B,C,H,W)

        # 계수도 (B,C)로 가정
        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(i, alphas, sigmas, ar, sr, p)  # (B,C)
        
        return sample_coeff[:, :, None, None] * sample + grad_coeff[:, :, None, None] * (X + E)

    def sample(self, xt, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = xt.device, xt.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        alphas = self.noise_schedule.marginal_alpha(timesteps)   # (steps+1,)
        sigmas = self.noise_schedule.marginal_std(timesteps)     # (steps+1,)
        ar = alphas[1:] / alphas[:-1]                            # (steps,)
        sr = sigmas[1:] / sigmas[:-1]

        x_pred = x_corr = xt
        xs, es = [], []
        context = nullcontext() if self.train_mode else torch.no_grad()
        with context:
            x, e = (self.checkpoint_model_fn(x_pred, timesteps[0]) if self.train_mode
                    else self.model_fn(x_pred, timesteps[0]))
            xs.append(x); es.append(e)

            # params: (B, 2, 5, C)  (배치/채널 일치한다고 가정)
            params, hidden = self.param_extractor({'x': xs[0], 'e': es[0],
                                                   't': timesteps[0:2], 'h': None, 'step': 0})

            use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
            for i in tqdm(range(self.steps), disable=not use_tqdm):
                order = (min(i + 1, self.steps - i, self.pred_order)
                         if self.lower_order_final else min(i + 1, self.pred_order))

                # Predictor: params[:, 0] → (B,5,C)
                x_pred = self.get_next_sample(i, x_corr, xs, es, alphas, sigmas, ar, sr,
                                              params[:, 0], order, corrector=False)

                if i < self.steps - 1:
                    x, e = (self.checkpoint_model_fn(x_pred, timesteps[i+1]) if self.train_mode
                            else self.model_fn(x_pred, timesteps[i+1]))
                    xs.append(x); es.append(e)
                    params, hidden = self.param_extractor({'x': xs[i+1], 'e': es[i+1],
                                                           't': timesteps[i+1:i+3],
                                                           'h': hidden, 'step': i+1})
                else:
                    break

                # Corrector
                if self.use_corrector:
                    order = min(self.corr_order, len(xs))
                    x_corr = self.get_next_sample(i, x_corr, xs, es, alphas, sigmas, ar, sr,
                                                  params[:, 1], order, corrector=True)
                else:
                    x_corr = x_pred

        return x_pred