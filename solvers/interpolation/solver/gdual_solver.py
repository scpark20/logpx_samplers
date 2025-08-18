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
        scale_learning=False,
        time_learning=True,
        use_corrector=True,
        train_mode=False,
    ):
        super().__init__(noise_schedule, 'dual_prediction')

        self.steps = steps
        self.skip_type = skip_type
        self.pred_order = pred_order
        self.corr_order = corr_order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps
        self.scale_learning = scale_learning
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
        if self.scale_learning:
            self.scale = nn.Parameter(torch.ones(1,))

    def get_next_sample(self, i, sample, xs, es, alphas, sigmas, alphas_ratio, sigmas_ratio, params, order, corrector=False):
        # corrector는 order+1임
        # i : scalar, sample : scalar
        # xs : [x_i, x_i-1, ..., x_0], es : [e_i, e_i-1, ..., e_0]
        # alphas : [a_0, a_1, ..., a_steps+1], sigmas : [s_0, s_1, ..., s_steps+1],
        # alphas_ratio : [ar_0, ar_1, ..., ar_steps], sigmas_ratio : [sr_0, sr_1, ..., sr_steps]
        # params : (1 or B, n_params)
        # order : scalar, corrector : boolean
        
        p = self.transform.unpack(params)
        
        j = 2 if corrector else 1
        # pred : [a_i+1, a_i, a_i-1, ..., a_i-order+1]
        # corr : [a_i+1, a_i, a_i-1, ..., a_i-order+2]
        alphas_win = torch.flip(alphas[i-order+j:i+2], dims=[0])
        sigmas_win = torch.flip(sigmas[i-order+j:i+2], dims=[0])
        # (B, order)
        yx = self.transform.y(alphas_win, sigmas_win, p, side='x')
        ye = self.transform.y(alphas_win, sigmas_win, p, side='e')
        # pred : (B, order), [[u_i+1, u_i, u_i-1, ..., u_i-order+1], ...]
        # corr : (B, order), [[u_i+1, u_i, u_i-1, ..., u_i-order+2], ...]
        ux = self.transform.L(yx, p, side='x')
        ue = self.transform.L(ye, p, side='e')
        
        # (B, order)
        coeffs_x = self.transform.get_coefficients(ux[:, 1], ux[:, 0], ux if corrector else ux[:, 1:], p, side='x')
        coeffs_e = self.transform.get_coefficients(ue[:, 1], ue[:, 0], ue if corrector else ue[:, 1:], p, side='e')
        
        # (B, order, C, H, W)
        xs_win = torch.stack(xs[i-order+j:i+j][::-1], dim=1)
        es_win = torch.stack(es[i-order+j:i+j][::-1], dim=1)
        
        # (B, C, H, W)
        X = (coeffs_x[:, :, None, None, None] * xs_win).sum(dim=1)
        E = (coeffs_e[:, :, None, None, None] * es_win).sum(dim=1)

        # (B,), (B,)
        sample_coeff, grad_coeff = self.transform.get_sample_grad_coeff(i, alphas, sigmas, alphas_ratio, sigmas_ratio, p)
        next_sample = sample_coeff[:, None, None, None]*sample + grad_coeff[:, None, None, None]*(X + E)

        return next_sample

    # ---------- sampling ----------
    def sample(self, xt, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = xt.device, xt.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        # (steps+1,)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)
        # (steps,)
        alphas_ratio = alphas[1:]/alphas[:-1]
        sigmas_ratio = sigmas[1:]/sigmas[:-1]
        
        # 초기 상태
        if self.scale_learning:
            xt = xt * self.scale
        x_pred = x_corr = xt
        xs = []
        es = []
        context = nullcontext() if self.train_mode else torch.no_grad()
        with context:
            x, e = self.checkpoint_model_fn(x_pred, timesteps[0]) if self.train_mode else self.model_fn(x_pred, timesteps[0])
            xs.append(x); es.append(e)
            # params : (1 or B, 2, n_params)
            params, hidden = self.param_extractor({'x':xs[0], 'e':es[0], 't': timesteps[0:2], 'h': None, 'step': 0})
            
            use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
            for i in tqdm(range(self.steps), disable=not use_tqdm):
                order = min(i + 1, self.steps - i, self.pred_order) if self.lower_order_final else min(i + 1, self.pred_order)

                # Predictor
                x_pred = self.get_next_sample(i, x_corr, xs, es, alphas, sigmas, alphas_ratio, sigmas_ratio,
                params[:, 0], order, corrector=False)
                
                if i < self.steps - 1:
                    x, e = self.checkpoint_model_fn(x_pred, timesteps[i+1]) if self.train_mode else self.model_fn(x_pred, timesteps[i+1]) 
                    xs.append(x); es.append(e)
                    params, hidden = self.param_extractor({'x':xs[i+1], 'e':es[i+1], 't': timesteps[i+1:i+3], 'h': hidden, 'step': i+1})
                else:
                    break
                
                # Corrector
                order = min(self.corr_order, len(xs))
                assert order >= 2
                if self.use_corrector:
                    x_corr = self.get_next_sample(i, x_corr, xs, es, alphas, sigmas, alphas_ratio, sigmas_ratio,
                    params[:, 1], order, corrector=True)
                else:
                    x_corr = x_pred
                
        return x_pred
