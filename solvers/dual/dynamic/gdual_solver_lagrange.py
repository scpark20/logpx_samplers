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
        self.train_mode = train_mode

        # learned timesteps (weights over (T - t_eps))
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

    def get_vandermonde_matrix(self, u):
        # u : (u_i, u_i-1, ..., u_i-p+1)
        return torch.vander(u, increasing=True)

    def get_moments(self, uc, un, tau, p):
        # uc : (B,), un : (B,), tau : (B,), p : scalar
        
        def F0(u, tau):
            return 1/tau * torch.exp(tau*u)

        def Fk(u, tau, k, Fk_1):
            if k == 0:
                return F0(u, tau)
            return torch.exp(tau*u)/tau * u**k - k/tau * Fk_1

        ms = []
        F_uc, F_un = None, None
        for k in range(p):
            F_uc, F_un = Fk(uc, tau, k, F_uc), F0(un, tau, k, F_un)
            ms.append(F_un - F_uc)
        ms = torch.stack(ms, dim=1)

        return ms


    def get_coefficients(self, u, uc, un, tau):
        # u : (B, p,), uc : (B,), un : (B,), tau : (B,)
        # V : (B, p, p)
        V = torch.vander(u, increasing=True)
        # W : (p, p)
        W = torch.linalg.inv(V).T
        # m : (p,)
        m = self.get_moments(self, uc, un, tau, len(u))
        return W.T @ m

    def get_next_sample(self, i, x, outputs, alphas, sigmas, params, order, corrector=False):
        p = self.transform.unpack(params)

        
        
        return out

    # ---------- sampling ----------
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        fn = self.checkpoint_model_fn if self.train_mode else self.model_fn

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        # noise schedule (벡터화)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)
        
        context = nullcontext() if self.train_mode else torch.no_grad()
        with context:
            # 초기 상태
            x_pred = x_corr = x
            outputs = [fn(x_pred, timesteps[0])]
            params, hidden = self.param_extractor({'x':outputs[0][0], 'e':outputs[0][0], 't': timesteps[0:2], 'h': None})
            
            use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0","False","false","")
            for i in tqdm(range(self.steps), disable=not use_tqdm):
                order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

                # Predictor
                x_pred = self.get_next_sample(i, x_corr, outputs, alphas, sigmas, params[:, 0], order, corrector=False)
                
                if i < self.steps - 1:
                    outputs.append(fn(x_pred, timesteps[i+1]))
                    params, hidden = self.param_extractor({'x':outputs[i+1][0], 'e':outputs[i+1][1], 't': timesteps[i+1:i+3], 'h': hidden})
                else:
                    break

                # Corrector
                if self.use_corrector:
                    x_corr = self.get_next_sample(i, x_corr, outputs, alphas, sigmas, params[:, 0], order, corrector=False)
                else:
                    x_corr = x_pred

        return x_pred
