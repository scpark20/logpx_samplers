import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .common import interpolate_fn, expand_dims
from torch.utils.checkpoint import checkpoint
from torch.autograd.graph import save_on_cpu

class Solver(nn.Module):
    def __init__(
        self,
        noise_schedule,
        algorithm_type
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["noise_prediction", "data_prediction", "vector_prediction", "dual_prediction"]
        self.algorithm_type = algorithm_type
        self.correcting_x0_fn = None

    # # ---------- time steps ----------
    # def learned_timesteps(self, device=None, dtype=None):
    #     if device is None: device = self.log_deltas.device
    #     if dtype  is None: dtype  = self.log_deltas.dtype
    #     T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
    #     t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)
    #     w = F.softmax(self.log_deltas, dim=0)   # (S,)
    #     deltas = (T - t_eps) * w
    #     c = torch.cumsum(deltas, dim=0)
    #     ts = torch.cat([T[None], T - c], dim=0)  # (S+1,)
    #     return ts

    def learned_timesteps(self, device=None, dtype=None):
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype

        T = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)

        # 경계 회피: [t_lo, t_hi] = [T/N, T*(1-1/N)]
        frac = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)
        t_lo = T * frac
        t_hi = T * (1.0 - frac)

        # 학습된 분할: softmax(log_deltas)로 [t_lo, t_hi]를 분할 (내림차순)
        w = F.softmax(self.log_deltas, dim=0)          # (S,)
        deltas = (t_hi - t_lo) * w                     # 합 = t_hi - t_lo
        c = torch.cumsum(deltas, dim=0)                # (S,)
        ts = torch.cat([t_hi[None], t_hi - c], dim=0)  # (S+1,) t_hi → … → t_lo
        return ts


    def set_model_fn(self, model_fn):
        def model_wrapper(x, t):
            # t is scalar or 1-length or [batch,]
            assert len(t.shape) == 0 or len(t) == 1 or len(t) == len(x)
            if len(t.shape) == 0 or len(t) == 1:
                t = t.expand(x.shape[0])
            #print(x.shape, t.shape)
            return model_fn(x, t)
        self.model = model_wrapper

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0    

    def checkpoint_model_fn(self, x, t):
        with save_on_cpu(pin_memory=True):
            #y = checkpoint(self.model_fn, x, t, use_reentrant=False)
            y = checkpoint(self.model_fn, x, t, use_reentrant=True)
        return y
    
    def model_fn(self, x, t):
        """
        Convert the model to the vector prediction model, the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "data_prediction":
            noise = self.model(x, t)
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            x0 = (x - sigma_t * noise) / alpha_t
            if self.correcting_x0_fn is not None:
                x0 = self.correcting_x0_fn(x0, t)
            return x0
        elif self.algorithm_type == "noise_prediction":
            return self.model(x, t)
        elif self.algorithm_type == "vector_prediction":
            noise = self.model(x, t)
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            vector = (x - noise) / -(1 - sigma_t)
            return vector
        elif self.algorithm_type == "dual_prediction":
            noise = self.model(x, t)
            alpha_t = self.noise_schedule.marginal_alpha(t)
            sigma_t = self.noise_schedule.marginal_std(t)
            x0 = (x - sigma_t[:, None, None, None] * noise) / alpha_t[:, None, None, None]
            if self.correcting_x0_fn is not None:
                x0 = self.correcting_x0_fn(x0, t)
            if x0.dtype != noise.dtype:
                noise = noise.to(dtype=x0.dtype)
            return (x0, noise)
        return None

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)
        
    def get_time_steps(self, skip_type, t_T, t_0, N, device, shift=1.0):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "time_uniform_flow":
            t_T = min(1.0 - 1e-3, t_T)
            t_0 = max(1e-3, t_0)
            betas = torch.linspace(t_T, t_0, N + 1).to(device)
            sigmas = 1.0 - betas
            sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
            return sigmas
        elif skip_type == "edm":
            # EDM-like polynomial schedule in rho(t) = sigma(t)/alpha(t).
            # Step 1: t_T, t_0 -> rho_T, rho_0
            t_T = min(1.0 - 1e-3, t_T)
            t_0 = max(1e-3, t_0)
            t_T_tensor = torch.as_tensor(t_T, device=device)
            t_0_tensor = torch.as_tensor(t_0, device=device)

            rho_T = self.noise_schedule.marginal_rho(t_T_tensor)  # scalar tensor
            rho_0 = self.noise_schedule.marginal_rho(t_0_tensor)

            # Step 2: polynomial interpolation in rho-space (EDM style)
            # rho_i = (rho_T^(1/r) + s_i * (rho_0^(1/r) - rho_T^(1/r)))^r
            # 보통 r ≈ 7 사용
            r = 7.0
            rho_T_r = rho_T.pow(1.0 / r)
            rho_0_r = rho_0.pow(1.0 / r)

            s = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=rho_T.dtype)
            rho_steps = (rho_T_r + s * (rho_0_r - rho_T_r)).pow(r)

            # Step 3: rho_i -> t_i via inverse_rho
            ts = self.noise_schedule.inverse_rho(rho_steps)
            return ts

        else:
            raise ValueError(
                f"Unsupported skip_type {skip_type}, need to be 'logSNR', "
                f"'time_uniform', 'time_quadratic', 'time_uniform_flow' or 'EDM'"
            )