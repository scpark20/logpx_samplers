import os
import torch
from tqdm import trange
from .common import interpolate_fn, expand_dims
from .solver import Solver

class Heun_Solver(Solver):
    def __init__(self, model_fn, noise_schedule):
        super().__init__(model_fn, noise_schedule)

    def sample(self, x, steps, skip_type='time_uniform_flow', flow_shift=1.0, callback=None):
        """Heun's method sampler with tuple support for conditional inputs."""
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        # detect device from latent tensor
        if isinstance(x, (tuple, list)):
            latent, *cond_args = x
            device = latent.device
        else:
            latent = x
            cond_args = []
            device = x.device

        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            

            x_t = (latent, *cond_args) if cond_args else latent
            for i in range(steps):
                t_i = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_i

                # unpack for predictor
                if isinstance(x_t, (tuple, list)):
                    latent_i, *rest = x_t
                else:
                    latent_i = x_t
                    rest = []

                # 1) Predictor slope
                v_i = self.model_fn(latent_i, t_i, *rest)
                latent_pred = latent_i + v_i * dt
                x_pred = (latent_pred, *rest) if rest else latent_pred

                # 2) Corrector slope
                if isinstance(x_pred, (tuple, list)):
                    latent_p, *rest_p = x_pred
                else:
                    latent_p = x_pred
                    rest_p = []

                try:
                    v_next = self.model_fn(latent_p, t_next, *rest_p)
                except RuntimeError:
                    # fallback to predictor slope on error
                    v_next = v_i

                # average slope and update latent
                v_avg = 0.5 * (v_i + v_next)
                latent_i = latent_i + v_avg * dt

                # reassemble x_t including conditional args
                x_t = (latent_i, *rest) if rest else latent_i

                if callback is not None:
                    callback(i, x_t)

        return x_t
