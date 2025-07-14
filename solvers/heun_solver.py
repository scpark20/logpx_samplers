import os
import torch
from tqdm import tqdm
from .common import interpolate_fn, expand_dims
from .solver import Solver

class Heun_Solver(Solver):
    def __init__(self, model_fn, noise_schedule):
        super().__init__(model_fn, noise_schedule)

    def sample(self, x, steps, skip_type='time_uniform_flow', flow_shift=1.0, callback=None):
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            #print(timesteps)
            #timesteps = torch.tensor([0.9997,0.9,0.8,0.7,0.3,0.0], device=device)
            #print(timesteps)

            x_t = x

            for i in range(steps):
                dt = timesteps[i + 1] - timesteps[i]

                # 1) Predictor: compute slope at t_i
                v_t= self.model_fn(x_t, timesteps[i])
                x_pred = x_t + v_t * dt

                # 2) Corrector: compute slope at t_next (fallback to v_i on error)
                try:
                    v_next = self.model_fn(x_pred, timesteps[i + 1])
                except RuntimeError:
                    v_next = v_t

                # 평균 기울기로 업데이트
                v_avg = 0.5 * (v_t + v_next)
                x_t = x_t + v_avg * dt

                # 기록용 콜백 호출 (gradient도 함께 전달)
                if callback is not None:
                    callback(i, x_t, v_t)  # v_i는 gradient 역할을 하므로 그대로 전달

        return x_t
