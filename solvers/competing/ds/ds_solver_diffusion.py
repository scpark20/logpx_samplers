import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class DS_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        algorithm_type="dual_prediction",
        checkpoint=False,
        **kwargs
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.checkpoint = checkpoint
        
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))
        self.M = nn.Parameter(torch.eye(steps) * 0)
        
    def sample(self, x, model_fn, output_traj=False, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        # noise_schedule이 텐서를 받아들일 수 있어야 자동미분이 유지됩니다.
        alphas = self.noise_schedule.marginal_alpha(timesteps)          # 벡터화된 구현 권장
        sigmas = self.noise_schedule.marginal_std(timesteps)
        lambdas = alphas / sigmas
        delta_lambdas = lambdas[1:] - lambdas[:-1]
        
        solver_coeffs = self.M
        pred_trajectory = []
        trajs = [x,]
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            xc, ec = self.checkpoint_model_fn(x, timesteps[i]) if self.checkpoint else self.model_fn(x, timesteps[i])
            pred_trajectory.append(xc)

            dpmeps = torch.zeros_like(xc)
            sum_solver_coeff = 0.0
            for j in range(i):
                if self.steps <= 6 and i == self.steps - 2 and j != i - 1: continue
                if self.steps <= 6 and i == self.steps - 1 and j != i - 1: continue
                dpmeps += solver_coeffs[i, j] * pred_trajectory[j]
                sum_solver_coeff += solver_coeffs[i, j]
            dpmeps += (1 - sum_solver_coeff) * pred_trajectory[-1]
            x = (sigmas[i+1] / sigmas[i]) * x + sigmas[i+1] * (delta_lambdas[i]) * dpmeps
            trajs.append(x)
            
        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs