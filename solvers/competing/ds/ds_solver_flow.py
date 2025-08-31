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
        algorithm_type="vector_prediction",
        checkpoint=False,
        **kwargs
    ):
        assert algorithm_type == 'vector_prediction'
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
        dt = timesteps[1:] - timesteps[:-1]
        
        solver_coeffs = self.M
        pred_trajectory = []
        trajs = [x,]
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            vc = self.checkpoint_model_fn(x, timesteps[i]) if self.checkpoint else self.model_fn(x, timesteps[i])
            pred_trajectory.append(vc)

            velocity = torch.zeros_like(vc)
            sum_solver_coeff = 0.0
            for j in range(i):
                if self.steps <= 6 and i == self.steps - 2 and j != i - 1: continue
                if self.steps <= 6 and i == self.steps - 1 and j != i - 1: continue
                velocity += solver_coeffs[i, j] * pred_trajectory[j]
                sum_solver_coeff += solver_coeffs[i, j]
                
            velocity += (1 - sum_solver_coeff) * pred_trajectory[-1]
            x = x + velocity*dt[i]
            trajs.append(x)
            
        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs