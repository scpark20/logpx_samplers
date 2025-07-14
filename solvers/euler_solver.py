import os
import torch
from tqdm import tqdm
from .common import interpolate_fn, expand_dims
from .solver import Solver

class Euler_Solver(Solver):
    def __init__(self, model_fn, noise_schedule):
        super().__init__(model_fn, noise_schedule)
        
    def sample(self, x, steps, skip_type='time_uniform_flow', flow_shift=1.0, callback=None):
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device
        
        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            #print(timesteps)
            x_t = x
            
            for i in range(steps):
                v_t = self.model_fn(x_t, timesteps[i])
                
                x_t = x_t + v_t * (timesteps[i+1] - timesteps[i])
                
                # Call callback function if provided
                if callback is not None:
                    callback(i, x_t, v_t)  # v_i는 gradient 역할을 하므로 그대로 전달
        
        return x_t
