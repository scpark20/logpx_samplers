import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from .networks import AMED_predictor

def get_amed_prediction(AMED_predictor, t_cur, t_next, unet_out):
    unet_enc = torch.mean(unet_out, dim=1)
    output = AMED_predictor(unet_enc, t_cur, t_next)
    output_list = [*output]
    
    if len(output_list) == 2:
        try:
            use_scale_time = AMED_predictor.module.scale_time
        except:
            use_scale_time = AMED_predictor.scale_time
        if use_scale_time:
            r, scale_time = output_list
            r = r.reshape(-1, 1, 1, 1)
            scale_time = scale_time.reshape(-1, 1, 1, 1)
            scale_dir = torch.ones_like(scale_time)
        else:
            r, scale_dir = output_list
            r = r.reshape(-1, 1, 1, 1)
            scale_dir = scale_dir.reshape(-1, 1, 1, 1)
            scale_time = torch.ones_like(scale_dir)
    elif len(output_list) == 3:
        r, scale_dir, scale_time = output_list
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = scale_time.reshape(-1, 1, 1, 1)
    else:
        r = output.reshape(-1, 1, 1, 1)
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    return r[:, 0, 0, 0], scale_dir[:, 0, 0, 0], scale_time[:, 0, 0, 0]

class AMED_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform",
        flow_shift=1.0,
        algorithm_type="dual_prediction",
        bottleneck_dim=1024,
        checkpoint=False,
        use_afs=True,
        **kwargs
    ):
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.checkpoint = checkpoint
        self.use_afs = use_afs
        
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        self.timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.predictor = AMED_predictor(sampler_stu='amed', sampler_tea='euler', bottleneck_input_dim=bottleneck_dim)

        # self.r = nn.Parameter(torch.zeros(steps,))
        # self.scale_dir = nn.Parameter(torch.zeros(steps,))
        # self.scale_time = nn.Parameter(torch.zeros(steps,))

    def eval_model(self, y, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        x = y * alpha[:, None, None, None]
        return self.checkpoint_model_fn(x, t) if self.checkpoint else self.model_fn(x, t)

    def get_x(self, y, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        return y * alpha

    def get_y(self, x, rho):
        t = self.noise_schedule.inverse_rho(rho)
        alpha = self.noise_schedule.marginal_alpha(t)
        return x / alpha

    def get_params(self, step):
        r = self.r[step]
        scale_dir = self.scale_dir[step]
        scale_time = self.scale_time[step]
        return torch.sigmoid(r), torch.exp(scale_dir), torch.exp(scale_time)

    def sample(self, x, model_fn, output_traj=False, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.timesteps.to(device=device, dtype=dtype)
        rhos = self.noise_schedule.marginal_rho(timesteps)
        
        y = self.get_y(x, rhos[0])
        trajs = [x,]
        for i in range(self.steps):
            if self.use_afs and i == 0:
                data, noise = torch.zeros_like(x), y / ((1 + rhos[i]**2).sqrt())
            else:
                data, noise = self.eval_model(y, rhos[i])
                
            # Mid
            #r, scale_dir, scale_time = self.get_params(i)
            r, scale_dir, scale_time = get_amed_prediction(self.predictor, rhos[i], rhos[i+1], data.detach())
            #rho_mid = (rhos[i+1]**r) * (rhos[i]**(1-r))
            rho_mid = (rhos[i+1]*r) + (rhos[i]*(1-r))
            y_next = y + (rho_mid - rhos[i:i+1])[:, None, None, None] * noise
            _, noise = self.eval_model(y_next, scale_time * rho_mid)

            # Final
            y = y + (scale_dir * (rhos[i+1:i+2] - rhos[i:i+1]))[:, None, None, None] * noise
            x = self.get_x(y, rhos[i+1])
            trajs.append(x)
        
        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs