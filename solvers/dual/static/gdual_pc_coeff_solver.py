import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class GDual_PC_Coeff_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        order=2,
        lower_order_final=True,
        eps=1e-8,
        algorithm_type="dual_prediction",    
    ):
        assert algorithm_type == 'dual_prediction'
        assert order <= 2
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps
        
        init_params = torch.zeros(steps, 2, 5)
        self.params = nn.Parameter(init_params)            
        
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        timesteps = self.get_time_steps(skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=self.steps, device=device, shift=self.flow_shift)
        
        x_pred = x
        x_corr = x
        xp, ep = None, None
        xn, en = None, None
        xc, ec = self.checkpoint_model_fn(x_corr, timesteps[0])
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i+1, self.steps - i, self.order) if self.lower_order_final else min(i+1, self.order)

            c_x, c_data, c_noise, c_datav, c_noisev = self.params[i, 0]
            x_pred = c_x*x_corr + c_data*xc + c_noise*ec
            if p > 1:
                x_pred = x_pred + c_datav*(xc-xp)/2 + c_noisev*(ec-ep)/2

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
            else:
                break    

            c_x, c_data, c_noise, c_datav, c_noisev = self.params[i, 1]
            x_corr = c_x*x_corr + c_data*xc + c_noise*ec + c_datav*(xn-xc)/2 + c_noisev*(en-ec)/2

            xp = xc; ep = ec; xc = xn; ec = en
 
        return x_pred