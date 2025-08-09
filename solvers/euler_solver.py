import torch
from .solver import Solver

class Euler_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
    ):
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def sample(self, x, steps, skip_type='time_uniform_flow', flow_shift=1.0, output_traj=False, model_fn=None, **kwargs):
        if model_fn is not None:
            self.model_fn = model_fn
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device
        
        trajs = []    
        with torch.no_grad():
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            lambdas = torch.tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps], device=device)
            signal_rates = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            noise_rates = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)
            
            x_t = x
            if output_traj:
                trajs.append(x_t.detach().cpu())

            for i in range(0, steps):
                h = lambdas[i+1] - lambdas[i]
                model_t = self.model_fn(x_t, timesteps[i])
                if self.algorithm_type == 'vector_prediction':
                    sample_coeff = 1
                    model_coeff = (timesteps[i+1] - timesteps[i])
                    x_t = sample_coeff*x_t + model_coeff*model_t
                    
                elif self.algorithm_type == 'data_prediction':
                    sample_coeff = noise_rates[i+1]/noise_rates[i]
                    model_coeff = -signal_rates[i+1]*torch.expm1(-h)
                    x_t = sample_coeff*x_t + model_coeff*model_t

                elif self.algorithm_type == 'noise_prediction':
                    sample_coeff = signal_rates[i+1]/signal_rates[i]
                    model_coeff = -noise_rates[i+1]*torch.expm1(h)
                    x_t = sample_coeff*x_t + model_coeff*model_t
                
                elif self.algorithm_type == 'dual_prediction':
                    data_coeff = signal_rates[i+1]-signal_rates[i]
                    noise_coeff = noise_rates[i+1]-noise_rates[i]
                    x_t = x_t + data_coeff*model_t[0] + noise_coeff*model_t[1]

                if output_traj:
                    trajs.append(x_t.detach().cpu())

        outputs = {'samples': x_t}
        if output_traj:            
            outputs['trajs'] = torch.stack(trajs, dim=1)

        return outputs