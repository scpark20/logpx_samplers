import os
import torch
from tqdm import tqdm
import math
from .solver import Solver

class General_DPM_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
    ):
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def get_kernel_matrix(self, lambdas):
        return torch.vander(lambdas, N=len(lambdas), increasing=True)

    def get_integral_vector(self, lambda_s, lambda_t, lambdas):
        def get_integral(s, t, k):
            return (t**(k+1) - s**(k+1)) / (k+1)
        vector = [get_integral(lambda_s, lambda_t, k) for k in range(len(lambdas))]
        return torch.Tensor(vector).to(lambdas.device)

    def get_coefficients(self, lambda_s, lambda_t, lambdas):
        # (p,)
        integral = self.get_integral_vector(lambda_s, lambda_t, lambdas)
        # (p, p)
        kernel = self.get_kernel_matrix(lambdas)
        kernel_inv = torch.linalg.inv(kernel)
        # (p,)
        coefficients = kernel_inv.T @ integral
        return coefficients

    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, lambdas, p, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        corrector : True or False
        '''
        
        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array)

        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]

        data_sum = sum([coeff * torch.exp(l) * data for coeff, l, data in zip(coeffs, lambdas, datas)])
        if self.algorithm_type == 'data_prediction':
            next_sample = noise_rates[i+1]/noise_rates[i]*sample + noise_rates[i+1]*data_sum
        elif self.algorithm_type == 'noise_prediction':
            next_sample = signal_rates[i+1]/signal_rates[i]*sample - signal_rates[i+1]*data_sum
        else:
            next_sample = None
        return next_sample

    def sample(self, x, steps, skip_type='time_uniform_flow', order=3, flow_shift=1.0, use_corrector=False, **kwargs):
        
        lower_order_final = True

        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device = x.device

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps]).to(device)
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps]).to(device)
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps]).to(device)
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in range(0, steps):

                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)

                h = lambdas[i+1] - lambdas[i]
                # ===predictor===
                if p == 1:
                    sample_coeff = noise_rates[i+1]/noise_rates[i]
                    model_coeff = -signal_rates[i+1]*torch.expm1(-h)
                    x_pred = sample_coeff*x + model_coeff*hist[i]
                else:
                    x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, p=p, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                if use_corrector:
                    x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, p=p, corrector=True)
                    x = x_corr
                else:
                    x = x_pred

        # 최종적으로 x를 반환
        return x