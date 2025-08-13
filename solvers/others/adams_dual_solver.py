import os
import torch
from tqdm import tqdm
import math
from .solver import Solver

class Adams_Dual_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dual_prediction",
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def get_kernel_matrix(self, times):
        return torch.vander(times, N=len(times), increasing=True)

    def get_integral_vector(self, s, t, times):
        def get_integral(s, t, k):
            return (t**(k+1) - s**(k+1)) / (k+1)
        vector = [get_integral(s, t, k) for k in range(len(times))]
        return torch.tensor(vector).to(times.device)

    def get_coefficients(self, time_s, time_t, times):
        # (p,)
        integral = self.get_integral_vector(time_s, time_t, times)
        # (p, p)
        kernel = self.get_kernel_matrix(times)
        kernel_inv = torch.linalg.inv(kernel)
        # (p,)
        coefficients = kernel_inv.T @ integral
        return coefficients

    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, times, p, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        corrector : True or False
        '''

        signal_diff = (signal_rates[i+1] - signal_rates[i])/(times[i+1] - times[i])
        noise_diff = (noise_rates[i+1] - noise_rates[i])/(times[i+1] - times[i])

        time_array = torch.flip(times[i-p+1:i+(2 if corrector else 1)], dims=[0])
        coeffs = self.get_coefficients(times[i], times[i+1], time_array)               

        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]

        data_sum = sum([coeff * (signal_diff*data[0] + noise_diff*data[1]) for coeff, data in zip(coeffs, datas)])
        next_sample = sample + data_sum
        return next_sample

    def sample(self, x, steps, skip_type='time_uniform_flow', order=3, flow_shift=1.0, use_corrector=True, **kwargs):
        
        lower_order_final = True  # 전체 스텝이 매우 작을 때 마지막 스텝에서 차수를 낮춰서 안정성 확보할지.

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        assert t_0 > 0 and t_T > 0, "Time range( t_0, t_T )는 0보다 커야 함. (Discrete DPMs: [1/N, 1])"

        # 텐서가 올라갈 디바이스 설정
        device = x.device

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift)
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps])
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps])
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in tqdm(range(0, steps), disable=os.getenv("TQDM", "False")):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)
                    
                # ===predictor===
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, timesteps, p=p, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                if use_corrector:
                    # ===corrector===
                    x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, timesteps, p=p, corrector=True)
                    x = x_corr
                else:
                    x = x_pred
        # 최종적으로 x를 반환
        return x