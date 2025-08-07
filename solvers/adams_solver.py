import os
import torch
from tqdm import tqdm
import math
from .solver import Solver

class Adams_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
    ):
        super().__init__(model_fn, noise_schedule, algorithm_type)
        
    def get_kernel_matrix(self, lambdas):
        return torch.vander(lambdas, N=len(lambdas), increasing=True)

    def get_integral(self, a: float, b: float, k: int) -> float:
        if k < 0 or not float(k).is_integer():
            raise ValueError("k must be a non-negative integer.")

        k = int(k)  # 확실하게 int 변환
        k_factorial = math.factorial(k)

        def F(x: float) -> float:
            # F(λ) = -k! * exp(-λ) * Σ_{m=0}^k [λ^m / m!]
            poly_sum = 0.0
            for m in range(k+1):
                poly_sum += (x**m) / math.factorial(m)

            return -k_factorial * math.exp(-x) * poly_sum
        
        def G(x: float) -> float:
            # G(λ) = (-1)^k * k! * exp(λ) * Σ_{m=0}^k [(-λ)^m / m!]
            poly_sum = 0.0
            for m in range(k+1):
                poly_sum += ((-x)**m) / math.factorial(m)

            return (-1)**k * k_factorial * math.exp(x) * poly_sum

        if self.algorithm_type == 'data_prediction':
            return G(b) - G(a)
        else:
            return F(b) - F(a)

    def get_integral_vector(self, lambda_s, lambda_t, lambdas):
        vector = [self.get_integral(lambda_s, lambda_t, k) for k in range(len(lambdas))]
        return torch.Tensor(vector, device=lambdas.device)

    def get_coefficients(self, lambda_s, lambda_t, lambdas):
        # (p,)
        integral = self.get_integral_vector(lambda_s, lambda_t, lambdas)
        # (p, p)
        kernel = self.get_kernel_matrix(lambdas)
        kernel_inv = torch.linalg.inv(kernel)
        # (p,)
        coefficients = kernel_inv.T @ integral
        return coefficients

    def get_vector_integral_vector(self, s, t, times):
        def get_vector_integral(s, t, k):
            return (t**(k+1) - s**(k+1)) / (k+1)
        vector = [get_vector_integral(s, t, k) for k in range(len(times))]
        return torch.tensor(vector).to(times.device)

    def get_vector_coefficients(self, time_s, time_t, times):
        # (p,)
        integral = self.get_vector_integral_vector(time_s, time_t, times)
        # (p, p)
        kernel = self.get_kernel_matrix(times)
        kernel_inv = torch.linalg.inv(kernel)
        # (p,)
        coefficients = kernel_inv.T @ integral
        return coefficients

    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, times, lambdas, p, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        corrector : True or False
        '''
        
        if self.algorithm_type == 'data_prediction' or self.algorithm_type == 'noise_prediction':
            # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
            # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
            lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

            # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
            # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
            coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array)

        elif self.algorithm_type == 'vector_prediction':
            time_array = torch.flip(times[i-p+1:i+(2 if corrector else 1)], dims=[0])
            coeffs = self.get_vector_coefficients(times[i], times[i+1], time_array)               
    
        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])
        if self.algorithm_type == 'data_prediction':
            next_sample = noise_rates[i+1]/noise_rates[i]*sample + noise_rates[i+1]*data_sum
        elif self.algorithm_type == 'noise_prediction':
            next_sample = signal_rates[i+1]/signal_rates[i]*sample - signal_rates[i+1]*data_sum
        elif self.algorithm_type == 'vector_prediction':
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
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps])
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
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, timesteps, lambdas, p=p, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                if use_corrector:
                    # ===corrector===
                    x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, timesteps, lambdas, p=p, corrector=True)
                    x = x_corr
                else:
                    x = x_pred
                    
        # 최종적으로 x를 반환
        return x