import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ..solver import Solver

class BNS_Solver(Solver):
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
        param_dim=(),
        O2_coeff=False,
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

        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

        self.a = nn.Parameter(torch.ones(steps,))
        self.b = nn.Parameter(torch.ones(steps, steps))
        
    def learned_timesteps(self, device=None, dtype=None):
        """
        log_deltas (length = steps)  ->  timesteps (length = steps+1, strictly decreasing)
        anchors: t[0] = T, t[-1] = t_eps
        """
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype

        T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)

        # 1) 양수 간격 + 총합 고정: softmax로 비율을 만들고 전체 스팬에 맞춤
        w = F.softmax(self.log_deltas, dim=0)            # (S,)
        deltas = (T - t_eps) * w                         # (S,), sum(deltas) = T - t_eps

        # 2) 누적합으로 감소하는 시간축 복원
        c = torch.cumsum(deltas, dim=0)                  # (S,)
        ts = torch.cat([T[None], T - c], dim=0)          # (S+1,)
        return ts
    
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        # noise_schedule이 텐서를 받아들일 수 있어야 자동미분이 유지됩니다.
        alphas = self.noise_schedule.marginal_alpha(timesteps)          # 벡터화된 구현 권장
        sigmas = self.noise_schedule.marginal_std(timesteps)
        delta_alphas = alphas[1:] - alphas[:-1]
        delta_sigmas = sigmas[1:] - sigmas[:-1]
        
        x0 = x
        vs = []
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            xc, ec = self.checkpoint_model_fn(x, timesteps[i])
            vc = delta_alphas[i]*xc + delta_sigmas[i]*ec
            vs.append(vc)

            x = x0 * self.a[i]
            for j in range(0, i+1):
                x = x + vs[j] * self.b[i, j]
            
        return x