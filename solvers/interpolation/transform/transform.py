import torch

class Transform:
    def push_away(self, x: torch.Tensor, pivot, eps: float, ste: bool = True) -> torch.Tensor:
        """
        |x - pivot| < eps 인 원소를 pivot ± eps로 밀어냄(부호 유지).
        ste=True면 straight-through estimator로 gradient는 x 기준 그대로 흐름.
        pivot은 스칼라나 브로드캐스트 가능한 텐서 가능.
        """
        d = x - pivot
        near = d.abs() < eps
        s = torch.sign(d)
        s = torch.where(s == 0, torch.ones_like(s), s)  # 정확히 pivot일 때 +eps로 밀기
        x_proj = torch.where(near, (torch.as_tensor(pivot, dtype=x.dtype, device=x.device) + s * eps), x)
        return x + (x_proj - x).detach() if ste else x_proj

    def y(self, alpha, sigma, p, side):
        # alpha: (L,), sigma: (L,), gamma: (B,)
        alpha = alpha.unsqueeze(0)  # (1, L)
        sigma = sigma.unsqueeze(0)  # (1, L)
        gamma = p['gamma'].unsqueeze(-1)  # (B, 1)

        if side == 'x':
            return torch.where(gamma >= 0, alpha * (sigma ** (-gamma)), alpha ** (1.0 + gamma))  # (B, L)
        else:
            return torch.where(gamma >= 0, sigma ** (1.0 - gamma), sigma * (alpha ** gamma))     # (B, L)
    
    def get_sample_grad_coeff(self, i, alphas, sigmas, alphas_ratio, sigmas_ratio, p):
        # i in [0, steps-1]
        # alpha : (steps+1,), alpha_ratio : (steps,)
        # gamma : (B,)
        
        gamma = p['gamma']
        # sample_coeff : (B,), grad_coeff : (B,)
        sample_coeff = torch.where(gamma >= 0, sigmas_ratio[i]**gamma,  alphas_ratio[i]**(-gamma))
        grad_coeff = torch.where(gamma >= 0,  sigmas[i+1]**gamma, alphas[i+1]**(-gamma))
        return sample_coeff, grad_coeff
