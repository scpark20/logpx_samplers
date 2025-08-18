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

    def log_y(self, log_alpha, log_sigma, p, side):
        gamma = p['gamma']
        if side == 'x':
            return torch.where(gamma >= 0, log_alpha - gamma * log_sigma, (1.0 + gamma) * log_alpha)
        else:    
            return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma, log_sigma + gamma * log_alpha)

    def y(self, alpha, sigma, p, side):
        gamma = p['gamma']
        if side == 'x':
            return torch.where(gamma >= 0, alpha*(sigma**(-gamma)), alpha**(1.0+gamma))
        else:    
            return torch.where(gamma >= 0, sigma**(1.0-gamma), sigma*(alpha**gamma))
    
    def O2(self, delta, p, side='x'):
        kappa = p['kappa_x'] if side=='x' else p['kappa_e']
        return kappa * (delta ** 2)

    def get_sample_grad_coeff(self, i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, p):
        gamma = p['gamma']
        sample_coeff = torch.where(gamma >= 0, torch.exp(gamma * log_sigma_ratio[i]), torch.exp(-gamma * log_alpha_ratio[i]))
        grad_coeff = torch.where(gamma >= 0, torch.exp(gamma * log_sigma[i+1]), torch.exp(-gamma * log_alpha[i+1]))
        return sample_coeff, grad_coeff