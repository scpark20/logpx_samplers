import torch

# -------------------------------
# Log–Linear transform utilities
# L(y, τ) = (1-τ) * y + τ * log(y)
# (τ=0 → linear, τ=1 → log)
# -------------------------------

class LogLinearTransform:
    def __init__(self, gamma_max=None, kappa_max=None, gamma_push=False, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.kappa_max = kappa_max

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

    def unpack(self, params, eps):
        # params: [B,5] or [*,5]
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        kx    = params[:, 3]
        ke    = params[:, 4]
        tau_x = torch.sigmoid(tau_x)
        tau_e = torch.sigmoid(tau_e)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kx = torch.tanh(kx) * self.kappa_max
            ke = torch.tanh(ke) * self.kappa_max
        if self.gamma_push:
            gamma = self.push_away(gamma, 1, eps)
            gamma = self.push_away(gamma, -1, eps)
        return gamma, tau_x, tau_e, kx, ke
    
    def delta_from_logy(self, logy_n, logy_c, tau):
        # Log–Linear: L(y,τ)=(1-τ)·y + τ·log y
        dlog = logy_n - logy_c                              # [B,1,1,1]
        return (1.0 - tau) * torch.exp(logy_c) * torch.expm1(dlog) + tau * dlog

    def delta_y_from_logs(self, logy_c: torch.Tensor, logy_n: torch.Tensor) -> torch.Tensor:
        """
        clamp 없이 y_n - y_c를 안정적으로 계산.
        아이디어: m = max(logy_c, logy_n)로 묶어서
                  y_n - y_c = exp(m) * [exp(logy_n - m) - exp(logy_c - m)]
                  여기서 (logy_n - m), (logy_c - m) ≤ 0 → overflow 회피.
        """
        m = torch.maximum(logy_c, logy_n)      # no clamp
        a = logy_n - m                         # ≤ 0
        b = logy_c - m                         # ≤ 0
        return torch.exp(m) * (torch.expm1(a) - torch.expm1(b))
    
    def L_from_logy(self, logy, tau):
        return (1.0 - tau) * torch.exp(logy) + tau * logy
    
    def logy_u(self, log_alpha, log_sigma, gamma):
        return torch.where(gamma >= 0, log_alpha - gamma * log_sigma, (1.0 + gamma) * log_alpha)
    
    def logy_v(self, log_alpha, log_sigma, gamma):
        return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma, log_sigma + gamma * log_alpha)

    def g_from_y(self, y, tau):
        return y / ((1.0 - tau) * y + tau)

    def O_delta_square(self, delta, kappa):
        return kappa * (delta ** 2)
    
    def get_sample_grad_coeff(self, i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, gamma):
        sample_coeff = torch.where(gamma >= 0, torch.exp(gamma * log_sigma_ratio[i]), torch.exp(-gamma * log_alpha_ratio[i]))
        grad_coeff = torch.where(gamma >= 0, torch.exp(gamma * log_sigma[i+1]), torch.exp(-gamma * log_alpha[i+1]))
        return sample_coeff, grad_coeff