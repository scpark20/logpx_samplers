import torch

# -------------------------------
# Linear–Log transform utilities
# L(y, τ) = (1-τ) log y + τ y
# -------------------------------
class LinearLogTransform:
    @staticmethod
    def L_from_logy(logy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Kept for completeness; not needed to get Δu if you use the closed form below.
        return (1.0 - tau) * logy + tau * torch.exp(logy)

    @staticmethod
    def logy_u(log_alpha: torch.Tensor, log_sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # y_u = α σ^{-γ}  (γ>=0)  |  y_u = α^{1+γ} (γ<0)
        return torch.where(gamma >= 0, log_alpha - gamma * log_sigma, (1.0 + gamma) * log_alpha)

    @staticmethod
    def logy_v(log_alpha: torch.Tensor, log_sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # y_v = σ^{1-γ}   (γ>=0)  |  y_v = σ α^{γ}  (γ<0)
        return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma, log_sigma + gamma * log_alpha)

    @staticmethod
    def g_from_y(y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # g(u) = dL^{-1}(u)/du = y / ((1-τ) + τ y)
        return y / ((1.0 - tau) + tau * y)

    @staticmethod
    def O_delta_square(delta: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        return kappa * (delta ** 2)

    @staticmethod
    def param_preprocess(gamma, tau_x, tau_e, kx, ke):
        return gamma, torch.sigmoid(tau_x), torch.sigmoid(tau_e), kx, ke