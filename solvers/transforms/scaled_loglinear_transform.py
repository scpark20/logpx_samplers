import torch
import torch.nn.functional as F

class ScaledLogLinearTransform:
    @staticmethod
    def _unpack(params):
        # params: [B, 7, 1, 1, 1]  (after you pick predictor/corrector slice)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        kx    = params[:, 3]
        ke    = params[:, 4]
        cx    = params[:, 5]
        ce    = params[:, 6]
        return gamma, tau_x, tau_e, kx, ke, cx, ce

    @staticmethod
    def _tau_c(params, x_side: bool):
        gamma, tau_x, tau_e, kx, ke, cx, ce = ScaledLogLinearTransform._unpack(params)
        # τ ∈ (0,1)
        tau = torch.sigmoid(tau_x) if x_side else torch.sigmoid(tau_e)
        # c > 0 보장 (Softplus + clamp로 안정화)
        c_raw = cx if x_side else ce
        c = F.softplus(c_raw) + 1e-6
        c = torch.clamp(c, 1e-4, 1e4)
        kappa = kx if x_side else ke
        return tau, c, gamma, kappa

    # -------- core maps --------
    @staticmethod
    def L_from_logy(logy, params, x_side=True):
        # L(y;τ,c) = (1-τ) * (y/c) + τ * log(y/c)  where y = exp(logy)
        tau, c, _, _ = ScaledLogLinearTransform._tau_c(params, x_side)
        return (1.0 - tau) * torch.exp(logy) / c + tau * (logy - torch.log(c))

    @staticmethod
    def delta_from_logy(logy_n, logy_c, params, x_side=True):
        # Δ = L(y_n)-L(y_c) = (1-τ)*(y_c/c)*expm1(dlog) + τ*dlog    (상수 -τlog c는 소거)
        tau, c, _, _ = ScaledLogLinearTransform._tau_c(params, x_side)
        dlog = logy_n - logy_c
        return (1.0 - tau) * torch.exp(logy_c) * torch.expm1(dlog) / c + tau * dlog

    @staticmethod
    def logy_u(log_alpha, log_sigma, params):
        # y_u = α σ^{-γ}  (γ>=0) | y_u = α^{1+γ} (γ<0)
        _, _, gamma, _ = ScaledLogLinearTransform._tau_c(params, True)
        return torch.where(gamma >= 0, log_alpha - gamma * log_sigma, (1.0 + gamma) * log_alpha)

    @staticmethod
    def logy_v(log_alpha, log_sigma, params):
        # y_v = σ^{1-γ}  (γ>=0) | y_v = σ α^{γ}  (γ<0)
        _, _, gamma, _ = ScaledLogLinearTransform._tau_c(params, False)
        return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma, log_sigma + gamma * log_alpha)

    @staticmethod
    def g_from_y(y, params, x_side=True):
        # g(y) = d(L^{-1})/du = 1 / L'(y) = c*y / ((1-τ) y + τ c)
        tau, c, _, _ = ScaledLogLinearTransform._tau_c(params, x_side)
        return (c * y) / ((1.0 - tau) * y + tau * c)

    @staticmethod
    def O_delta_square(delta, params, x_side=True):
        _, _, _, kappa = ScaledLogLinearTransform._tau_c(params, x_side)
        return kappa * (delta ** 2)

    @staticmethod
    def get_sample_grad_coeff(i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params):
        # c는 샘플/그라드 계수에 영향 없음 (γ만 사용)
        _, _, gamma, _ = ScaledLogLinearTransform._tau_c(params, True)
        sample_coeff = torch.where(gamma >= 0, torch.exp(gamma * log_sigma_ratio[i]),
                                              torch.exp(-gamma * log_alpha_ratio[i]))
        grad_coeff   = torch.where(gamma >= 0, torch.exp(gamma * log_sigma[i+1]),
                                              torch.exp(-gamma * log_alpha[i+1]))
        return sample_coeff, grad_coeff
