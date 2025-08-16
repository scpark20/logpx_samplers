import math
import torch
from .transform import Transform

class LogAffineTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_max=None,
        tau_offset=0,
        log_kappa_max=None,
        log_kappa_offset=0,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.log_kappa_max = log_kappa_max
        self.log_kappa_offset = log_kappa_offset
        self.eps = eps
        
    def unpack(self, params):
        # params: [B, 5]
        gamma   = params[:, 0]
        tau_x   = params[:, 1]
        tau_e   = params[:, 2]
        log_kappa_x = params[:, 3]
        log_kappa_e = params[:, 4]
        
        tau_x = torch.sigmoid(tau_x + self.tau_offset)
        tau_e = torch.sigmoid(tau_e + self.tau_offset)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.log_kappa_max is not None:
            log_kappa_x = torch.tanh(log_kappa_x + self.log_kappa_offset) * self.log_kappa_max
            log_kappa_e = torch.tanh(log_kappa_e + self.log_kappa_offset) * self.log_kappa_max
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
                'kappa_x': torch.exp(log_kappa_x), 'kappa_e': torch.exp(log_kappa_e)}

    def L(self, y, p, side='x'):
        # y : (B, L)
        # tau : (B,)
        tau = p['tau_x'] if side=='x' else p['tau_e']
        tau = tau[:, None]

        # u : (B, L)
        u = torch.log1p(tau*y) / tau
        return u

    def _cdf_diff(self, z_hi: torch.Tensor, z_lo: torch.Tensor) -> torch.Tensor:
        """
        Φ(z_hi) - Φ(z_lo): 표준정규 CDF 차이.
        기본은 ndtr 차이로 충분하고, 극단 꼬리에서는 logcdf/logsf로 분기해도 됨.
        """
        return torch.special.ndtr(z_hi) - torch.special.ndtr(z_lo)

    def get_integral(self, uc, un, us, tau, kappa):
        """
        ∫_{uc}^{un} exp(τ u) * exp(-((u - c)/s)^2) du  의 배치 폐형식.
        uc, un: (B,)
        us:     (B, M)   -- 중심 c_j 들
        tau:    (B,)
        kappa:  (B,)     -- 폭 s (u-도메인 단위)
        return: (B, M)   -- 각 중심별 적분 ℓ_j
        """
        B, M = us.shape
        device, dtype = us.device, us.dtype

        s   = kappa.view(B, 1)  # (B,1)
        sig = s / math.sqrt(2.0)                 # (B,1)
        tau = tau.view(B, 1)                     # (B,1)

        c = us                                   # (B,M)
        m = c + tau * (sig ** 2)                 # (B,M)

        z_hi = (un.view(B,1) - m) / sig          # (B,M)
        z_lo = (uc.view(B,1) - m) / sig          # (B,M)

        # prefactor: √(2π) σ
        pref = math.sqrt(2.0*math.pi) * sig      # (B,1)
        expo = torch.exp(tau * c + 0.5 * (tau * sig) ** 2)  # (B,M)

        cdfd = self._cdf_diff(z_hi, z_lo)        # (B,M)
        ell  = pref * expo * cdfd                # (B,M)  (broadcast OK)
        return ell

    def get_kernel(self, us, kappa):
        """
        RBF Gram K_{jk} = exp(-((u_j - u_k)/s)^2)
        us:    (B, M)
        kappa: (B,)
        return: (B, M, M)
        """
        B, M = us.shape
        s = kappa.view(B, 1, 1)    # (B,1,1)
        du = us.unsqueeze(2) - us.unsqueeze(1)      # (B,M,M)
        K  = torch.exp(- (du / s) ** 2)             # (B,M,M)

        # 작은 릿지 추가(수치안정)
        lam = 1e-6 * K.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True).unsqueeze(-1)  # (B,1,1)
        I = torch.eye(M, device=us.device, dtype=us.dtype).unsqueeze(0)                   # (1,M,M)
        return K #+ lam * I

    def get_coefficients(self, uc, un, us, p, side='x'):
        # uc : u_i,   (B,)
        # un : u_i+1, (B,)
        # us : [u_i, u_i-1, ..., u_i-order+1], (B, order)

        kappa = p['kappa_x'] if side == 'x' else p['kappa_e']
        tau = p['tau_x'] if side == 'x' else p['tau_e']
        # (B, order)
        integral = self.get_integral(uc, un, us, tau, kappa)
        # (B, order, order)
        kernel = self.get_kernel(us, kappa)
        # (B, order)
        L, _ = torch.linalg.cholesky_ex(kernel)                  # (B,M,M)
        coefficients = torch.cholesky_solve(integral.unsqueeze(-1), L).squeeze(-1)         # (B,M)
        return coefficients

    