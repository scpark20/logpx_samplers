import math
import torch
import torch.nn.functional as F
from .transform import Transform

import math
import torch
import torch.nn.functional as F
from .ch_transform import Transform

class LogAffineTransform(Transform):
    def __init__(self,
        gamma_push=False, gamma_max=None,
        tau_offset=0, tau_max=None,
        log_kappa_max=None, log_kappa_offset=0,
        tau_tol=1e-6, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.tau_max = tau_max
        self.log_kappa_max = log_kappa_max
        self.log_kappa_offset = log_kappa_offset
        self.tau_tol = tau_tol
        self.eps = eps

    def unpack(self, params):  # params: (B, 5, C)
        gamma       = params[:, 0]                    # (B,C)
        tau_x       = params[:, 1] + self.tau_offset  # (B,C)
        tau_e       = params[:, 2] + self.tau_offset  # (B,C)
        log_kappa_x = params[:, 3]                    # (B,C)
        log_kappa_e = params[:, 4]                    # (B,C)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        if self.tau_max is None:
            tau_x = F.softplus(tau_x); tau_e = F.softplus(tau_e)
        else:
            tau_x = self.tau_max * torch.sigmoid(tau_x)
            tau_e = self.tau_max * torch.sigmoid(tau_e)

        if self.log_kappa_max is not None:
            log_kappa_x = torch.tanh(log_kappa_x + self.log_kappa_offset) * self.log_kappa_max
            log_kappa_e = torch.tanh(log_kappa_e + self.log_kappa_offset) * self.log_kappa_max

        return {
            'gamma':   gamma,                   # (B,C)
            'tau_x':   tau_x,                   # (B,C)
            'tau_e':   tau_e,                   # (B,C)
            'kappa_x': torch.exp(log_kappa_x),  # (B,C)
            'kappa_e': torch.exp(log_kappa_e),  # (B,C)
        }

    @staticmethod
    def _bc_to_yshape(bc, y):
        # bc:(B,C) -> (B,C,1,...,1)  (y와 동일 rank로 브로드캐스트)
        return bc.view(bc.shape[0], bc.shape[1], *([1] * (y.ndim - 2)))

    def L(self, y, p, side='x'):  # y: (B,C,...)  / tau: (B,C)
        tau = p['tau_x'] if side == 'x' else p['tau_e']
        tau_b = self._bc_to_yshape(tau, y)
        mask = (tau_b * y).abs() < self.tau_tol
        tau_hat = torch.where(mask, torch.ones_like(tau_b), tau_b)
        u_nonlin = torch.log1p(tau_b * y) / tau_hat
        return torch.where(mask, y, u_nonlin)

    @staticmethod
    def _cdf_diff(z_hi, z_lo):
        return torch.special.ndtr(z_hi) - torch.special.ndtr(z_lo)

    # ---- 적분/커널/계수: 모두 (B,C,...) 고정 ----
    def get_integral(self, uc, un, us, tau, kappa):  # uc,un:(B,C), us:(B,C,M), tau,kappa:(B,C)
        B, C, M = us.shape
        s   = kappa.unsqueeze(-1)                 # (B,C,1)
        sig = s / math.sqrt(2.0)                  # (B,C,1)
        tau = tau.unsqueeze(-1)                   # (B,C,1)

        m    = us + tau * (sig ** 2)              # (B,C,M)
        z_hi = (un.unsqueeze(-1) - m) / sig
        z_lo = (uc.unsqueeze(-1) - m) / sig

        pref = math.sqrt(2.0*math.pi) * sig       # (B,C,1)
        expo = torch.exp(tau * us + 0.5 * (tau * sig) ** 2)  # (B,C,M)
        return pref * expo * self._cdf_diff(z_hi, z_lo)      # (B,C,M)

    def get_kernel(self, us, kappa):  # us:(B,C,M), kappa:(B,C) -> (B,C,M,M)
        s  = kappa.unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)
        du = us.unsqueeze(-1) - us.unsqueeze(-2)  # (B,C,M,M)
        return torch.exp(- (du / s) ** 2)

    def get_coefficients(self, uc, un, us, p, side='x'):  # -> (B,C,M)
        tau   = p['tau_x'] if side == 'x' else p['tau_e']   # (B,C)
        kappa = p['kappa_x'] if side == 'x' else p['kappa_e']  # (B,C)

        ell = self.get_integral(uc, un, us, tau, kappa)     # (B,C,M)
        K   = self.get_kernel(us, kappa)                    # (B,C,M,M)

        B, C, M, _ = K.shape
        Kf   = K.reshape(B*C, M, M).float()
        ellf = ell.reshape(B*C, M, 1).float()

        L, info = torch.linalg.cholesky_ex(Kf)
        if (info > 0).any():
            I  = torch.eye(M, device=K.device).unsqueeze(0)
            dm = Kf.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True).unsqueeze(-1)
            L  = torch.linalg.cholesky(Kf + (1e-6 * dm + 1e-12) * I)

        coef = torch.cholesky_solve(ellf, L).reshape(B, C, M)
        return coef.to(K.dtype)
