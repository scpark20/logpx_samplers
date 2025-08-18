import numpy as np
import torch
import torch.nn.functional as F
from .transform import Transform

class LogAffineTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_max=None,
        tau_offset=0,
        tau_max=None,
        log_kappa_max=None,
        log_kappa_offset=0,
        tau_tol=1e-6,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.tau_max = tau_max
        self.tau_tol = tau_tol
        self.log_kappa_max = log_kappa_max
        self.log_kappa_offset = log_kappa_offset
        self.eps = eps
        
    def unpack(self, params):
        # params: [B, 5, n_channels, 1, 1]
        gamma   = params[:, 0]
        tau_x = params[:, 1] + self.tau_offset
        tau_e = params[:, 2] + self.tau_offset
        log_kappa_x = params[:, 3]
        log_kappa_e = params[:, 4]

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)
        
        if self.tau_max is None:
            # [0, +inf): softplus  (학습 초반엔 작고, 점차 커질 수 있음)
            tau_x = F.softplus(tau_x)
            tau_e = F.softplus(tau_e)
        else:
            # [0, tau_max): sigmoid로 상한도 함께 관리
            tau_x = self.tau_max * torch.sigmoid(tau_x)
            tau_e = self.tau_max * torch.sigmoid(tau_e)

        if self.log_kappa_max is not None:
            log_kappa_x = torch.tanh(log_kappa_x + self.log_kappa_offset) * self.log_kappa_max
            log_kappa_e = torch.tanh(log_kappa_e + self.log_kappa_offset) * self.log_kappa_max

        params = {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
                'kappa_x': torch.exp(log_kappa_x), 'kappa_e': torch.exp(log_kappa_e)}
        return params

    def L(self, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])[:, None]
        print('in L :', y, tau)
        u_lin    = y
        u_nonlin = torch.log1p(tau * y) / tau   # τ>0에서 안정
        return torch.where(tau.abs() < self.tau_tol, u_lin, u_nonlin)

    def get_integral(self, uc, un, us, tau, kappa, du):
        print('get integral :', uc, un, us, tau, kappa)
        
        def _log_erf_diff(a, b):
            return torch.log(torch.erfc(b)) + torch.log(1.0 - torch.exp(torch.log(torch.erfc(a)) - torch.log(torch.erfc(b))))

        # r_{i-j}
        r = (us - uc) / du
        a = tau*du
        log_prefactor = 0.5*np.log(np.pi) +\
                        torch.log(kappa) - np.log(2) + torch.log(du) +\
                        tau*un + 0.25*(kappa*a)**2 + a*(r-1)
        upper = r/kappa + 0.5*kappa*a
        lower = upper - 1./kappa
        result = torch.exp(log_prefactor + _log_erf_diff(upper, lower))
        print(result)
        return result

    def get_kernel(self, us, du, kappa):
        # (B, 1, 1)
        beta = (1/kappa * du)[:, None, None]
        diff = us.unsqueeze(2) - us.unsqueeze(1)
        K = torch.exp(-beta**2 * diff**2)
        return K

    def get_coefficients(self, uc, un, us, p, side='x'):
        # uc : u_i,   (B,)
        # un : u_i+1, (B,)
        # us : [u_i, u_i-1, ..., u_i-order+1], (B, order)

        kappa = p['kappa_x'] if side == 'x' else p['kappa_e']
        tau = p['tau_x'] if side == 'x' else p['tau_e']
        du = un - uc
        # (B, order)
        integral = self.get_integral(uc, un, us, tau, kappa, du)
        # (B, order, order)
        kernel = self.get_kernel(us, du, kappa)
        # (B, order)
        L, _ = torch.linalg.cholesky_ex(kernel)                  # (B,M,M)
        coefficients = torch.cholesky_solve(integral.unsqueeze(-1), L).squeeze(-1)         # (B,M)
        return coefficients

    