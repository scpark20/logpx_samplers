import torch
import torch.nn.functional as F
from .transform import Transform

class LogAffineTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_max=None,
        tau_offset=0,
        tau_max=None,
        kappa_max=None,
        eps=1e-2,
        tau_tol=1e-6,
        except_gamma=False,
        except_tau=False,
        except_kappa=False,
        shared_taukappa=False,
        ):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.tau_max = tau_max
        self.kappa_max = kappa_max
        self.eps = eps
        self.tau_tol = tau_tol

        self.except_gamma = except_gamma
        self.except_tau = except_tau
        self.except_kappa = except_kappa
        self.shared_taukappa = shared_taukappa
        
    def unpack(self, params):
        # params: [B, ?] or [B, ?, C] (gamma, tau_x, tau_e가 앞 3개라고 가정)
        gamma = params[:, 0]
        tau_x = params[:, 1] + self.tau_offset
        tau_e = params[:, 2] + self.tau_offset
        k_x = params[:, 3]
        k_e = params[:, 4]

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.tau_max is None:
            # [0, +inf): softplus  (학습 초반엔 작고, 점차 커질 수 있음)
            tau_x = F.softplus(tau_x)
            tau_e = F.softplus(tau_e)
        else:
            # [0, tau_max): sigmoid로 상한도 함께 관리
            tau_x = self.tau_max * torch.sigmoid(tau_x)
            tau_e = self.tau_max * torch.sigmoid(tau_e)
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)
        if self.kappa_max:
            k_x = torch.tanh(k_x) * self.kappa_max
            k_e = torch.tanh(k_e) * self.kappa_max

        if self.except_gamma:
            gamma = torch.ones_like(gamma)
        if self.except_tau:
            tau_x = torch.ones_like(tau_x)
            tau_e = torch.ones_like(tau_e)
        if self.except_kappa:
            k_x = torch.zeros_like(k_x)
            k_e = torch.zeros_like(k_e)
        if self.shared_taukappa:
            tau_e = tau_x
            k_e = k_x
            
        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
        'kappa_x': k_x, 'kappa_e': k_e}

    def L(self, log_y, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])
        u_lin    = y
        u_nonlin = torch.log1p(tau * y) / tau   # τ>0에서 안정
        return torch.where(tau.abs() < self.tau_tol, u_lin, u_nonlin)
