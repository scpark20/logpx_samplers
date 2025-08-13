import torch
import torch.nn.functional as F
from .transform import Transform

def _sigmoid_range(x, lo, hi):
    # smooth map: R -> (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)

class AdaptiveLogPowerTransform(Transform):
    """
    L = (1-τ) q log y + τ y^p
    g = y / ((1-τ) q + τ p y^p)
    - p는 [p_lo, p_hi]로 제한 (p>1이면 큰 y에서 강한 덤핑)
    - q = rho * p 로 소프트 타이 (단조/스케일 안정)
    - τ ∈ [tau_min, 1 - tau_min] 로 분모>0 보장
    - Δu는 로그-안정식으로 제공 (expm1)
    """
    def __init__(self,
                 gamma_max=3.0, kappa_max=3.0,
                 p_bounds=(1.0, 3.0),     # p in (1,3): 강곡률 대응
                 rho_bounds=(0.5, 2.0),   # q = rho * p, rho in (0.5,2)
                 tau_min=1e-3,            # τ in [tau_min, 1-tau_min]
                 gamma_push=True, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max  = gamma_max
        self.kappa_max  = kappa_max
        self.p_lo, self.p_hi     = p_bounds
        self.rho_lo, self.rho_hi = rho_bounds
        self.tau_min = tau_min
        self.eps = eps

    def unpack(self, params):
        """
        params: [B,  nine] = [gamma, tau_x, tau_e, p_x, p_e, rho_x, rho_e, kappa_x, kappa_e]
        """
        (gamma, tau_x, tau_e,
         p_x, p_e, rho_x, rho_e,
         kappa_x, kappa_e) = [params[:, i] for i in range(9)]

        # τ를 (tau_min, 1 - tau_min)로
        tau_x = _sigmoid_range(tau_x, self.tau_min, 1.0 - self.tau_min)
        tau_e = _sigmoid_range(tau_e, self.tau_min, 1.0 - self.tau_min)

        # p를 [p_lo, p_hi], rho를 [rho_lo, rho_hi]
        p_x   = _sigmoid_range(p_x,   self.p_lo,   self.p_hi)
        p_e   = _sigmoid_range(p_e,   self.p_lo,   self.p_hi)
        rho_x = _sigmoid_range(rho_x, self.rho_lo, self.rho_hi)
        rho_e = _sigmoid_range(rho_e, self.rho_lo, self.rho_hi)

        # q = rho * p  (소프트 타이)
        q_x = rho_x * p_x
        q_e = rho_e * p_e

        # gamma/kappa 제한
        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        # ±1 근처 도메인 특이점 회피(필요시)
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {
            'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
            'p_x': p_x, 'p_e': p_e, 'q_x': q_x, 'q_e': q_e,
            'kappa_x': kappa_x, 'kappa_e': kappa_e
        }

    # -------- core formulas --------
    def L(self, log_y, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p   = params['p_x']   if side=='x' else params['p_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        # y^p = exp(p * log y) 로 안정 계산
        pow_term = torch.exp(p * log_y)
        return (1.0 - tau) * q * log_y + tau * pow_term

    def g(self, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p   = params['p_x']   if side=='x' else params['p_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        # y^p = exp(p * log y) (y>0 가정: 호출부에서 y=exp(log_y) 생성)
        y_pow = torch.exp(p * torch.log(y))
        denom = (1.0 - tau) * q + tau * p * y_pow
        return y / denom