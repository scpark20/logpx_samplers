import torch
import torch.nn.functional as F
from .transform import Transform

def _sigmoid_range(x, lo, hi):
    # smooth map: R -> (lo, hi)
    return lo + (hi - lo) * torch.sigmoid(x)

class ShiftedLogPowerTransform(Transform):
    """
    L = (1-τ) q log(y+c) + τ * ((y+c)^p - c^p)
    g = (y+c) / ((1-τ) q + τ p (y+c)^p)
    - 강한 곡률에서 g ↓ (p>1 권장)
    - y≈0에서도 c>0로 특이점 회피
    """
    def __init__(self,
                 gamma_max=3.0, kappa_max=3.0,
                 p_bounds=(0.8, 3.0),      # p in (0.8, 3.0) → p>1이면 강한 덤핑
                 q_bounds=(0.1, 4.0),      # q in (0.1, 4.0)
                 c_bounds=(1e-3, 5.0),     # c in (1e-3, 5.0)
                 tau_min=1e-3,             # τ in [tau_min, 1-tau_min]
                 gamma_push=True, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max  = gamma_max
        self.kappa_max  = kappa_max
        self.p_lo, self.p_hi = p_bounds
        self.q_lo, self.q_hi = q_bounds
        self.c_lo, self.c_hi = c_bounds
        self.tau_min = tau_min
        self.eps = eps

    def unpack(self, params):
        # params: [B, 11] = [gamma, tau_x, tau_e, p_x, p_e, q_x, q_e, c_x, c_e, kappa_x, kappa_e]
        (gamma, tau_x, tau_e,
         p_x, p_e, q_x, q_e, c_x, c_e, kappa_x, kappa_e) = [params[:, i] for i in range(11)]

        # smooth bounded maps (clamp 대신)
        tau_x = _sigmoid_range(tau_x, self.tau_min, 1.0 - self.tau_min)
        tau_e = _sigmoid_range(tau_e, self.tau_min, 1.0 - self.tau_min)
        p_x   = _sigmoid_range(p_x,   self.p_lo, self.p_hi)
        p_e   = _sigmoid_range(p_e,   self.p_lo, self.p_hi)
        q_x   = _sigmoid_range(q_x,   self.q_lo, self.q_hi)
        q_e   = _sigmoid_range(q_e,   self.q_lo, self.q_hi)
        c_x   = _sigmoid_range(c_x,   self.c_lo, self.c_hi)
        c_e   = _sigmoid_range(c_e,   self.c_lo, self.c_hi)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {
            'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
            'p_x': p_x, 'p_e': p_e, 'q_x': q_x, 'q_e': q_e,
            'c_x': c_x, 'c_e': c_e, 'kappa_x': kappa_x, 'kappa_e': kappa_e
        }

    # ----- core formulas -----
    def L(self, log_y, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p   = params['p_x']   if side=='x' else params['p_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        c   = params['c_x']   if side=='x' else params['c_e']

        # log(y+c) = logaddexp(log y, log c)  (수치안정)
        logc = torch.log(c)
        log_y_shift = torch.logaddexp(log_y, logc)

        # (y+c)^p = exp(p * log(y+c))
        pow_shift = torch.exp(p * log_y_shift)

        return (1.0 - tau) * q * log_y_shift + tau * (pow_shift - c ** p)

    def g(self, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p   = params['p_x']   if side=='x' else params['p_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        c   = params['c_x']   if side=='x' else params['c_e']

        ys = y + c
        # 분모는 (1-τ)q + τ p (y+c)^p  >  min{ (1-τ_min)q_lo, τ_min p_lo c_lo^p } > 0
        denom = (1.0 - tau) * q + tau * p * torch.exp(p * torch.log(ys))
        return ys / denom

    # ----- Δu를 로그 도메인에서 안정 계산 (권장) -----
    def delta_u_from_logs(self, log_yc, log_yn, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p   = params['p_x']   if side=='x' else params['p_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        c   = params['c_x']   if side=='x' else params['c_e']

        logc = torch.log(c)
        # log(y+c)
        log_sc = torch.logaddexp(log_yc, logc)
        log_sn = torch.logaddexp(log_yn, logc)

        # Δ log(y+c)
        dlog = log_sn - log_sc

        # Δ (y+c)^p = exp(m) * (expm1(a_n-m) - expm1(a_c-m))
        a_n = p * log_sn
        a_c = p * log_sc
        m = torch.maximum(a_n, a_c)
        delta_pow = torch.exp(m) * (torch.expm1(a_n - m) - torch.expm1(a_c - m))

        return (1.0 - tau) * q * dlog + tau * delta_pow
