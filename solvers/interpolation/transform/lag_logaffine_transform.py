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
        
        tau_x = torch.sigmoid(tau_x + self.tau_offset)
        tau_e = torch.sigmoid(tau_e + self.tau_offset)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e}

    def L(self, y, p, side='x'):
        # y : (B, L)
        # tau : (B,)
        tau = p['tau_x'] if side=='x' else p['tau_e']
        tau = tau[:, None]

        # u : (B, L)
        u = torch.log1p(tau*y) / tau
        return u

    # -------------------- ∫ e^{τu} u^k du (k=0..K-1), 배치 --------------------
    @staticmethod
    def _exp_poly_integrals(uc, un, tau, K, tau_tol=1e-6):
        """
        I_k = ∫_{uc}^{un} e^{τu} u^k du,  k=0..K-1
        τ≈0는 다항 적분, 그 외에는 재귀식으로 계산. 둘 다 구해 where로 전환(미분 가능).
        """
        B = uc.shape[0]
        uc = uc.view(B, 1); un = un.view(B, 1); tau = tau.view(B, 1)

        # # (1) τ≈0 다항 적분
        # ks = torch.arange(1, K + 1, device=uc.device, dtype=uc.dtype).unsqueeze(0)  # (1,K)
        # I_poly = (un.pow(ks) - uc.pow(ks)) / ks                                     # (B,K)

        # (2) τ!=0 재귀
        exp_un = torch.exp(tau * un); exp_uc = torch.exp(tau * uc)
        J_un = exp_un / tau
        J_uc = exp_uc / tau
        rows = [J_un - J_uc]  # I_0
        for k in range(1, K):
            k_over_tau = (float(k)) / tau
            J_un = (un.pow(k) * exp_un) / tau - k_over_tau * J_un
            J_uc = (uc.pow(k) * exp_uc) / tau - k_over_tau * J_uc
            rows.append(J_un - J_uc)
        I_rec = torch.cat(rows, dim=1)                                              # (B,K)

        #mask = (tau.abs() < tau_tol).expand_as(I_poly)
        #return torch.where(mask, I_poly, I_rec)                                     # (B,K)
        return I_rec


    # -------------------- 라그랑주 적분 벡터 ℓ = [I_0..I_{M-1}] --------------------
    def get_integral(self, uc, un, us, tau):
        """
        반환 ℓ: (B, M),  I_k = ∫_{uc}^{un} e^{τu} u^k du
        us는 길이 M(차수)만 알려주는데 사용.
        """
        _, M = us.shape
        ell = self._exp_poly_integrals(uc, un, tau, K=M, tau_tol=1e-6)  # (B,M)
        return ell
        
    # -------------------- 반더몬드 행렬 M (B,M,M), M_{j,k}=u_j^k --------------------
    @staticmethod
    def vandermonde(us):
        """
        us: (B, M)  ->  M: (B, M, M),  열 k가 u^k (k=0..M-1, increasing)
        torch.vander(1D) 대신 배치 벡터화 버전.
        """
        B, M = us.shape
        k = torch.arange(M, device=us.device, dtype=us.dtype)  # (M,)
        return us.unsqueeze(-1).pow(k)                          # (B,M,M)

    # -------------------- 가중치 W = (M^{-1})^T ℓ  (즉, M^T W = ℓ) --------------------
    def get_coefficients(self, uc, un, us, p, side='x'):
        """
        predictor: us=[u_i, u_{i-1}, ...]
        corrector: us=[u_{i+1}, u_i, ...]  (총 노드가 predictor보다 1개 많음)
        반환 W: (B, M) — 노드값 가중치 (이걸로 xs_win, es_win과 곱-합)
        """
        tau = p['tau_x'] if side == 'x' else p['tau_e']
        ell = self.get_integral(uc, un, us, tau)          # (B,M)
        M   = self.vandermonde(us)                               # (B,M,M)
        A   = M.transpose(1, 2)                                   # M^T, (B,M,M)
        rhs = ell.unsqueeze(-1)                                   # (B,M,1)
        W = torch.linalg.lstsq(A, rhs).solution.squeeze(-1)
        #W = torch.linalg.solve(A, rhs).squeeze(-1)           # (B,M)
        return W