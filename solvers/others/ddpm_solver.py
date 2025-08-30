import torch
from ..solver import Solver

class DDPM_Solver(Solver):
    """
    DDPM ancestral sampler (noise_prediction / epsilon-pred).
    - Uses posterior variance (beta_tilde) and supports DDIM via `eta`.
    - Schedule-agnostic: only requires noise_schedule.{marginal_alpha, marginal_std}.
      Assumes VP-style: alpha(t)^2 + sigma(t)^2 = 1.
    """
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        eta: float = 1.0,                 # eta=1 -> DDPM, eta=0 -> DDIM
        algorithm_type="noise_prediction",  # fixed to epsilon-pred
        **kwargs
    ):
        super().__init__(noise_schedule, algorithm_type)
        assert algorithm_type == "noise_prediction", "DDPM_Solver is epsilon-pred only."
        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.eta = eta

    @torch.no_grad()
    def sample(self, x, model_fn, **kwargs):
        """
        Args:
            x:      (N, C, H, W) latent at the *noisiest* time (t ~ T).
            model_fn(x_t, t): returns epsilon prediction at (x_t, t).
        Returns:
            {'samples': x_Tto0, 'trajs': ..., 'timesteps': ..., 'alphas': ..., 'sigmas': ...}
        """
        self.set_model_fn(model_fn)

        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        device, dtype = x.device, x.dtype

        # Discretization: must be in *descending* time order (T -> 0)
        timesteps = self.get_time_steps(
            skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=self.steps, device=device, shift=self.flow_shift
        )
        # α(t), σ(t) at each discrete time (length = steps+1)
        alphas = torch.tensor(
            [self.noise_schedule.marginal_alpha(t) for t in timesteps],
            device=device, dtype=dtype
        )
        sigmas = torch.tensor(
            [self.noise_schedule.marginal_std(t) for t in timesteps],
            device=device, dtype=dtype
        )

        x_t = x
        for i in range(self.steps):
            # Current/next (note: `next` is the *less noisy* step)
            a_cur, s_cur = alphas[i],   sigmas[i]
            a_nxt, s_nxt = alphas[i+1], sigmas[i+1]

            # ᾱ = a^2 ; effective per-step beta, posterior variance
            abar_cur = a_cur * a_cur
            abar_nxt = a_nxt * a_nxt

            # Forward-process beta for this jump (nxt -> cur):
            #   abar_cur = abar_nxt * (1 - beta)  =>  beta = 1 - abar_cur/abar_nxt
            beta = 1.0 - (abar_cur / abar_nxt)
            beta = torch.clamp(beta, min=0.0, max=0.9999).to(dtype)

            # Posterior variance (beta_tilde): beta * (1 - abar_nxt) / (1 - abar_cur)
            denom = (1.0 - abar_cur).clamp(min=1e-12)
            posterior_var = beta * (1.0 - abar_nxt) / denom
            posterior_var = posterior_var.clamp(min=0.0)

            # Choose noise scale (η·sqrt(posterior_var)); η=1→DDPM, η=0→DDIM
            sigma = self.eta * torch.sqrt(posterior_var)

            # Predict epsilon and x0
            eps = self.model_fn(x_t, timesteps[i])  # epsilon-pred
            x0_pred = (x_t - s_cur * eps) / a_cur   # x0 = (x - σ ε)/α

            # Closed form ancestral/DDIM step:
            # x_{next} = sqrt(abar_next) * x0_pred + sqrt(1 - abar_next - sigma^2) * eps + sigma * z
            c0 = a_nxt
            c1 = torch.sqrt((1.0 - abar_nxt - sigma * sigma).clamp(min=0.0))
            if i == self.steps - 1:
                # Final step: no noise
                x_t = c0 * x0_pred + c1 * eps
            else:
                z = torch.randn_like(x_t)
                x_t = c0 * x0_pred + c1 * eps + sigma * z
            
        outputs = {'samples': x_t}
        return outputs
