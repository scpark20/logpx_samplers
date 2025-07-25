# https://github.com/wl-zhao/DC-Solver/blob/main/stable-diffusion/ldm/models/diffusion/dc_solver/dc_solver.py

import torch
import numpy as np


class DCSolver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0=True,
        thresholding=False,
        max_val=1.0,
        variant="bh1",
        dc_order=2,
    ):
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.dc_order = dc_order

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s).to(s.device)
            ),
            dims,
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(
            t
        ), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(
                torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims
            )
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device, shift=1.0):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(
                lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1
            ).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1)
                .pow(t_order)
                .to(device)
            )
            return t
        elif skip_type == "time_uniform_flow":
            betas = torch.linspace(t_T, t_0, N + 1).to(device)
            sigmas = 1.0 - betas
            sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
            return sigmas
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self, steps, order, skip_type, t_T, t_0, device
    ):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(
        self, x, model_prev_list, t_prev_list, t, order, **kwargs
    ):
        if len(t.shape) == 0:
            t = t.view(-1)
        if "bh" in self.variant:
            return self.multistep_uni_pc_bh_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )
        else:
            raise NotImplementedError()

    def multistep_uni_pc_bh_update(
        self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True
    ):
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.dim()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0
        ), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0] if self.predict_x0 else h[0]
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * h_phi_1, dims) * model_prev_0
            )

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * (
                    corr_res + rhos_c[-1] * D1_t
                )
        else:
            x_t_ = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * (
                    corr_res + rhos_c[-1] * D1_t
                )
        return x_t, model_t

    def dynamic_compensation(self, model_prev_list, t_prev_list, ratio):
        t_prev, t = t_prev_list[-2], t_prev_list[-1]

        t_ = t * ratio + t_prev * (1 - ratio)
        model_t_dc = torch.zeros_like(model_prev_list[-1])
        for i in range(self.dc_order + 1):
            term = model_prev_list[-(i + 1)]
            for j in range(self.dc_order + 1):
                if i != j:
                    coeff = (t_ - t_prev_list[-(j + 1)]) / (
                        t_prev_list[-(i + 1)] - t_prev_list[-(j + 1)]
                    )
                    term = term * coeff[0]
            model_t_dc = model_t_dc + term
        return model_t_dc

    def find_optim_ratio(
        self, vec_t, x, model_prev_list, t_prev_list, dc_ratio_initial, corrector_kwargs
    ):
        if len(model_prev_list) < self.dc_order + 1:  # skip the warming up
            return dc_ratio_initial
        scalar_t = vec_t[0].detach().item()
        ratio_param = torch.nn.Parameter(
            torch.tensor([dc_ratio_initial], device=vec_t.device), requires_grad=True
        )

        index = np.where(self.ref_ts > scalar_t)[0].max()
        # estimate
        x_t_gt = self.ref_xs[index]

        model_bak = model_prev_list[-1].clone()

        def closure(ratio_param):
            model_t_dc = self.dynamic_compensation(
                model_prev_list, t_prev_list, ratio=ratio_param
            )
            if model_t_dc is not None:
                model_prev_list[-1] = model_t_dc
            x_t_pred, _ = self.multistep_uni_pc_update(
                x, model_prev_list, t_prev_list, vec_t, **corrector_kwargs
            )
            loss = torch.nn.functional.mse_loss(x_t_pred, x_t_gt)
            # rewind
            model_prev_list[-1] = model_bak
            return loss

        optimizer = torch.optim.AdamW([ratio_param], lr=0.1)
        for iter_ in range(40):
            optimizer.zero_grad()
            loss = closure(ratio_param)
            loss.backward()
            optimizer.step()
            print(f"iter [{iter_}]", ratio_param.item(), loss.item())
        torch.cuda.empty_cache()
        return ratio_param.data.item()

    def search_dc(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
        corrector=False,
        dc_ratio_initial=1.0,
        flow_shift=1.0,
    ):
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        dc_ratios = []
        if method == "multistep":
            assert steps >= order
            timesteps = self.get_time_steps(
                skip_type=skip_type,
                t_T=t_T,
                t_0=t_0,
                N=steps,
                device=device,
                shift=flow_shift,
            )
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.

                for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    corrector_kwargs = dict(
                        order=init_order,
                        use_corrector=True,
                    )
                    with torch.enable_grad():
                        dc_ratio = self.find_optim_ratio(
                            vec_t,
                            x,
                            model_prev_list,
                            t_prev_list,
                            dc_ratio_initial,
                            corrector_kwargs,
                        )
                        dc_ratios.append(dc_ratio)

                    if dc_ratio != 1.0:
                        model_prev_list[-1] = self.dynamic_compensation(
                            model_prev_list, t_prev_list, ratio=dc_ratio
                        )
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        init_order,
                        use_corrector=True,
                    )
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)

                for step in range(order, steps + 1):
                    vec_t = timesteps[step].expand(x.shape[0])
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        use_corrector = False
                    else:
                        use_corrector = True

                    corrector_kwargs = dict(
                        order=step_order,
                        use_corrector=use_corrector,
                    )
                    with torch.enable_grad():
                        dc_ratio = self.find_optim_ratio(
                            vec_t,
                            x,
                            model_prev_list,
                            t_prev_list,
                            dc_ratio_initial,
                            corrector_kwargs,
                        )
                        dc_ratios.append(dc_ratio)

                    if dc_ratio != 1.0:
                        model_prev_list[-1] = self.dynamic_compensation(
                            model_prev_list, t_prev_list, ratio=dc_ratio
                        )

                    x, model_x = self.multistep_uni_pc_update(
                        x, model_prev_list, t_prev_list, vec_t, **corrector_kwargs
                    )
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        if model_x is None:
                            model_x = self.model_fn(x, vec_t)
                        model_prev_list[-1] = model_x
        else:
            raise NotImplementedError()
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return dc_ratios

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
        corrector=False,
        dc_ratios=None,
        flow_shift=1.0,
    ):
        if dc_ratios is None:
            dc_ratios = [1.0] * steps
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        if method == "multistep":
            assert steps >= order
            timesteps = self.get_time_steps(
                skip_type=skip_type,
                t_T=t_T,
                t_0=t_0,
                N=steps,
                device=device,
                shift=flow_shift,
            )
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for init_order in range(1, order):
                    dc_ratio = dc_ratios.pop(0)
                    if dc_ratio != 1.0:
                        model_prev_list[-1] = self.dynamic_compensation(
                            model_prev_list, t_prev_list, ratio=dc_ratio
                        )
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        init_order,
                        use_corrector=True,
                    )
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list.append(model_x)
                    t_prev_list.append(vec_t)
                for step in range(order, steps + 1):
                    dc_ratio = dc_ratios.pop(0)
                    if dc_ratio != 1.0:
                        model_prev_list[-1] = self.dynamic_compensation(
                            model_prev_list, t_prev_list, ratio=dc_ratio
                        )
                    vec_t = timesteps[step].expand(x.shape[0])
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        use_corrector = False
                    else:
                        use_corrector = True
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        step_order,
                        use_corrector=use_corrector,
                    )
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        if model_x is None:
                            model_x = self.model_fn(x, vec_t)
                        model_prev_list[-1] = model_x
        else:
            raise NotImplementedError()
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1
    )
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)
    ).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
