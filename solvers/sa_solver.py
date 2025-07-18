import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from .common import expand_dims

class SA_Solver:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
            correcting_x0_fn=None,
            correcting_xt_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995
    ):
        """
        Construct a SA-Solver
        The default value for algorithm_type is "data_prediction" and we recommend not to change it to
        "noise_prediction". For details, please see Appendix A.2.4 in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.predict_x0 = algorithm_type == "data_prediction"

        self.sigma_min = float(self.noise_schedule.edm_sigma(torch.tensor([1e-3])))
        self.sigma_max = float(self.noise_schedule.edm_sigma(torch.tensor([1])))

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """

        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, order, device, shift=1.0):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = lambda_T + torch.linspace(torch.tensor(0.).cpu().item(),
                                                     (lambda_0 - lambda_T).cpu().item() ** (1. / order), N + 1).pow(
                order).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time':
            t = torch.linspace(t_T ** (1. / order), t_0 ** (1. / order), N + 1).pow(order).to(device)
            return t
        elif skip_type == 'karras':
            sigma_min = max(0.002, self.sigma_min)
            sigma_max = min(80, self.sigma_max)
            sigma_steps = torch.linspace(sigma_max ** (1. / 7), sigma_min ** (1. / 7), N + 1).pow(7).to(device)
            return self.noise_schedule.edm_inverse_sigma(sigma_steps)
        elif skip_type == "time_uniform_flow":
            betas = torch.linspace(t_T, t_0, N + 1).to(device)
            sigmas = 1.0 - betas
            sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
            return sigmas
        else:
            raise ValueError(
                f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time' or 'karras'"
            )

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def get_coefficients_exponential_negative(self, order, interval_start, interval_end):
        """
        Calculate the integral of exp(-x) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For noise_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        if order == 0:
            return torch.exp(-interval_end) * (torch.exp(interval_end - interval_start) - 1)
        elif order == 1:
            return torch.exp(-interval_end) * (
                        (interval_start + 1) * torch.exp(interval_end - interval_start) - (interval_end + 1))
        elif order == 2:
            return torch.exp(-interval_end) * (
                        (interval_start ** 2 + 2 * interval_start + 2) * torch.exp(interval_end - interval_start) - (
                            interval_end ** 2 + 2 * interval_end + 2))
        elif order == 3:
            return torch.exp(-interval_end) * (
                        (interval_start ** 3 + 3 * interval_start ** 2 + 6 * interval_start + 6) * torch.exp(
                    interval_end - interval_start) - (interval_end ** 3 + 3 * interval_end ** 2 + 6 * interval_end + 6))

    def get_coefficients_exponential_positive(self, order, interval_start, interval_end, tau):
        """
        Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For data_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # after change of variable(cov)
        interval_end_cov = (1 + tau ** 2) * interval_end
        interval_start_cov = (1 + tau ** 2) * interval_start

        if order == 0:
            return torch.exp(interval_end_cov) * (1 - torch.exp(-(interval_end_cov - interval_start_cov))) / (
            (1 + tau ** 2))
        elif order == 1:
            return torch.exp(interval_end_cov) * ((interval_end_cov - 1) - (interval_start_cov - 1) * torch.exp(
                -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 2)
        elif order == 2:
            return torch.exp(interval_end_cov) * ((interval_end_cov ** 2 - 2 * interval_end_cov + 2) - (
                        interval_start_cov ** 2 - 2 * interval_start_cov + 2) * torch.exp(
                -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 3)
        elif order == 3:
            return torch.exp(interval_end_cov) * (
                        (interval_end_cov ** 3 - 3 * interval_end_cov ** 2 + 6 * interval_end_cov - 6) - (
                            interval_start_cov ** 3 - 3 * interval_start_cov ** 2 + 6 * interval_start_cov - 6) * torch.exp(
                    -(interval_end_cov - interval_start_cov))) / ((1 + tau ** 2) ** 4)


    def lagrange_polynomial_coefficient(self, order, lambda_list):
        """
        Calculate the coefficient of lagrange polynomial
        For lagrange interpolation
        """
        assert order in [0, 1, 2, 3]
        assert order == len(lambda_list) - 1
        if order == 0:
            return [[1]]
        elif order == 1:
            return [[1 / (lambda_list[0] - lambda_list[1]), -lambda_list[1] / (lambda_list[0] - lambda_list[1])],
                    [1 / (lambda_list[1] - lambda_list[0]), -lambda_list[0] / (lambda_list[1] - lambda_list[0])]]
        elif order == 2:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1])
            return [[1 / denominator1,
                     (-lambda_list[1] - lambda_list[2]) / denominator1,
                     lambda_list[1] * lambda_list[2] / denominator1],

                    [1 / denominator2,
                     (-lambda_list[0] - lambda_list[2]) / denominator2,
                     lambda_list[0] * lambda_list[2] / denominator2],

                    [1 / denominator3,
                     (-lambda_list[0] - lambda_list[1]) / denominator3,
                     lambda_list[0] * lambda_list[1] / denominator3]
                    ]
        elif order == 3:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2]) * (
                        lambda_list[0] - lambda_list[3])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2]) * (
                        lambda_list[1] - lambda_list[3])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1]) * (
                        lambda_list[2] - lambda_list[3])
            denominator4 = (lambda_list[3] - lambda_list[0]) * (lambda_list[3] - lambda_list[1]) * (
                        lambda_list[3] - lambda_list[2])
            return [[1 / denominator1,
                     (-lambda_list[1] - lambda_list[2] - lambda_list[3]) / denominator1,
                     (lambda_list[1] * lambda_list[2] + lambda_list[1] * lambda_list[3] + lambda_list[2] * lambda_list[
                         3]) / denominator1,
                     (-lambda_list[1] * lambda_list[2] * lambda_list[3]) / denominator1],

                    [1 / denominator2,
                     (-lambda_list[0] - lambda_list[2] - lambda_list[3]) / denominator2,
                     (lambda_list[0] * lambda_list[2] + lambda_list[0] * lambda_list[3] + lambda_list[2] * lambda_list[
                         3]) / denominator2,
                     (-lambda_list[0] * lambda_list[2] * lambda_list[3]) / denominator2],

                    [1 / denominator3,
                     (-lambda_list[0] - lambda_list[1] - lambda_list[3]) / denominator3,
                     (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[3] + lambda_list[1] * lambda_list[
                         3]) / denominator3,
                     (-lambda_list[0] * lambda_list[1] * lambda_list[3]) / denominator3],

                    [1 / denominator4,
                     (-lambda_list[0] - lambda_list[1] - lambda_list[2]) / denominator4,
                     (lambda_list[0] * lambda_list[1] + lambda_list[0] * lambda_list[2] + lambda_list[1] * lambda_list[
                         2]) / denominator4,
                     (-lambda_list[0] * lambda_list[1] * lambda_list[2]) / denominator4]

                    ]

    def get_coefficients_fn(self, order, interval_start, interval_end, lambda_list, tau):
        """
        Calculate the coefficient of gradients.
        """
        assert order in [1, 2, 3, 4]
        assert order == len(lambda_list), 'the length of lambda list must be equal to the order'
        coefficients = []
        lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        for i in range(order):
            coefficient = sum(
                lagrange_coefficient[i][j]
                * self.get_coefficients_exponential_positive(
                    order - 1 - j, interval_start, interval_end, tau
                )
                if self.predict_x0
                else lagrange_coefficient[i][j]
                * self.get_coefficients_exponential_negative(
                    order - 1 - j, interval_start, interval_end
                )
                for j in range(order)
            )
            coefficients.append(coefficient)
        assert len(coefficients) == order, 'the length of coefficients does not match the order'
        return coefficients

    def adams_bashforth_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """
        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = [ns.marginal_lambda(t_prev_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_moulton_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        t_list = t_prev_list + [t]
        lambda_list = [ns.marginal_lambda(t_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_bashforth_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, with the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = [ns.marginal_lambda(t_prev_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)


        if self.predict_x0:
            if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to unipc. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                gradient_coefficients[0] += 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2)) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(
                    t_prev_list[-2]))
                gradient_coefficients[1] -= 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2)) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(
                    t_prev_list[-2]))

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part



        return x_t

    def adams_moulton_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        t_list = t_prev_list + [t]
        lambda_list = [ns.marginal_lambda(t_list[-(i + 1)]) for i in range(order)]
        gradient_coefficients = self.get_coefficients_fn(order, ns.marginal_lambda(t_prev_list[-1]), lambda_t,
                                                         lambda_list, tau)

        if self.predict_x0:
            if order == 2:  ## if order = 2 we do a modification that does not influence the convergence order similar to UniPC. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                gradient_coefficients[0] += 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2 * h))
                gradient_coefficients[1] -= 1.0 * torch.exp((1 + tau ** 2) * lambda_t) * (
                            h / 2 - (h * (1 + tau ** 2) - 1 + torch.exp((1 + tau ** 2) * (-h))) / (
                                (1 + tau ** 2) ** 2 * h))

        for i in range(order):
            if self.predict_x0:
                gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(- tau ** 2 * lambda_t) * gradient_coefficients[
                    i] * model_prev_list[-(i + 1)]
            else:
                gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def sample_few_steps(self, x, tau, steps=5, t_start=None, t_end=None, skip_type='time', skip_order=1,
                         predictor_order=3, corrector_order=4, pc_mode='PEC', return_intermediate=False, shift=1.0
                         ):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.
        """

        skip_first_step = False
        skip_final_step = True
        lower_order_final = True
        denoise_to_zero = False

        assert pc_mode in ['PEC', 'PECE'], 'Predictor-corrector mode only supports PEC and PECE'
        
        # print("noise_schedule.total_N : ", self.noise_schedule.total_N)

        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= max(predictor_order, corrector_order - 1)
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, order=skip_order,
                                            device=device, shift=shift)
            print("timesteps : ", timesteps)

            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            noise = torch.randn_like(x)
            t_prev_list = [t]
            # do not evaluate if skip_first_step
            if skip_first_step:
                if self.predict_x0:
                    alpha_t = self.noise_schedule.marginal_alpha(t)
                    sigma_t = self.noise_schedule.marginal_std(t)
                    model_prev_list = [(1 - sigma_t) / alpha_t * x]
                else:
                    model_prev_list = [x]
            else:
                model_prev_list = [self.model_fn(x, t)]

            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)

            # determine the first several values
            for step in tqdm(range(1, max(predictor_order, corrector_order - 1))):

                t = timesteps[step]
                predictor_order_used = min(predictor_order, step)
                corrector_order_used = min(corrector_order, step + 1)
                noise = torch.randn_like(x)
                # predictor step
                x_p = self.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                            noise=noise, t=t)
                
                # print(f"x_p {step}: ", x_p)

                # evaluation step
                model_x = self.model_fn(x_p, t)

                # update model_list
                model_prev_list.append(model_x)
                # corrector step
                if corrector_order > 0:
                    x = self.adams_moulton_update_few_steps(order=corrector_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                            noise=noise, t=t)
                else:
                    x = x_p

                # evaluation step if correction and mode = pece
                if corrector_order > 0 and pc_mode == 'PECE':
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)

            for step in tqdm(range(max(predictor_order, corrector_order - 1), steps + 1)):
                if lower_order_final:
                    predictor_order_used = min(predictor_order, steps - step + 1)
                    corrector_order_used = min(corrector_order, steps - step + 2)

                else:
                    predictor_order_used = predictor_order
                    corrector_order_used = corrector_order
                t = timesteps[step]
                noise = torch.randn_like(x)

                # predictor step
                if skip_final_step and step == steps and not denoise_to_zero:
                    x_p = self.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=0,
                                                                model_prev_list=model_prev_list,
                                                                t_prev_list=t_prev_list, noise=noise, t=t)
                else:
                    x_p = self.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=tau(t),
                                                                model_prev_list=model_prev_list,
                                                                t_prev_list=t_prev_list, noise=noise, t=t)

                # if step == steps :
                #     print('t : ', t)
                #     print(f"x_p {step}: ", x_p)

                # evaluation step
                # do not evaluate if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_x = self.model_fn(x_p, t)

                # update model_list
                # do not update if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_prev_list.append(model_x)

                # corrector step
                # do not correct if skip_final_step and step = steps
                if corrector_order > 0 and (not skip_final_step or step < steps):
                    x = self.adams_moulton_update_few_steps(order=corrector_order_used, x=x, tau=tau(t),
                                                            model_prev_list=model_prev_list,
                                                            t_prev_list=t_prev_list, noise=noise, t=t)
                else:
                    x = x_p

                # evaluation step if mode = pece and step != steps
                if corrector_order > 0 and (pc_mode == 'PECE' and step < steps):
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)
                del model_prev_list[0]

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        return (x, intermediates) if return_intermediate else x

    def sample_more_steps(self, x, tau, steps=20, t_start=None, t_end=None, skip_type='time', skip_order=1,
                          predictor_order=3, corrector_order=4, pc_mode='PEC', return_intermediate=False, shift=1.0
                          ):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.
        """

        skip_first_step = False
        skip_final_step = False
        lower_order_final = True
        denoise_to_zero = True

        assert pc_mode in ['PEC', 'PECE'], 'Predictor-corrector mode only supports PEC and PECE'
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= max(predictor_order, corrector_order - 1)
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, order=skip_order,
                                            device=device, shift=shift)
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            noise = torch.randn_like(x)
            t_prev_list = [t]
            # do not evaluate if skip_first_step
            if skip_first_step:
                if self.predict_x0:
                    alpha_t = self.noise_schedule.marginal_alpha(t)
                    sigma_t = self.noise_schedule.marginal_std(t)
                    model_prev_list = [(1 - sigma_t) / alpha_t * x]
                else:
                    model_prev_list = [x]
            else:
                model_prev_list = [self.model_fn(x, t)]

            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)

            # determine the first several values
            for step in tqdm(range(1, max(predictor_order, corrector_order - 1))):

                t = timesteps[step]
                predictor_order_used = min(predictor_order, step)
                corrector_order_used = min(corrector_order, step + 1)
                noise = torch.randn_like(x)
                # predictor step
                x_p = self.adams_bashforth_update(order=predictor_order_used, x=x, tau=tau(t),
                                                  model_prev_list=model_prev_list, t_prev_list=t_prev_list, noise=noise,
                                                  t=t)
                # evaluation step
                model_x = self.model_fn(x_p, t)

                # update model_list
                model_prev_list.append(model_x)
                # corrector step
                if corrector_order > 0:
                    x = self.adams_moulton_update(order=corrector_order_used, x=x, tau=tau(t),
                                                  model_prev_list=model_prev_list, t_prev_list=t_prev_list, noise=noise,
                                                  t=t)
                else:
                    x = x_p

                # evaluation step if mode = pece
                if corrector_order > 0 and pc_mode == 'PECE':
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)

            for step in tqdm(range(max(predictor_order, corrector_order - 1), steps + 1)):
                if lower_order_final:
                    predictor_order_used = min(predictor_order, steps - step + 1)
                    corrector_order_used = min(corrector_order, steps - step + 2)

                else:
                    predictor_order_used = predictor_order
                    corrector_order_used = corrector_order
                t = timesteps[step]
                noise = torch.randn_like(x)

                # predictor step
                if skip_final_step and step == steps and not denoise_to_zero:
                    x_p = self.adams_bashforth_update(order=predictor_order_used, x=x, tau=0,
                                                      model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                      noise=noise, t=t)
                else:
                    x_p = self.adams_bashforth_update(order=predictor_order_used, x=x, tau=tau(t),
                                                      model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                      noise=noise, t=t)

                # evaluation step
                # do not evaluate if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_x = self.model_fn(x_p, t)

                # update model_list
                # do not update if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_prev_list.append(model_x)

                # corrector step
                # do not correct if skip_final_step and step = steps
                if corrector_order > 0:
                    if not skip_final_step or step < steps:
                        x = self.adams_moulton_update(order=corrector_order_used, x=x, tau=tau(t),
                                                      model_prev_list=model_prev_list, t_prev_list=t_prev_list,
                                                      noise=noise, t=t)
                    else:
                        x = x_p
                else:
                    x = x_p

                # evaluation step if mode = pece and step != steps
                if corrector_order > 0 and (pc_mode == 'PECE' and step < steps):
                    model_x = self.model_fn(x, t)
                    del model_prev_list[-1]
                    model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)
                del model_prev_list[0]

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x

    def sample(self, x, tau, steps=20, mode='few_steps', t_start=None, t_end=None, skip_type='time', skip_order=1, predictor_order=3,
               corrector_order=4, pc_mode='PEC', return_intermediate=False, flow_shift=1.0
               ):
        """
        For the PC-mode, please refer to the wiki page 
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.

        'few_steps' mode is recommended. The differences between 'few_steps' and 'more_steps' are as below:
        1) 'few_steps' do not correct at final step and do not denoise to zero, while 'more_steps' do these two.
        Thus the NFEs for 'few_steps' = steps, NFEs for 'more_steps' = steps + 2
        For most of the experiments and tasks, we find these two operations do not have much help to sample quality.
        2) 'few_steps' use a rescaling trick as in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        We find it will slightly improve the sample quality especially in few steps.
        """
        assert mode in ['few_steps', 'more_steps'], "mode must be either 'few_steps' or 'more_steps'"
        if mode == 'few_steps':
            return self.sample_few_steps(x=x, tau=tau, steps=steps, t_start=t_start, t_end=t_end, skip_type=skip_type,
                                         skip_order=skip_order, predictor_order=predictor_order,
                                         corrector_order=corrector_order, pc_mode=pc_mode,
                                         return_intermediate=return_intermediate, shift=flow_shift)
        else:
            return self.sample_more_steps(x=x, tau=tau, steps=steps, t_start=t_start, t_end=t_end, skip_type=skip_type,
                                          skip_order=skip_order, predictor_order=predictor_order,
                                          corrector_order=corrector_order, pc_mode=pc_mode,
                                          return_intermediate=return_intermediate, shift=flow_shift)

