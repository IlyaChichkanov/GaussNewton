from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, atan2, fmax, reshape, vertcat
from mhe.mhe_base_model_interface import MheCogeGenerator, MheModel

from params import MheParams


class DynamicModel(MheModel):
    def __init__(self, params: MheParams, name: str):
        self.L = params.wheelbase  # колёсная база
        self.Gr = 1.0 / 13.322467  # передаточное число руля
        self.name = name

    @property
    def state_length(self):
        return 3

    @property
    def param_length(self):
        return 4

    @property
    def bounds_param(self) -> list[tuple]:
        return [(0.01, 100.0), (0.5, 1.5), (0.7, 1.5), (0.5, 1.5)]

    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf), (-1.0, 1.0), (-2.0, 2.0)]

    @property
    def bounds_noise(self) -> list[tuple]:
        return [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]

    def main_dynamics(self, state, params, input_signals):
        # Входы: vx, steering
        psi = state[0]
        vy = state[1]
        wz = state[2]

        C_bar = params[0]
        k = params[1]
        rho = params[2]
        mu = params[3]

        vx = input_signals[0]
        steering = input_signals[1]

        # Защита от малых скоростей
        vx_safe = fmax(vx, 0.1)

        # Углы увода
        alpha_f = atan2(vy + self.L * wz, vx_safe)
        alpha_r = atan2(vy, vx_safe)

        # Угол поворота управляемых колёс
        rwa = steering * self.Gr

        C_bar_scaled = 1.0 / C_bar * 50
        a0 = C_bar_scaled * (1 + rho)
        a1 = k * C_bar_scaled * (-1 + mu * rho)
        b0 = C_bar_scaled * rho * (1 + mu) / (mu * self.L)
        b1 = k * C_bar_scaled * rho * (1 + mu) / self.L

        # Производные
        dpsi = wz
        dvy = a0 * (rwa - alpha_f) + a1 * alpha_r - vx * wz
        dwz = b0 * (rwa - alpha_f) + b1 * alpha_r
        return vertcat(dpsi, dvy, dwz)


class DynamicModelUdersteer(MheModel):
    def __init__(self, params: MheParams, name: str):
        self.use_offset = params.use_offset
        self.wheelbase = params.wheelbase
        self.name = name

    @property
    def bounds_param(self) -> list[tuple]:
        bounds = [(0.0, 2.0), (np.deg2rad(-5), np.deg2rad(5), (0.05, 10000))]
        return bounds

    @property
    def param_length(self):
        return 3

    @property
    def state_length(self):
        return 2

    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf), (-2, 2)]

    @property
    def bounds_noise(self) -> list[tuple]:
        return [(-0.01, 0.01), (-0.01, 0.01)]

    # def main_dynamics(self, state, params, input_signals):
    #     psi, w = state[0], state[1]
    #     v_x, steering = input_signals[0], input_signals[1]
    #     GR_eps, offset = params[0], params[1]
    #     tau =  0.2#params[2]
    #     rwa = GR_eps * (steering) + offset
    #     dw = (v_x * np.tan(rwa) / (self.wheelbase) - w) / tau
    #     dheading = w
    #     return vertcat(dheading, dw)
    def main_dynamics(self, state, params, input_signals):
        psi, w = state[0], state[1]
        v_x, steering = input_signals[0], input_signals[1]
        GR_eps, offset = params[0], params[1]
        tau =  params[2]
        rwa = GR_eps * (steering) + offset
        dw = (v_x * np.tan(rwa) / (self.wheelbase) - w) * tau*0.01
        dheading = w
        return vertcat(dheading, dw)
 
class DynamicMheCodegenerator(MheCogeGenerator):
    def __init__(self, params: MheParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        self.model: MheModel = DynamicModelUdersteer(params, model_name)
        #self.model: MheModel = DynamicModel(params, model_name)
        
    def get_model(self) -> MheModel:
        return self.model

    def set_ocp_problem(self) -> AcadosOcp:
        ocp_mhe = AcadosOcp()
        model: MheModel = self.get_model()
        model_acados: AcadosModel = model.make_continious_acados_model()
        ocp_mhe.model = model_acados
        x = model_acados.x
        u = model_acados.u
        p = model_acados.p
        nx = model.state_length
        n_theta = model.param_length
        nx_augmented = nx + n_theta
        state, thetas = x[:nx], x[nx:]  # thetas длины 3
        y_ref = SX.sym("y_ref", nx)
        x_prior = SX.sym("x_prior", nx)
        param_prior = SX.sym("param_prior", n_theta)  # =3
        p_prior_weights = SX.sym("p_prior_weights", n_theta * n_theta)
        # Cost expressions (как у вас, но с учётом размерностей)
        P0 = reshape(p_prior_weights, n_theta, n_theta) * self.params.fim_scaler
        Q0 = self.params.state_prior_q0
        R = self.params.measurements_residual_r
        W = self.params.noise_peanlty_w
        stage_cost_expr = (state - y_ref).T @ R @ (state - y_ref) + u.T @ W @ u
        initial_cost_expr = (state - x_prior).T @ Q0 @ (state - x_prior) + (thetas - param_prior).T @ P0 @ (
            thetas - param_prior
        )
        ocp_mhe.model.cost_expr_ext_cost = stage_cost_expr
        ocp_mhe.model.cost_expr_ext_cost_e = 0  # Terminal cost
        ocp_mhe.model.cost_expr_ext_cost_0 = initial_cost_expr
        ocp_mhe.model.p = vertcat(p, y_ref, x_prior, param_prior, p_prior_weights)
        ocp_mhe.parameter_values = np.zeros((2 + nx + nx_augmented + n_theta * n_theta,))
        # Set cost type to EXTERNAL
        ocp_mhe.cost.cost_type = "EXTERNAL"
        ocp_mhe.cost.cost_type_e = "EXTERNAL"
        ocp_mhe.cost.cost_type_0 = "EXTERNAL"

        ocp_mhe.solver_options.nlp_solver_type = "SQP_RTI"  # Faster than SQP for real-time
        ocp_mhe.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp_mhe.solver_options.hessian_approx = "EXACT"

        # Настройки решателя (как у вас)
        ocp_mhe.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # ocp_mhe.solver_options.hpipm_options = {
        #     'tol': 1e-6,
        #     'reg_epsilon': 1e-6,        # regularization on the Hessian
        #     'reg_epsilon_s': 1e-6,       # regularization on the slack variables (if any)
        #     'iter_max': 1000
        # }
        # ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp_mhe.solver_options.nlp_solver_type = "SQP"
        ocp_mhe.solver_options.nlp_solver_max_iter = 15
        ocp_mhe.solver_options.print_level = 2
        ocp_mhe.solver_options.nlp_solver_stats_level = 1
        return ocp_mhe
