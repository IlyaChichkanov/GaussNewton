# ruff: noqa: I001
from pathlib import Path
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, vertcat, reshape, fmax

from mhe.mhe_base_model_interface import MheModel, MheCogeGenerator
from params import MheParams


class KinematicModelOffset(MheModel):
    def __init__(self, params: MheParams, name: str):
        super().__init__(nx=1, nu=2, np=1)
        self.wheelbase = params.wheelbase
        self.name = name


    @property
    def bounds_param(self) -> list[tuple]:
        #return [(-5,5)]
        return [(np.deg2rad(-5), np.deg2rad(5))]

    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf)]

    @property
    def bounds_noise(self) -> list[tuple]:
        return [(-0.01, 0.01)]

    def main_dynamics(self, state, params, input_signals):
        v, steering = input_signals[0], input_signals[1]
        offset = params[0]
        GR = 1 / 13.322467
        rwa = GR * (steering) + offset
        dheading = v * np.tan(rwa) / self.wheelbase
        return vertcat(dheading)


class KinematicModel(MheModel):
    def __init__(self, params: MheParams, name: str):
        param_length = 1
        if(params.use_offset):
            param_length = 2
        super().__init__(nx=1, nu=2, np=param_length)
        self.use_offset = params.use_offset
        self.wheelbase = params.wheelbase
        self.name = name

    @property
    def bounds_param(self) -> list[tuple]:
        bounds = [(0, 20.0), (np.deg2rad(-3), np.deg2rad(3))]
        if (not self.use_offset):
            return bounds[0:1]
        return bounds
    
    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf)]

    @property
    def bounds_noise(self) -> list[tuple]:
        return [(-0.01, 0.01)]

    def main_dynamics(self, state, params, input_signals):
        v, steering = input_signals[0], input_signals[1]
        GR_eps = params[0]
        offset = 0
        if (self.use_offset):
            offset = params[1]
        rwa = GR_eps * (steering) + offset
        dheading = v * np.tan(rwa) / self.wheelbase
        return vertcat(dheading)

class KinematicModelWithDelay(MheModel):
    """
    Кинематическая модель велосипеда с задержкой актуатора.
    Задержка аппроксимируется фильтром Паде 1‑го порядка.
    Состояние: [psi, x_d] (курс и внутренняя переменная фильтра)
    Параметры: [GR, offset, tau_d] (если use_offset=True) или [GR, tau_d]
    """
    def __init__(self, params, name: str):
        self.L = params.wheelbase
        self.name = name
        super().__init__(nx=1, nu=2, np=2)

    @property
    def bounds_param(self) -> list[tuple]:
        bounds = [
            (0.02, 0.2),                    # GR
            (np.deg2rad(-5), np.deg2rad(5)),# offset
            #(0.01, 0.5)                     # tau_d [сек]
        ]
        return bounds

    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf), (-10.0, 10.0)]  # psi неограничен, x_d – разумный предел

    @property
    def bounds_noise(self) -> list[tuple]:
        # Шум для psi и x_d (оба могут быть малы)
        return [(-0.01, 0.01), (-0.01, 0.01)]

    def main_dynamics(self, state, params, input_signals):
        # state: [psi, x_d]
        psi = state[0]
        delta_actual = state[1]

        # Извлечение параметров
        if self.use_offset:
            GR = params[0]
            offset = params[1]
            #tau_d = params[2]
        else:
            GR = params[0]
            offset = 0.0
            #tau_d = params[1]

        tau_d = 0.2
        v = input_signals[0]
        steering = input_signals[1]

        # Угол поворота колёс по команде
        delta_cmd = GR * steering + offset

        # Защита от нулевой и отрицательной задержки
        tau_d_safe = fmax(tau_d, 1e-6)

        # # Динамика фильтра Паде
        # dx_d = (2.0 / tau_d_safe) * (delta_cmd - x_d)
        dx_d = (1.0 / tau_d_safe) * (delta_cmd - delta_actual)
        # # Реальный угол колёс
        # delta_actual = 2.0 * delta_cmd - x_d

        # Кинематика курса
        v_safe = fmax(v, 0.1)
        dpsi = v_safe * np.tan(delta_actual) / self.L

        return vertcat(dpsi, dx_d)
    
class KinematicMheCodegenerator(MheCogeGenerator):
    def __init__(self, params: MheParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        if (params.use_only_offset):
            self.model: MheModel = KinematicModelOffset(params, model_name)
        else:
            self.model: MheModel = KinematicModel(params, model_name)
            #self.model: MheModel = KinematicModelWithDelay(params, model_name)

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
        y_ref = SX.sym('y_ref', nx)
        x_prior = SX.sym('x_prior', nx)
        param_prior = SX.sym('param_prior', n_theta)  # =3
        p_prior_weights = SX.sym('p_prior_weights', n_theta * n_theta)
        # Cost expressions (как у вас, но с учётом размерностей)
        P0 = reshape(p_prior_weights, n_theta, n_theta) * self.params.fim_scaler
        Q0 = self.params.state_prior_q0
        R = self.params.measurements_residual_r
        W = self.params.noise_peanlty_w
        stage_cost_expr = (state - y_ref).T @ R @ (state - y_ref) + u.T @ W @ u
        initial_cost_expr = (state - x_prior).T @ Q0 @ (state - x_prior) +\
              (thetas - param_prior).T @ P0 @ (thetas - param_prior)
        ocp_mhe.model.cost_expr_ext_cost = stage_cost_expr
        ocp_mhe.model.cost_expr_ext_cost_e = 0  # Terminal cost
        ocp_mhe.model.cost_expr_ext_cost_0 = initial_cost_expr
        ocp_mhe.model.p = vertcat(p, y_ref, x_prior, param_prior, p_prior_weights)
        ocp_mhe.parameter_values = np.zeros((2 + nx + nx_augmented + n_theta * n_theta,))
        # Set cost type to EXTERNAL
        ocp_mhe.cost.cost_type = 'EXTERNAL'
        ocp_mhe.cost.cost_type_e = 'EXTERNAL'
        ocp_mhe.cost.cost_type_0 = 'EXTERNAL'

        ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'  # Faster than SQP for real-time
        ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp_mhe.solver_options.nlp_solver_max_iter = 15
        ocp_mhe.solver_options.hpipm_options = {
            # 'tol': 1e-6,
            # 'reg_epsilon': 1e-6,        # regularization on the Hessian
            # 'reg_epsilon_s': 1e-6,       # regularization on the slack variables (if any)
            # 'iter_max': 1000
                'scale': 1,  # включить автоматическое масштабирование
                'scale_ux': 1,  # масштабировать состояния и управления
        }
        ocp_mhe.solver_options.hessian_approx = 'EXACT'
        ocp_mhe.solver_options.print_level = 0
        ocp_mhe.solver_options.nlp_solver_stats_level = 1
        return ocp_mhe
