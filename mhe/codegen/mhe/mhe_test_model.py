# ruff: noqa: I001
from pathlib import Path
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, vertcat, reshape, fmax

from mhe.mhe_base_model_interface import MheModel, MheCogeGenerator
from params import MheParams

class OscillatorModel(MheModel):
    def __init__(self, params: MheParams, name: str):
        self.name = name
        super().__init__(nx=2, nu=0, np=2)

    @property
    def bounds_param(self) -> list[tuple]:
        bounds = [(-20, 20.0), (-20, 20.0)]
        return bounds

    @property
    def bounds_state(self) -> list[tuple]:
        return [(-np.inf, np.inf), (-np.inf, np.inf)]

    @property
    def bounds_noise(self) -> list[tuple]:
        return [(-0.01, 0.01), (-0.01, 0.01)]


    def main_dynamics(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        omega, zeta = params[0], params[1]
        dx1 = x2
        dx2 = -omega**2 * x1 - 2*zeta*omega * x2
        return vertcat(dx1, dx2)
    
class MassSpringDamper(MheModel):
    """Simple 2‑state, 2‑parameter model: x1 = position, x2 = velocity.
       Parameters: k (stiffness), c (damping). Input: external force."""
    def __init__(self, mass=1.0):
        super().__init__(nx=2, nu=1, np=2)
        self.mass = mass
        self.name = "mass_spring_damper"


    @property
    def bounds_param(self) -> list[tuple]:
        # physical bounds: k >= 0, c >= 0
        return [(0.0, 100.0), (0.0, 20.0)]

    @property
    def bounds_state(self) -> list[tuple]:
        # no bounds on states (or you can add)
        return [(-np.inf, np.inf), (-np.inf, np.inf)]

    @property
    def bounds_noise(self) -> list[tuple]:
        # process noise bounds (optional)
        return [(-0.1, 0.1), (-0.1, 0.1)]

    def main_dynamics(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        k, c = params[0], params[1]
        u = input_signals[0] if input_signals.shape[0] > 0 else 0.0

        dx1 = x2
        dx2 = (u - k*x1 - c*x2) / self.mass
        return vertcat(dx1, dx2)
    
class OscilatorMheCodegenerator(MheCogeGenerator):
    def __init__(self, params: MheParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        #self.model: MheModel = OscillatorModel(params, model_name)
        self.model: MheModel = MassSpringDamper()

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
        ocp_mhe.parameter_values = np.zeros((p.size()[0] + nx + nx_augmented + n_theta * n_theta,))
        # Set cost type to EXTERNAL
        ocp_mhe.cost.cost_type = 'EXTERNAL'
        ocp_mhe.cost.cost_type_e = 'EXTERNAL'
        ocp_mhe.cost.cost_type_0 = 'EXTERNAL'

        ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'  # Faster than SQP for real-time
        ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp_mhe.solver_options.nlp_solver_max_iter = 35
        ocp_mhe.solver_options.hpipm_options = {
            # 'tol': 1e-6,
            # 'reg_epsilon': 1e-6,        # regularization on the Hessian
            # 'reg_epsilon_s': 1e-6,       # regularization on the slack variables (if any)
            # 'iter_max': 1000
                'scale': 1,  # включить автоматическое масштабирование
                'scale_ux': 1,  # масштабировать состояния и управления
        }
        ocp_mhe.solver_options.hessian_approx = 'EXACT'
        ocp_mhe.solver_options.print_level = 1
        ocp_mhe.solver_options.nlp_solver_stats_level = 1
        return ocp_mhe
