
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi import SX, bilin, fmax, vertcat

from mpc.mpc_base_model_interface import MpcCogeGenerator, MpcModel
from params import CarParams, MpcParams
from ocp_utils import is_discrete, quadform


class DynamicModel(MpcModel):
    def __init__(self, car_params: CarParams, model_name: str):
        self.params = car_params
        self.use_sliping = True
        self.model_name = model_name

    @property
    def state_length(self) -> int:
        return 4

    @property
    def param_length(self) -> int:
        return 0

    def main_dynamics(self, state, rwa, inp_signals, params) -> SX:
        tau, psi, vy, wz = state[0], state[1], state[2], state[3]
        vx, c = inp_signals[0], inp_signals[1]
        if (self.use_sliping):
            dtau = np.sin(psi) + vy / vx * np.cos(psi)
            dpsi = wz - c * (vx * np.cos(psi) - vy * np.sin(psi)) / (1 - c * tau * vx)
        else:
            dtau = np.sin(psi)
            dpsi = wz - c * (vx * np.cos(psi)) / (1 - c * tau * vx)

        v_eps = 0.05
        alfa_f = np.arctan2(vy + self.params.wheelbase * wz, fmax(vx, v_eps)) - rwa
        alfa_r = np.arctan2(vy, fmax(vx, v_eps))

        l_f = 1.44
        l_r = self.params.wheelbase - l_f
        m = 1580
        J = 6293
        Cf = 8.72e4
        Cr = 1.1e5

        Ff = -alfa_f * Cf
        Fr = -alfa_r * Cr

        wz_dot = (l_f * Ff - l_r * Fr) / J
        vy_dot = (Ff + Fr) / m - vx * wz + wz_dot * l_r

        return vertcat(dtau, dpsi, vy_dot, wz_dot)


class DynamicMheCodegenerator(MpcCogeGenerator):
    def __init__(self, params: MpcParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        self.model: MpcModel = DynamicModel(params.car_params, model_name)
        self.params = params

    def get_model(self) -> MpcModel:
        return self.model

    def set_ocp_problem(self):
        # Create OCP object
        ocp = AcadosOcp()
        base_model = self.get_model()
        rwa_pos: int = base_model.state_length
        model = base_model.make_discrete_acados_model(\
            self.params.ts, self.params.n_delay, self.params.use_ddu_control)
        # model = base_model.make_continuous_acados_model(model_name)
        ocp.model = model

        # Use EXTERNAL cost to avoid y_expr issues
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # Define cost function directly using states and controls
        x, u = model.x, model.u
        vx, c = model.p.elements()[:2]
        tau, psi, vy, wz = x[0], x[1], x[2], x[3]
        rwa = x[rwa_pos]
        # Weighting matrices
        Q = np.diag([self.params.r_dist])  # state weights
        cost_expr = quadform(Q, tau)
        cost_expr_e = self.params.final_cost * bilin(Q, tau)

        ocp.constraints.idxbu = np.array([0])  # Constrain Δu
        if (self.params.use_ddu_control):
            ocp.constraints.lbu = np.array([-self.params.car_params.ddu_max])
            ocp.constraints.ubu = np.array([self.params.car_params.ddu_max])
            drwa = x[rwa_pos + 1]
            ocp.model.con_h_expr = vertcat(rwa, drwa)
            ocp.constraints.lh = \
                np.array([-self.params.car_params.u_max, -self.params.car_params.du_max])  # Lower bounds
            ocp.constraints.uh = \
                np.array([self.params.car_params.u_max, self.params.car_params.du_max])   # Upper bounds
            cost_expr += quadform(np.diag([self.params.r_ddu * self.params.ts]), u)
        else:
            ocp.constraints.lbu = np.array([-self.params.car_params.du_max])
            ocp.constraints.ubu = np.array([self.params.car_params.du_max])
            drwa = u
            ocp.model.con_h_expr = vertcat(rwa)
            ocp.constraints.lh = np.array([-self.params.car_params.u_max])  # Lower bounds
            ocp.constraints.uh = np.array([self.params.car_params.u_max])   # Upper bounds

        jerk = vx * vx * drwa / self.params.car_params.wheelbase
        jerk_violation = ca.fmax(0, ca.sqrt(jerk**2 + 1e-8) - self.params.jerk_comf)
        cost_expr += quadform(np.diag([self.params.r_du * self.params.ts]), drwa)
        cost_expr += quadform(np.diag([self.params.r_jerk]), jerk_violation)
        cost_expr += bilin(np.diag([self.params.r_w]), wz - c * vx)

        ocp.constraints.x0 = np.zeros(len(model.x.elements()))

        discrete: bool = is_discrete(model)

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        if (discrete):
            ocp.solver_options.integrator_type = 'DISCRETE'
        else:
            ocp.solver_options.integrator_type = 'ERK'

        ocp.model.cost_expr_ext_cost = cost_expr
        ocp.model.cost_expr_ext_cost_e = cost_expr_e

        return ocp
