
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi import SX, vertcat

from mpc.mpc_base_model_interface import MpcCogeGenerator, MpcModel
from params import CarParams, MpcParams
from ocp_utils import quadform


class KinematicModel(MpcModel):
    def __init__(self, car_params: CarParams, model_name: str):
        self.gear_ratio = car_params.gear_ratio
        self.wheelbase = car_params.wheelbase
        self.model_name = model_name

    @property
    def state_length(self) -> int:
        return 2

    @property
    def param_length(self) -> int:
        return 3 

    def main_dynamics(self, state, rwa_cmd, imp_signals, params) -> SX:
        v, c = imp_signals[0], imp_signals[1]
        tau, psi = state[0], state[1]
        # offset = params[0]
        # GR, offset = params[0], params[1]
        #GR_rel = GR * self.gear_ratio
        dtau = np.sin(psi)
        denominator = 1 - c * tau * v
        GR_rel = 1
        offset = 0
        rwa = GR_rel * rwa_cmd  + offset
        dpsi = v * ca.tan(rwa) / self.wheelbase - v * c * np.cos(psi) / denominator
        dx = vertcat(dtau, dpsi)
        return dx


class KinematicMheCodegenerator(MpcCogeGenerator):
    def __init__(self, params: MpcParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        self.model: MpcModel = KinematicModel(params.car_params, model_name)
        self.params = params

    def get_model(self) -> MpcModel:
        return self.model

    def set_ocp_problem(self):
        # Create OCP object
        ocp = AcadosOcp()
        base_model = self.get_model()
        rwa_pos: int = base_model.state_length
        model = base_model.make_discrete_acados_model(self.params.ts, self.params.n_delay, self.params.use_ddu_control)
        # model = base_model.make_continuous_acados_model(model_name)
        ocp.model = model

        # Use EXTERNAL cost to avoid y_expr issues
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # Define cost function directly using states and controls
        x, u = model.x, model.u
        vx, c = model.p.elements()[:2]

        tau, psi = x[0], x[1]
        rwa = x[rwa_pos]

        cost_expr = quadform(np.diag([self.params.r_dist, self.params.r_ang]), vertcat(tau, psi))

        cost_expr_e = self.params.final_cost * \
                        quadform(np.diag([self.params.r_dist, self.params.r_ang]), vertcat(tau, psi))

        cost_expr += quadform(np.diag([self.params.r_u]), rwa - ca.atan(c * self.params.car_params.wheelbase))

        ocp.constraints.lbu = np.array([-self.params.car_params.du_max])
        ocp.constraints.ubu = np.array([self.params.car_params.du_max])

        ocp.model.con_h_expr = vertcat(rwa)
        ocp.constraints.lh = np.array([-self.params.car_params.u_max])  # Lower bounds
        ocp.constraints.uh = np.array([self.params.car_params.u_max])   # Upper bounds
        if (self.params.use_ddu_control):
            ocp.constraints.lbu = np.array([-self.params.car_params.ddu_max])
            ocp.constraints.ubu = np.array([self.params.car_params.ddu_max])
            drwa = x[rwa_pos + 1]
            ocp.model.con_h_expr = vertcat(ocp.model.con_h_expr, drwa)
            ocp.constraints.lh = np.hstack((ocp.constraints.lh, -self.params.car_params.du_max))
            ocp.constraints.uh = np.hstack((ocp.constraints.uh, self.params.car_params.du_max))
            cost_expr += quadform(np.diag([self.params.r_ddu]), u)
        else:
            drwa = u

        jerk = vx * vx * drwa / self.params.car_params.wheelbase
        jerk_comf_overshot = ca.fmax(0, ca.sqrt(jerk**2 + 1e-8) - self.params.jerk_comf)
        a_comf = vx * vx * rwa / self.params.car_params.wheelbase
        cost_expr += quadform(np.diag([self.params.r_du]), drwa)
        cost_expr += quadform(np.diag([self.params.r_jerk]), jerk_comf_overshot)
        use_hard_contrainth = False
        if (use_hard_contrainth):
            ocp.model.con_h_expr = vertcat(ocp.model.con_h_expr, jerk)
            ocp.constraints.lh = np.hstack((ocp.constraints.lh, -4.0))
            ocp.constraints.uh = np.hstack((ocp.constraints.uh, 4.0))
            ocp.model.con_h_expr = vertcat(rwa, jerk)
            ocp.constraints.lh = np.array([-self.params.car_params.u_max, -self.params.jerk_max])  # Lower bounds
            ocp.constraints.uh = np.array([self.params.car_params.u_max, self.params.jerk_max])   # Upper bounds
        else:
            jerk_hard = ca.fmax(0, ca.sqrt(jerk**2 + 1e-8) - self.params.jerk_max)
            cost_expr += quadform(np.diag([10000]), jerk_hard)
            a_comf_hard = ca.fmax(0, ca.sqrt(a_comf**2 + 1e-8) - self.params.a_comf_max)
            cost_expr += quadform(np.diag([10000]), a_comf_hard)

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        #ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.nlp_solver_max_iter = 10
        ocp.solver_options.eval_residual_at_max_iter = True
        ocp.model.cost_expr_ext_cost = cost_expr
        ocp.model.cost_expr_ext_cost_e = cost_expr_e
        ocp.solver_options.qp_solver_warm_start = True

        return ocp
