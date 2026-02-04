
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, bilin, fmax, vertcat

from mpc.mpc_base_model_interface import (
    MpcModel,
    discrete_bicycle_model_ddu,
    discrete_bicycle_model_du,
)
from params import CarParams, MpcParams
from identification.codegen.ocp_utils import generate_header, is_discrete, quadform


class DynamicModel(MpcModel):
    def __init__(self, car_params: CarParams):
        tau = SX.sym('tau')
        psi = SX.sym('psi')
        vy = SX.sym('vy')
        wz = SX.sym('wz')
        self.x = vertcat(tau, psi, vy, wz)
        self.params = car_params
        self.use_sliping = True

    def get_state(self) -> SX:
        return self.x

    def continuous_dynamics(self, x, rwa, p) -> SX:
        tau, psi, vy, wz = x[0], x[1], x[2], x[3]
        vx, c = p[0], p[1]
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


def set_ocp_problem(params: MpcParams, model_name: str, generated_folder: Path):
    # Create OCP object
    ocp = AcadosOcp()
    base_model = DynamicModel(params.car_params)
    rwa_pos: int = base_model.x.shape[0]
    state_length: int = rwa_pos + 1
    if (params.use_ddu_control):
        state_length += 1
        model = discrete_bicycle_model_ddu(params, model_name, base_model)
    else:
        model = discrete_bicycle_model_du(params, model_name, base_model)

    ocp.parameter_values = np.zeros(model.p.shape[0])
    ocp.model = model

    # Dimensions
    Tf = params.mpc_horizont * params.ts  # prediction horizon [s]
    N = params.mpc_horizont    # number of shooting nodes

    ocp.solver_options.N_horizon = N

    # Use EXTERNAL cost to avoid y_expr issues
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # Define cost function directly using states and controls
    x, u = model.x, model.u
    vx, c = model.p.elements()
    tau, psi, vy, wz = x[0], x[1], x[2], x[3]
    rwa = x[rwa_pos]
    # Weighting matrices
    Q = np.diag([params.r_dist])  # state weights
    cost_expr = quadform(Q, tau)
    cost_expr_e = params.final_cost * bilin(Q, tau)

    ocp.constraints.idxbu = np.array([0])  # Constrain Δu
    if (params.use_ddu_control):
        ocp.constraints.lbu = np.array([-params.car_params.ddu_max])
        ocp.constraints.ubu = np.array([params.car_params.ddu_max])
        drwa = x[rwa_pos + 1]
        ocp.model.con_h_expr = vertcat(rwa, drwa)
        ocp.constraints.lh = np.array([-params.car_params.u_max, -params.car_params.du_max])  # Lower bounds
        ocp.constraints.uh = np.array([params.car_params.u_max, params.car_params.du_max])   # Upper bounds
        cost_expr += quadform(np.diag([params.r_ddu * params.ts]), u)
    else:
        ocp.constraints.lbu = np.array([-params.car_params.du_max])
        ocp.constraints.ubu = np.array([params.car_params.du_max])
        drwa = u
        ocp.model.con_h_expr = vertcat(rwa)
        ocp.constraints.lh = np.array([-params.car_params.u_max])  # Lower bounds
        ocp.constraints.uh = np.array([params.car_params.u_max])   # Upper bounds

    jerk = vx * vx * drwa / params.car_params.wheelbase
    jerk_violation = ca.fmax(0, ca.sqrt(jerk**2 + 1e-8) - params.jerk_comf)
    cost_expr += quadform(np.diag([params.r_du * params.ts]), drwa)
    cost_expr += quadform(np.diag([params.r_jerk]), jerk_violation)
    cost_expr += bilin(np.diag([params.r_w]), wz - c * vx)

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
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = True

    ocp.solver_options.code_export_directory = str(generated_folder)
    ocp.code_export_directory = str(generated_folder)
    ocp.solver_options.json_file = str(generated_folder / 'acados_ocp_nlp2.json')
    solver = AcadosOcpSolver(ocp, json_file=ocp.solver_options.json_file, build=True, generate=True)
    definitions = {
        "STATE_LENGTH": state_length,
        "RWA_POS": rwa_pos,
        "CURV_CONTROL": 0,
        "USE_DDU_CONTROL": int(params.use_ddu_control),
    }
    generate_header(generated_folder / model_name / "include/mpc_settings.h",\
                     definitions, f'MPC_SETTINGS_{model_name}')
