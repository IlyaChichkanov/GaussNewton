
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

from mpc.mpc_base_model_interface import MpcModel, discrete_bicycle_model_ddu, discrete_bicycle_model_du
from params import CarParams, MpcParams
from identification.codegen.ocp_utils import generate_header, is_discrete, quadform


class KinematicModel(MpcModel):
    def __init__(self, car_params: CarParams):
        tau = SX.sym('tau')
        psi = SX.sym('psi')
        self.x = vertcat(tau, psi)

    def get_state(self) -> SX:
        return self.x

    def continuous_dynamics(self, x, rwa, p) -> SX:
        v, c = p[0], p[1]
        tau, psi = x[0], x[1]
        dtau = np.sin(psi)
        denominator = 1 - c * tau * v
        dpsi = v * rwa - v * c * np.cos(psi) / denominator
        dx = vertcat(dtau, dpsi)
        return dx


def set_ocp_problem(params: MpcParams, model_name: str, generated_folder: Path):
    # Create OCP object
    ocp = AcadosOcp()
    base_model = KinematicModel(params.car_params)
    rwa_pos: int = base_model.x.shape[0]
    state_length: int = rwa_pos + 1
    if (params.use_ddu_control):
        state_length += 1
        model = discrete_bicycle_model_ddu(params, model_name, base_model)
    else:
        model = discrete_bicycle_model_du(params, model_name, base_model)

    # model = continuous_bicycle_model_du(params, model_name, base_model)
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

    tau, psi = x[0], x[1]
    rwa = x[rwa_pos]

    cost_expr = quadform(np.diag([params.r_dist, params.r_ang]), vertcat(tau, psi))
    cost_expr_e = params.final_cost * quadform(np.diag([params.r_dist, params.r_ang]), vertcat(tau, psi))

    ocp.constraints.idxbu = np.array([0])  # Constrain Δu
    if (params.use_ddu_control):
        ocp.constraints.lbu = np.array([-params.car_params.ddu_max])
        ocp.constraints.ubu = np.array([params.car_params.ddu_max])
        drwa = x[rwa_pos + 1]
        ocp.model.con_h_expr = vertcat(rwa, drwa)
        ocp.constraints.lh = np.array([-params.car_params.u_max, -params.car_params.du_max])  # Lower bounds
        ocp.constraints.uh = np.array([params.car_params.u_max, params.car_params.du_max])   # Upper bounds
        cost_expr += quadform(np.diag([params.r_ddu]), u)
    else:
        ocp.constraints.lbu = np.array([-params.car_params.du_max])
        ocp.constraints.ubu = np.array([params.car_params.du_max])
        drwa = u
        ocp.model.con_h_expr = vertcat(rwa)
        ocp.constraints.lh = np.array([-params.car_params.u_max])  # Lower bounds
        ocp.constraints.uh = np.array([params.car_params.u_max])   # Upper bounds

    jerk = vx * vx * drwa
    jerk_violation = ca.fmax(0, ca.sqrt(jerk**2 + 1e-8) - params.jerk_comf)
    cost_expr += quadform(np.diag([params.r_du]), drwa)
    cost_expr += quadform(np.diag([params.r_u]), rwa - c)
    cost_expr += quadform(np.diag([params.r_jerk]), jerk_violation)

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
        "CURV_CONTROL": 1,
        "USE_DDU_CONTROL": int(params.use_ddu_control),
    }
    generate_header(generated_folder / model_name / "include/mpc_settings.h", \
                    definitions, f'MPC_SETTINGS_{model_name}')
