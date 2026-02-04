
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

from mhe.mhe_base_model_interface import MheModel, make_discrete_mhe_model, make_continious_mhe_model
from params import MheParams
from ocp_utils import generate_header, is_discrete


class KinematicModel(MheModel):
    def __init__(self, params: MheParams):
        heading = SX.sym('heading')
        GR = SX.sym('GR')
        x = vertcat(heading, GR)
        self.use_offset = params.use_offset
        if (params.use_offset):
            offset = SX.sym('offset')
            x = vertcat(x, offset)
            param_length = 2
        else:
            offset = 0
            param_length = 1

        self.x = x
        self.state_length = 1
        self.param_length = param_length
        self.wheelbase = params.wheelbase

    def get_state(self) -> SX:
        return self.x

    def continuous_dynamics(self, x, noise, p) -> SX:
        heading, params = x[0], x[1:]
        v, steering = p[0], p[1]
        GR = params[0]
        offset = 0
        if (self.use_offset):
            offset = params[1]
        dheading = v * np.tan(GR * (steering) + offset) / self.wheelbase + noise
        dx = vertcat(dheading, SX(np.zeros(params.shape)))
        return dx


def set_ocp_problem(params: MheParams, model_name: str, generated_folder: Path):
    ocp_mhe = AcadosOcp()
    model = KinematicModel(params)
    model_mhe: AcadosModel = make_continious_mhe_model(model_name, params, model)

    ocp_mhe.model = model_mhe
    nx_augmented = model_mhe.x.rows() - params.delay
    n_inp_param = model_mhe.p.rows()
    nx = model_mhe.state_length

    # Create symbolic variables for external cost
    x = ocp_mhe.model.x
    u = ocp_mhe.model.u
    p = ocp_mhe.model.p
    state, thetas = x[:nx], x[nx:]

    # Define measurement reference as additional parameter
    y_ref = SX.sym('y_ref', nx)  # Measurement at current stage
    x_prior = SX.sym('x_prior', nx)  # Prior state for arrival cost
    param_prior = SX.sym('p_prior', model_mhe.param_length)  # Prior state for arrival cost
    # Define custom external cost functions

    ocp_mhe.model.p = vertcat(p, y_ref, x_prior, param_prior)

    # Stage cost (for stages 1..N-1)
    # Cost = ||x[:nx] - y_ref||_R^2 + ||u||_Q^2
    stage_cost_expr = (state - y_ref).T @ params.measurements_residual_r @ (state - y_ref)  # Measurement cost
    stage_cost_expr += u.T @ params.noise_peanlty_w @ u  # Process noise cost

    # Initial stage cost (includes arrival cost)
    # Cost = ||x[:nx] - y_ref||_R^2 + ||u||_Q^2 + ||x - x_prior||_Q0^2
    initial_cost_expr = (state - x_prior).T @ params.state_prior_q0 @ (state - x_prior)
    initial_cost_expr += (thetas - param_prior).T @ params.params_prior_p0 @ (thetas - param_prior)
    # Set cost type to EXTERNAL
    ocp_mhe.cost.cost_type = 'EXTERNAL'
    ocp_mhe.cost.cost_type_e = 'EXTERNAL'
    ocp_mhe.cost.cost_type_0 = 'EXTERNAL'

    # Define external cost expressions
    ocp_mhe.model.cost_expr_ext_cost = stage_cost_expr
    ocp_mhe.model.cost_expr_ext_cost_e = 0  # Terminal cost
    ocp_mhe.model.cost_expr_ext_cost_0 = initial_cost_expr
    ocp_mhe.parameter_values = np.zeros((n_inp_param + nx + nx_augmented,))

    # Set number of shooting nodes
    ocp_mhe.solver_options.N_horizon = params.mhe_horizont

    # Set QP solver (HPIPM is generally better for MHE)
    ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp_mhe.solver_options.hessian_approx = 'EXACT'
    ocp_mhe.solver_options.sim_method_num_stages = 4
    ocp_mhe.solver_options.sim_method_num_steps = 3
    discrete: bool = is_discrete(model_mhe)
    if (discrete):
        ocp_mhe.solver_options.integrator_type = 'DISCRETE'
    else:
        ocp_mhe.solver_options.integrator_type = 'ERK'
    # Set prediction horizon
    ocp_mhe.solver_options.tf = params.mhe_horizont * params.dt

    # NLP solver options
    ocp_mhe.solver_options.nlp_solver_type = 'SQP'  # Faster than SQP for real-time

    ocp_mhe.constraints.lbu = np.array([-0.001])  # Bound on noises
    ocp_mhe.constraints.ubu = np.array([0.001])
    ocp_mhe.constraints.idxbu = np.array([0])

    # Create solver
    ocp_mhe.solver_options.code_export_directory = str(generated_folder)
    ocp_mhe.code_export_directory = str(generated_folder)
    ocp_mhe.solver_options.json_file = str(generated_folder / 'mhe_config.json')
    ocp_mhe.solver_options.eval_residual_at_max_iter = True
    ocp_mhe.solver_options.print_level = 0
    ocp_mhe.solver_options.nlp_solver_stats_level = 1
    acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file=ocp_mhe.solver_options.json_file, build=True, generate=True)

    definitions = {
        "STATE_LENGTH": nx,
        "STATE_LENGTH_AUG": nx_augmented
    }
    generate_header(generated_folder / model_name / "include/mhe_settings.hpp",\
                     definitions, f'MPC_SETTINGS_{model_name}')
    return acados_solver_mhe