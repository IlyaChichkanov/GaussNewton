
import os
from abc import ABC
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, CodeGenerator, Function, jacobian, vertcat, reshape, fmax
import casadi as ca
from scipy.optimize import fsolve
from mhe.params import MheParams
from commom_utils.ocp_utils import generate_header, is_discrete
from commom_utils.ode_system import ODESystem



class MheModel(ABC):
    def __init__(self, system: ODESystem, use_noise: bool):
        self.use_noise = use_noise
        self.system = system
        self.state_length = system.nx
        self.input_length = system.nu
        self.param_length = system.n_theta
        self.obs_length = system.n_obs

    def continuous_dynamics(self, state, params, noise, input_signals) -> SX:
        dstate = self.system.get_derivative(state, params, input_signals)
        if (self.use_noise):
            dstate += noise
        dx = vertcat(dstate, SX(np.zeros(self.param_length)))
        return dx

    def h_x(self, state: SX, theta: SX, u: SX) -> SX:
        return self.system.observation(state, theta, u)

    def make_continious_acados_model(self) -> AcadosModel:
        """Continuous bicycle model without input delay for MHE."""
        # State variables.
        x = SX.sym('x', self.state_length)
        w_noise = SX.sym('w_noise', self.state_length)
        parameters = SX.sym('params', self.param_length)
        # Parameters
        p = SX.sym('u', self.input_length)
        dx = self.continuous_dynamics(x, parameters, w_noise, p)
        # Create Acados model
        acados_model = AcadosModel()
        acados_model.f_expl_expr = dx
        xdot = SX.sym('xdot', self.state_length + self.param_length)
        acados_model.xdot = xdot
        acados_model.f_impl_expr = xdot - dx
        acados_model.f_expl_expr = dx
        acados_model.x = vertcat(x, parameters)
        if (self.use_noise):
            acados_model.u = w_noise
        acados_model.param_length = self.param_length
        acados_model.state_length = self.state_length
        acados_model.p = p
        return acados_model

    def create_observation_function(self, fun_name) -> Function:
        x = SX.sym('x', self.state_length)
        theta = SX.sym('theta', self.param_length)
        u = SX.sym('u', self.input_length)
        y = self.system.observation(x, theta, u)
        J_yx = jacobian(y, x)
        J_ytheta = jacobian(y, theta)
        obs_func = Function(fun_name, [x, theta, u],
                            [y, J_yx, J_ytheta],
                            ['x', 'theta', 'u'],
                            ['y', 'Jy_x', 'Jy_theta'])
        return obs_func

    def create_step_function(self, dt, fun_name) -> Function:
        print(f'create_step_function {fun_name}')
        print(self.param_length)
        x = SX.sym('x', self.state_length)
        theta = SX.sym('theta', self.param_length)
        u = SX.sym('u', self.input_length)  # [vx, steering]

        # Один шаг RK4
        k1 = self.system.get_derivative(x, theta, u)
        k2 = self.system.get_derivative(x + 0.5 * dt * k1, theta, u)
        k3 = self.system.get_derivative(x + 0.5 * dt * k2, theta, u)
        k4 = self.system.get_derivative(x + dt * k3, theta, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Якобианы
        J_x = jacobian(x_next, x)          # ∂x_next/∂x
        J_theta = jacobian(x_next, theta)  # ∂x_next/∂θ
        step_func = Function(fun_name, [x, theta, u], [x_next, J_x, J_theta],
                            ['x', 'theta', 'u'], ['x_next', 'Jx', 'Jtheta'])
        return step_func

    def compute_augmented_fim(self, dt, input_signals_data, x0, theta, r_inv):
        nx = self.state_length
        n_theta = self.param_length
        n_obs = self.obs_length
        N = input_signals_data.shape[0]

        input_sym = ca.SX.sym('input', N, self.input_length)
        x0_sym = ca.SX.sym('x0', nx)
        theta_sym = ca.SX.sym('theta', n_theta)

        x = x0_sym
        y_list = []
        for k in range(N):
            u = input_sym[k, :]
            k1 = self.system.get_derivative(x, theta_sym, u)
            k2 = self.system.get_derivative(x + 0.5 * dt * k1, theta_sym, u)
            k3 = self.system.get_derivative(x + 0.5 * dt * k2, theta_sym, u)
            k4 = self.system.get_derivative(x + dt * k3, theta_sym, u)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y = self.h_x(x, theta_sym, u)
            y_list.append(y)

        y_all = ca.vertcat(*y_list)
        aug_sym = ca.vertcat(x0_sym, theta_sym)
        J = ca.jacobian(y_all, aug_sym)   # (N*n_obs, nx+np)

        # Весовая матрица W
        if isinstance(r_inv, int | float):
            W = r_inv * ca.DM.eye(N * n_obs)
        else:
            r_inv = np.asarray(r_inv)
            if r_inv.ndim == 1:
                R_mat = np.diag(r_inv)
            else:
                R_mat = r_inv
            if R_mat.shape != (n_obs, n_obs):
                raise ValueError(f"r_inv must be {n_obs}x{n_obs}, got {R_mat.shape}")
            W = ca.DM.zeros(N * n_obs, N * n_obs)
            for i in range(N):
                W[i * n_obs:(i + 1) * n_obs, i * n_obs:(i + 1) * n_obs] = ca.DM(R_mat)

        F = ca.mtimes([J.T, W, J])

        x0_col = np.asarray(x0).reshape(-1, 1)
        theta_col = np.asarray(theta).reshape(-1, 1)
        assert x0_col.shape == (self.state_length, 1), f"x0 shape {x0_col.shape}"
        assert theta_col.shape == (self.param_length, 1), f"theta shape {theta_col.shape}"
        args = [input_signals_data, x0_col, theta_col]
        F_num = ca.Function('F_aug', [input_sym, x0_sym, theta_sym], [F])(*args)
        return np.array(F_num).reshape((nx + n_theta, nx + n_theta))


class MheCogeGenerator(ABC):
    def __init__(self, mhe_model: ODESystem, params: MheParams, generated_folder: Path, model_name: str):
        self.params = params
        self.generated_folder = generated_folder
        self.model_name = model_name
        self.mhe_model = MheModel(mhe_model, params.use_noise)
        state_length = self.mhe_model.state_length
        param_length = self.mhe_model.param_length
        obs_length = self.mhe_model.obs_length
        assert len(params.state_prior_q0) == state_length
        assert len(params.noise_peanlty_w) == state_length
        assert len(params.bounds_state) == state_length
        assert len(params.bounds_noise) == state_length
        assert len(params.measurements_residual_r) == obs_length

        assert len(params.bounds_param) == param_length
        assert params.mhe_horizont > 10
        assert params.dt > 0

    def set_ocp_problem(self) -> AcadosOcp:
        ocp_mhe = AcadosOcp()
        model: MheModel = self.mhe_model
        model_acados: AcadosModel = model.make_continious_acados_model()
        model_acados.name = self.model_name
        ocp_mhe.model = model_acados
        x = model_acados.x
        noise = model_acados.u
        input_signal = model_acados.p
        nx = model.state_length
        nu = model.input_length
        n_obs_len = model.obs_length
        n_theta = model.param_length
        state, thetas = x[:nx], x[nx:]  # thetas длины 3
        y_meas = SX.sym('y_meas', n_obs_len)
        x_prior = SX.sym('x_prior', nx)
        param_prior = SX.sym('param_prior', n_theta)  # =3

        n_aug = nx + n_theta
        p_prior_weights = SX.sym('p_prior_weights', n_aug * n_aug)

        P_aug = reshape(p_prior_weights, n_aug, n_aug) #* self.params.fim_scaler   # оставьте 1.0

        x_aug = vertcat(state, thetas)
        x_aug_prior = vertcat(x_prior, param_prior)

        ocp_mhe.model.p = vertcat(input_signal, y_meas, x_prior, param_prior, p_prior_weights)
        ocp_mhe.parameter_values = np.zeros((nu + n_obs_len + nx + n_theta + n_aug * n_aug,))
                # Cost expressions (как у вас, но с учётом размерностей)

        Q0 = self.params.state_prior_q0
        R = self.params.measurements_residual_r
        W = self.params.noise_peanlty_w

        residual = model.h_x(state, thetas, input_signal) - y_meas

        stage_cost_expr = residual.T @ R @ residual
        if (self.params.use_noise):
            stage_cost_expr += noise.T @ W @ noise
        initial_cost_expr = (x_aug - x_aug_prior).T @ P_aug @ (x_aug - x_aug_prior)
        ocp_mhe.model.cost_expr_ext_cost = stage_cost_expr
        ocp_mhe.model.cost_expr_ext_cost_e = 0  # Terminal cost
        ocp_mhe.model.cost_expr_ext_cost_0 = initial_cost_expr

        # Set cost type to EXTERNAL
        ocp_mhe.cost.cost_type = 'EXTERNAL'
        ocp_mhe.cost.cost_type_e = 'EXTERNAL'
        ocp_mhe.cost.cost_type_0 = 'EXTERNAL'

        ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'  # Faster than SQP for real-time
        ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp_mhe.solver_options.nlp_solver_max_iter = 15
        ocp_mhe.solver_options.levenberg_marquardt = 1e-3
        ocp_mhe.solver_options.hpipm_options = {
            # 'tol': 1e-6,
            # 'reg_epsilon': 1e-6,        # regularization on the Hessian
            # 'reg_epsilon_s': 1e-6,       # regularization on the slack variables (if any)
            # 'iter_max': 1000
                'scale': 1,  # включить автоматическое масштабирование
                'scale_ux': 1,  # масштабировать состояния и управления
        }
        ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp_mhe.solver_options.print_level = 0
        ocp_mhe.solver_options.nlp_solver_stats_level = 1
        return ocp_mhe

    def get_model(self) -> MheModel:
        return self.mhe_model

    def modify_ocp_problem(self, ocp_mhe: AcadosOcp) -> AcadosOcp:
        return ocp_mhe

    def generate_code(self):
        ocp_mhe = self.set_ocp_problem()
        ocp_mhe = self.modify_ocp_problem(ocp_mhe)

        ocp_mhe.solver_options.N_horizon = self.params.mhe_horizont
        ocp_mhe.solver_options.tf = self.params.mhe_horizont * self.params.dt
        model = self.get_model()
        nx = model.state_length
        bounds_state = self.params.bounds_state
        lb_state = np.array([b[0] for b in bounds_state])
        ub_state = np.array([b[1] for b in bounds_state])
        finite_mask = np.isfinite(lb_state) & np.isfinite(ub_state)
        idx_state = np.arange(0, nx)[finite_mask]
        lb_state = lb_state[finite_mask]
        ub_state = ub_state[finite_mask]

        bounds_param = self.params.bounds_param
        lb_theta = np.array([b[0] for b in bounds_param])
        ub_theta = np.array([b[1] for b in bounds_param])
        finite_mask = np.isfinite(lb_theta) & np.isfinite(ub_theta)
        idx_theta = np.arange(nx, nx + len(lb_theta))[finite_mask]
        lb_theta = lb_theta[finite_mask]
        ub_theta = ub_theta[finite_mask]

        ocp_mhe.constraints.lbx = np.hstack((lb_state, lb_theta))
        ocp_mhe.constraints.ubx = np.hstack((ub_state, ub_theta))
        ocp_mhe.constraints.idxbx = np.hstack((idx_state, idx_theta))
        if (self.params.use_noise):
            bounds_noise = self.params.bounds_noise
            ocp_mhe.constraints.lbu = np.array([b[0] for b in bounds_noise])
            ocp_mhe.constraints.ubu = np.array([b[1] for b in bounds_noise])
            ocp_mhe.constraints.idxbu = np.arange(0, nx)

        print(ocp_mhe.constraints.lbx)
        print(ocp_mhe.constraints.ubx)
        print(ocp_mhe.constraints.idxbx)

        ocp_mhe.solver_options.integrator_type = 'IRK'
        ocp_mhe.solver_options.sim_method_num_stages = 3   # 3 stages → 5th order
        ocp_mhe.solver_options.sim_method_newton_tol = 1e-8
        ocp_mhe.solver_options.sim_method_newton_iter = 5

        print(self.model_name)
        print(model.param_length, model.state_length)
        # Create solver
        ocp_mhe.solver_options.code_export_directory = str(self.generated_folder)
        ocp_mhe.code_export_directory = str(self.generated_folder)
        ocp_mhe.solver_options.json_file = str(self.generated_folder / 'mhe_config.json')
        ocp_mhe.solver_options.eval_residual_at_max_iter = True

        acados_solver_mhe = \
            AcadosOcpSolver(ocp_mhe, json_file=ocp_mhe.solver_options.json_file, build=True, generate=True)
        self.generate_functions(self.params.dt)
        self.generate_header()
        return acados_solver_mhe

    def generate_header(self):
        model = self.get_model()
        diag = np.diag(self.params.measurements_residual_r)
        # Convert to a comma‑separated list with braces
        res_vector_str = "{" + ", ".join(f"{x:.15g}" for x in diag) + "}"
        model_name = self.model_name.upper()
        definitions = {
            f'PARAM_LENGTH_{model_name}': model.param_length,
            f'STATE_LENGTH_{model_name}': model.state_length,
            f'OBSERV_LENGTH_{model_name}': model.obs_length,
            f'INPUT_LENGTH_{model_name}': model.input_length,
            f'RES_VECTOR_{model_name}': res_vector_str,
        }
        generate_header(self.generated_folder / self.model_name / "include/mhe_settings.hpp",\
                        definitions, f'MPC_SETTINGS_{self.model_name}')

    def generate_functions(self, dt: float):
        step_func = self.get_model().create_step_function(dt, f'step_{self.model_name}')
        obs_function = self.get_model().create_observation_function(f'obs_{self.model_name}')
        output_dir = str(self.generated_folder / self.model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Base name for the generated files (no path, no extension)
        base_name = f'fim_{self.model_name}_generator'

        # Create the code generator with the base name
        cg = CodeGenerator(base_name, {'main': True, 'cpp': True, 'with_header': True, 'mex': False})
        cg.add(step_func)
        cg.add(obs_function)

        # Generate to the full path (directory + base name, still without extension)
        cg.generate(os.path.join(output_dir, ""))
