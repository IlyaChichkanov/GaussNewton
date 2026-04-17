
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
    def __init__(self, system: ODESystem):
        self.system = system
        self.state_length = system.nx
        self.input_length = system.nu
        self.param_length = system.np
        self.obs_length = system.n_obs

    def continuous_dynamics(self, state, params, noise, input_signals) -> SX:
        dstate = self.system.get_derivative(state, params, input_signals)
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
        acados_model.x = vertcat(x, parameters)
        acados_model.u = w_noise
        acados_model.param_length = self.param_length
        acados_model.state_length = self.state_length
        acados_model.p = p
        return acados_model

    def make_discrete_acados_model(self, ts: float) -> AcadosModel:
        """Continuous bicycle model without input delay for MHE."""
        # State variables.
        x = SX.sym('x', self.state_length)
        w_noise = SX.sym('w_noise', self.state_length)
        parameters = SX.sym('params', self.param_length)
        # Parameters
        vx = SX.sym('vx')
        steering = SX.sym('steering')
        p = vertcat(vx, steering)
        delayed_buf_u = SX.sym('delayed_u', 0)
        # Parameters
        # Continuous dynamics from base model
        delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else steering
        p = vertcat(vx, delayed_u)
        k1 = self.continuous_dynamics(x, parameters, w_noise, p)
        k2 = self.continuous_dynamics(x + 0.5 * ts * k1, parameters, w_noise, p)
        k3 = self.continuous_dynamics(x + 0.5 * ts * k2, parameters, w_noise, p)
        k4 = self.continuous_dynamics(x + ts * k3, parameters, w_noise, p)

        x_next = x + (ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if (delayed_buf_u.shape[0] > 0):
            x = vertcat(x, delayed_buf_u)
            buf_next = vertcat(steering, delayed_buf_u[:-1])
            x_next = vertcat(x_next, buf_next)

        # Create Acados model
        acados_model = AcadosModel()
        acados_model.disc_dyn_expr = x_next
        acados_model.x = vertcat(x, parameters)
        acados_model.u = w_noise
        acados_model.param_length = self.param_length
        acados_model.state_length = self.state_length
        acados_model.p = vertcat(vx, steering)
        return acados_model
    
    def create_step_function(self, dt, name) -> Function:
        print(f'create_step_function {name}')
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
        fun_name = f'step_{name}'
        step_func = Function(fun_name, [x, theta, u], [x_next, J_x, J_theta],
                            ['x', 'theta', 'u'], ['x_next', 'Jx', 'Jtheta'])
        return step_func
    
    def create_intefrate_function(self, dt, name) -> Function:
        print(f'create_step_function {name}')
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

        fun_name = f'step_{name}'
        integrate_func = Function(fun_name, [x, theta, u], [x_next],
                            ['x', 'theta', 'u'], ['x_next'])
        return integrate_func 
    
    def compute_fim(self, N, dt, input_signals_data, x0, theta, R_inv=None) -> np.ndarray:
        """
        Compute Fisher Information Matrix for parameters theta based on measurements.

        Parameters:
            N : int
                Number of integration steps (and measurements).
            dt : float
                Time step.
            input_signals_data : numpy array of shape (N, self.input_length)
                Input signals at each step.
            x0 : initial state (vector of length nx)
            theta : parameter vector (length n_theta)
            R_inv : measurement weight matrix (n_obs x n_obs) or scalar.
                If None, identity is used.
        Returns:
            FIM : (n_theta, n_theta) numpy array
        """
        nx = self.state_length
        nu = self.input_length
        n_theta = len(theta)
        n_obs = self.obs_length

        # Symbolic variables
        input_sym = ca.SX.sym('input', N, nu)   # (N, nu)
        x0_sym = ca.SX.sym('x0', nx)
        theta_sym = ca.SX.sym('theta', n_theta)

        x = x0_sym
        y_list = []

        for k in range(N):
            u = input_sym[k, :]   # row k
            # RK4 step
            k1 = self.system.get_derivative(x, theta_sym, u)
            k2 = self.system.get_derivative(x + 0.5*dt*k1, theta_sym, u)
            k3 = self.system.get_derivative(x + 0.5*dt*k2, theta_sym, u)
            k4 = self.system.get_derivative(x + dt*k3, theta_sym, u)
            x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            # Measurement at this step (after integration)
            y = self.h_x(x, theta_sym, u)
            y_list.append(y)

        # Stack all measurements vertically: (N * n_obs, 1)
        y_all = ca.vertcat(*y_list)

        # Jacobian of measurements w.r.t. parameters
        J = ca.jacobian(y_all, theta_sym)   # shape (N*n_obs, n_theta)

        # Build weight matrix W (block diagonal)
        if R_inv is None:
            W = ca.DM.eye(N * n_obs)
        elif isinstance(R_inv, (int, float)):
            W = R_inv * ca.DM.eye(N * n_obs)
        else:
            R_inv = np.asarray(R_inv)
            if R_inv.ndim == 1:
                R_mat = np.diag(R_inv)
            else:
                R_mat = R_inv
            if R_mat.shape != (n_obs, n_obs):
                raise ValueError(f"R_inv must be {n_obs}x{n_obs}, got {R_mat.shape}")
            # Create block diagonal matrix
            W = ca.DM.zeros(N * n_obs, N * n_obs)
            for i in range(N):
                W[i*n_obs:(i+1)*n_obs, i*n_obs:(i+1)*n_obs] = ca.DM(R_mat)

        # Fisher Information Matrix = J^T * W * J
        F = ca.mtimes([J.T, W, J])

        # Substitute numeric data
        # Note: input_signals_data must be of shape (N, nu)
        args = [input_signals_data, x0, theta]
        F_num = ca.Function('F', [input_sym, x0_sym, theta_sym], [F])(*args)

        return np.array(F_num).reshape((n_theta, n_theta))


    def compute_observed_fim(self, N, dt, simU, simY, initial_x0, theta_est, R_inv=None):
        """
        Вычисляет наблюдаемую информационную матрицу Фишера (гессиан) в точке оценки.
        
        Параметры:
            mhe_model: модель
            dt: шаг дискретизации
            simU: массив входов (N, n_in)
            simY: массив измерений (N+1, nx)
            initial_x0: начальное состояние (nx)
            theta_est: оценка параметров (n_theta)
            R_inv: обратная ковариация шума измерений (nx x nx)
        
        Возвращает:
            F_obs: наблюдаемая FIM (n_theta x n_theta)
        """
        nx = self.state_length
        n_theta = len(theta_est)
        N = simU.shape[0]   # число шагов

        if R_inv is None:
            R_inv = np.eye(nx)
        elif isinstance(R_inv, (int, float)):
            R_inv = R_inv * np.eye(nx)
        else:
            R_inv = np.asarray(R_inv)
            if R_inv.ndim == 1:
                R_inv = np.diag(R_inv)
            if R_inv.shape != (nx, nx):
                raise ValueError(f"R_inv must be {nx}x{nx}")

        # Символьные переменные
        theta_sym = ca.SX.sym('theta', n_theta)
        x_sym = ca.SX.sym('x', nx)          # начальное состояние фиксировано (initial_x0)
        u_sym = ca.SX.sym('u', N, simU.shape[1])

        # Функция одного шага RK4
        def step(x, theta, u):
            k1 = self.main_dynamics(x, theta, u)
            k2 = self.main_dynamics(x + 0.5*dt*k1, theta, u)
            k3 = self.main_dynamics(x + 0.5*dt*k2, theta, u)
            k4 = self.main_dynamics(x + dt*k3, theta, u)
            return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Симуляция траектории выхода (все состояния)
        y_sim = []
        
        x = initial_x0
        for k in range(N):
            u = u_sym[k, :]
            x = step(x, theta_sym, u)
            y_sim.append(x)
        y_all = ca.vertcat(*y_sim)   # (N+1)*nx x 1

        # Измерения (константы)
        Y_meas = ca.DM(simY.reshape(-1, 1))

        # Невязка и взвешенная сумма квадратов
        residuals = y_all - Y_meas
        W_blocks = [ca.DM(R_inv) for _ in range(N)]
        W = ca.diagcat(*W_blocks)
        J_meas = ca.mtimes([residuals.T, W, residuals])

        # Гессиан J_meas по параметрам
        H = ca.hessian(J_meas, theta_sym)[0]   # матрица 2-х производных

        # Вычисление в точке оценки
        func_H = ca.Function('H', [theta_sym, u_sym], [H])
        H_num = func_H(theta_est, simU)

        # Наблюдаемая FIM = 0.5 * H (поскольку в MHE cost = J_meas, а логарифм правдоподобия = -0.5 * J_meas)
        F_obs = 0.5 * np.array(H_num)

        # Добавляем априорный вклад (если есть)
        # Здесь нужно добавить P0, если он использовался в MHE
        # F_obs += P0
        return F_obs

class MheCogeGenerator(ABC):
    def __init__(self, mhe_model: ODESystem, params: MheParams, generated_folder: Path, model_name: str):
        self.params = params
        self.generated_folder = generated_folder
        self.model_name = model_name
        self.mhe_model = MheModel(mhe_model)
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
        niose = model_acados.u
        input_signal = model_acados.p
        nx = model.state_length
        nu = model.input_length
        n_obs_len = model.obs_length
        n_theta = model.param_length
        nx_augmented = nx + n_theta
        state, thetas = x[:nx], x[nx:]  # thetas длины 3
        y_meas = SX.sym('y_meas', n_obs_len)
        x_prior = SX.sym('x_prior', nx)
        param_prior = SX.sym('param_prior', n_theta)  # =3
        p_prior_weights = SX.sym('p_prior_weights', n_theta * n_theta)

        ocp_mhe.model.p = vertcat(input_signal, y_meas, x_prior, param_prior, p_prior_weights)
        ocp_mhe.parameter_values = np.zeros((nu + n_obs_len + nx_augmented + n_theta * n_theta,))
        # Cost expressions (как у вас, но с учётом размерностей)
        P0 = reshape(p_prior_weights, n_theta, n_theta) * self.params.fim_scaler
        Q0 = self.params.state_prior_q0
        R = self.params.measurements_residual_r
        W = self.params.noise_peanlty_w

        residual = model.h_x(state, thetas, input_signal) - y_meas

        stage_cost_expr = residual.T @ R @ residual + niose.T @ W @ niose
        initial_cost_expr = (state - x_prior).T @ Q0 @ (state - x_prior) +\
              (thetas - param_prior).T @ P0 @ (thetas - param_prior)
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
        ocp_mhe.solver_options.levenberg_marquardt = 1e-6
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
        bounds_noise = self.params.bounds_noise
        ocp_mhe.constraints.lbu = np.array([b[0] for b in bounds_noise])
        ocp_mhe.constraints.ubu = np.array([b[1] for b in bounds_noise])
        ocp_mhe.constraints.idxbu = np.arange(0, nx)

        print(ocp_mhe.constraints.lbx)
        print(ocp_mhe.constraints.ubx)
        print(ocp_mhe.constraints.idxbx)

        discrete: bool = is_discrete(ocp_mhe.model)
        if (discrete):
            ocp_mhe.solver_options.integrator_type = 'DISCRETE'
            ocp_mhe.solver_options.sim_method_num_stages = 4
            ocp_mhe.solver_options.sim_method_num_steps = 4
        else:
            # ocp_mhe.solver_options.integrator_type = 'ERK'
            # ocp_mhe.solver_options.sim_method_num_stages = 4
            # ocp_mhe.solver_options.sim_method_num_steps = 4

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
        self.generate_fim_function(self.params.dt)
        return acados_solver_mhe
        # self.generate_header()
        # 

    def generate_header(self):
        model = self.get_model()
        diag = np.diag(self.params.measurements_residual_r)
        # Convert to a comma‑separated list with braces
        res_vector_str = "{" + ", ".join(f"{x:.15g}" for x in diag) + "}"
        definitions = {
            "STATE_LENGTH": model.state_length,
            "STATE_LENGTH_AUG": model.param_length + model.state_length,
            "RES_VECTOR": res_vector_str
        }
        generate_header(self.generated_folder / self.model_name / "include/mhe_settings.hpp",\
                        definitions, f'MPC_SETTINGS_{self.model_name}')

    def generate_fim_function(self, dt: float):
        fim_func = self.get_model().create_step_function(dt, self.model_name)
        output_dir = str(self.generated_folder / self.model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Base name for the generated files (no path, no extension)
        base_name = f'fim_{self.model_name}_generator'

        # Create the code generator with the base name
        cg = CodeGenerator(base_name, {'main': True, 'cpp': True, 'with_header': True, 'mex': False})
        cg.add(fim_func)

        # Generate to the full path (directory + base name, still without extension)
        cg.generate(os.path.join(output_dir, ""))
