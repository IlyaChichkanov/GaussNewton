
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, CodeGenerator, Function, jacobian, vertcat
import casadi as ca
from params import MheParams
from ocp_utils import generate_header, is_discrete


class MheModel(ABC):
    def __init__(self, nx, nu, np):
        self.state_length = nx
        self.input_length = nu
        self.param_length = np

    @abstractmethod
    def main_dynamics(self, state: SX, params: SX, input_signals: SX) -> SX:
        pass

    @property
    @abstractmethod
    def bounds_param(self) -> list[tuple]:
        pass

    @property
    @abstractmethod
    def bounds_state(self) -> list[tuple]:
        pass

    @property
    @abstractmethod
    def bounds_noise(self) -> list[tuple]:
        pass

    def continuous_dynamics(self, state, params, noise, input_signals) -> SX:
        dstate = self.main_dynamics(state, params, input_signals)
        dstate += noise
        dx = vertcat(dstate, SX(np.zeros(self.param_length)))
        return dx

    def create_step_function(self, dt):
        print(f'create_step_function {self.name}')
        print(self.param_length)
        x = SX.sym('x', self.state_length)
        theta = SX.sym('theta', self.param_length)
        u = SX.sym('u', self.input_length)  # [vx, steering]

        # Один шаг RK4
        k1 = self.main_dynamics(x, theta, u)
        k2 = self.main_dynamics(x + 0.5 * dt * k1, theta, u)
        k3 = self.main_dynamics(x + 0.5 * dt * k2, theta, u)
        k4 = self.main_dynamics(x + dt * k3, theta, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Якобианы
        J_x = jacobian(x_next, x)          # ∂x_next/∂x
        J_theta = jacobian(x_next, theta)  # ∂x_next/∂θ
        fun_name = f'step_{self.name}'
        step_func = Function(fun_name, [x, theta, u], [x_next, J_x, J_theta],
                            ['x', 'theta', 'u'], ['x_next', 'Jx', 'Jtheta'])
        return step_func

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
        acados_model.name = self.name
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
        acados_model.name = self.name
        return acados_model
    
    def compute_fim(self, N, dt, input_signals_data, x0, theta, R_inv=None):
        """
        Compute Fisher Information Matrix for parameters theta.

        Parameters:
            input_signals_data : numpy array of shape (N, self.input_signal_length)
                Input signals at each step.
            x0 : initial state (vector of length nx)
            theta : parameter vector (length n_theta)
            R_inv : measurement weight (optional)
        """
        nx = self.state_length
        nu = self.input_length
        n_theta = len(theta)

        # Symbolic variables
        input_signals_sym = ca.SX.sym('input', N, nu)   # (N x input_signal_length)
        x0_sym = ca.SX.sym('x0', nx)
        theta_sym = ca.SX.sym('theta', n_theta)

        x = x0_sym
        y_list = []

        for k in range(N):
            u = input_signals_sym[k, :]   # row k
            k1 = self.main_dynamics(x, theta_sym, u)
            k2 = self.main_dynamics(x + 0.5*dt*k1, theta_sym, u)
            k3 = self.main_dynamics(x + 0.5*dt*k2, theta_sym, u)
            k4 = self.main_dynamics(x + dt*k3, theta_sym, u)
            x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            y_list.append(x)

        y_all = ca.vertcat(x0_sym, *y_list)
        J = ca.jacobian(y_all, theta_sym)

        # Build weight matrix W (same as before)
        if R_inv is None:
            W = ca.DM.eye((N+1)*nx)
        elif isinstance(R_inv, (int, float)):
            W = R_inv * ca.DM.eye((N+1)*nx)
        else:
            R_inv = np.asarray(R_inv)
            if R_inv.ndim == 1:
                R_mat = np.diag(R_inv)
            else:
                R_mat = R_inv
            if R_mat.shape != (nx, nx):
                raise ValueError(f"R_inv must be {nx}x{nx}, got {R_mat.shape}")
            W = ca.DM.zeros((N+1)*nx, (N+1)*nx)
            for i in range(N+1):
                W[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = ca.DM(R_mat)

        F = ca.mtimes([J.T, W, J])

        # Substitute numeric data
        # input_signals_data must be a 2D array (N, input_signal_length)
        args = [input_signals_data, x0, theta]
        F_num = ca.Function('F', [input_signals_sym, x0_sym, theta_sym], [F])(*args)

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
    def __init__(self, params: MheParams, generated_folder: Path, model_name: str):
        self.params = params
        self.generated_folder = generated_folder
        self.model_name = model_name

    @abstractmethod
    def set_ocp_problem(self) -> AcadosOcp:
        pass

    @abstractmethod
    def get_model() -> MheModel:
        pass

    def generate_code(self):
        ocp_mhe = self.set_ocp_problem()
        ocp_mhe.solver_options.N_horizon = self.params.mhe_horizont
        ocp_mhe.solver_options.tf = self.params.mhe_horizont * self.params.dt
        model = self.get_model()
        nx = model.state_length
        bounds_state = model.bounds_state
        lb_state = np.array([b[0] for b in bounds_state])
        ub_state = np.array([b[1] for b in bounds_state])
        finite_mask = np.isfinite(lb_state) & np.isfinite(ub_state)
        idx_state = np.arange(0, nx)[finite_mask]
        lb_state = lb_state[finite_mask]
        ub_state = ub_state[finite_mask]

        bounds_param = model.bounds_param
        lb_theta = np.array([b[0] for b in bounds_param])
        ub_theta = np.array([b[1] for b in bounds_param])
        finite_mask = np.isfinite(lb_theta) & np.isfinite(ub_theta)
        idx_theta = np.arange(nx, nx + len(lb_theta))[finite_mask]
        lb_theta = lb_theta[finite_mask]
        ub_theta = ub_theta[finite_mask]

        ocp_mhe.constraints.lbx = np.hstack((lb_state, lb_theta))
        ocp_mhe.constraints.ubx = np.hstack((ub_state, ub_theta))
        ocp_mhe.constraints.idxbx = np.hstack((idx_state, idx_theta))

        ocp_mhe.constraints.lbu = np.array([b[0] for b in model.bounds_noise])
        ocp_mhe.constraints.ubu = np.array([b[1] for b in model.bounds_noise])
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
        fim_func = self.get_model().create_step_function(dt)
        output_dir = str(self.generated_folder / self.model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Base name for the generated files (no path, no extension)
        base_name = f'fim_{self.model_name}_generator'

        # Create the code generator with the base name
        cg = CodeGenerator(base_name, {'main': True, 'cpp': True, 'with_header': True, 'mex': False})
        cg.add(fim_func)

        # Generate to the full path (directory + base name, still without extension)
        cg.generate(os.path.join(output_dir, ""))
