from abc import ABC
from pathlib import Path

import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat
from commom_utils.ocp_utils import generate_header, is_discrete, quadform
from commom_utils.ode_system import ODESystem
from mpc.params import CarParams, MpcParams
from commom_utils.systems import *

class MpcModel(ABC):
    def __init__(self, system: ODESystem):
        self.system = system
        self.state_length = system.nx
        self.input_length = system.nu
        self.param_length = system.np
        self.obs_length = system.n_obs

    def make_sym_variables(self) -> tuple[SX]:
        # State variables
        x = SX.sym('x', self.state_length) # tau, psi ..
        input_params = SX.sym('params', self.param_length)
        cmd = SX.sym('inp_control')
        # Control input
        # Parameters
        vx = SX.sym('vx')
        c = SX.sym('c')
        return x, cmd, vx, c, input_params

    def make_discrete_acados_model(self, ts: float, n_delay: int, use_ddu: bool) -> AcadosModel:
        delayed_buf_u = SX.sym('delayed_u', n_delay)
        ddu = SX.sym('ddu')
        du = SX.sym('du')
        x, cmd, vx, c, inp_params = self.make_sym_variables()
        # RK4 integration
        delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else cmd
        k1 = self.system.get_derivative(x, vertcat(delayed_u, vx, c), inp_params)
        k2 = self.system.get_derivative(x + 0.5 * ts * k1, vertcat(delayed_u, vx, c), inp_params)
        k3 = self.system.get_derivative(x + 0.5 * ts * k2, vertcat(delayed_u, vx, c), inp_params)
        k4 = self.system.get_derivative(x + ts * k3, vertcat(delayed_u, vx, c), inp_params)
        x_next = x + (ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        delay_dynamics = []
        for i in range(n_delay):
            if i == 0:
                delay_dynamics.append(cmd)
            else:
                delay_dynamics.append(delayed_buf_u[i - 1])

        x = vertcat(x, cmd)
        rwa_next = cmd + du * ts
        x_next = vertcat(x_next, rwa_next)
        control_output = du
        if (use_ddu):
            x = vertcat(x, du)
            du_next = du + ddu * ts
            x_next = vertcat(x_next, du_next)
            control_output = ddu

        x_next = vertcat(x_next, *delay_dynamics)
        x = vertcat(x, delayed_buf_u)
        model = AcadosModel()
        model.disc_dyn_expr = x_next
        model.x = x
        model.u = control_output
        model.p = vertcat(vx, c, inp_params)
        return model

    def make_continuous_acados_model(self) -> AcadosModel:
        """Continuous bicycle model without input delay for MPC."""
        x, cmd, vx, c, inp_params = self.make_sym_variables()
        # Continuous dynamics from base model
        base_dynamics = self.system.get_derivative(x, vertcat(cmd, vx, c), inp_params)

        # Full dynamics with rwa integration
        dx = vertcat(base_dynamics)

        # Create Acados model
        acados_model = AcadosModel()
        acados_model.f_expl_expr = dx
        acados_model.x = x
        acados_model.u = cmd
        acados_model.p = vertcat(vx, c, inp_params)
        return acados_model


class MpcCogeGenerator(ABC):
    def __init__(self, params: MpcParams, generated_folder: Path, model_name: str):
        self.params = params
        self.generated_folder = generated_folder
        self.model_name = model_name

    def generate_code(self):
        ocp_mpc: AcadosOcp = self.set_ocp_problem()
        Tf = self.params.mpc_horizont * self.params.ts  # prediction horizon [s]
        ocp_mpc.solver_options.N_horizon = self.params.mpc_horizont
        ocp_mpc.solver_options.nlp_solver_stats_level = 1
        ocp_mpc.solver_options.print_level = 1

        ocp_mpc.solver_options.tf = Tf
        ocp_mpc.solver_options.qp_solver_warm_start = True
        ocp_mpc.parameter_values = np.zeros(ocp_mpc.model.p.shape[0])
        ocp_mpc.constraints.x0 = np.zeros(len(ocp_mpc.model.x.elements()))
        ocp_mpc.constraints.idxbu = np.array([0])  # Constrain Δu
        discrete: bool = is_discrete(ocp_mpc.model)
        if (discrete):
            ocp_mpc.solver_options.integrator_type = 'DISCRETE'
        else:
            ocp_mpc.solver_options.integrator_type = 'ERK'
        ocp_mpc.solver_options.code_export_directory = str(self.generated_folder)
        ocp_mpc.code_export_directory = str(self.generated_folder)
        ocp_mpc.solver_options.json_file = str(self.generated_folder / 'acados_ocp_nlp2.json')
        ocp_mpc.model.name = self.model_name
        print("con_h_expr size:", ocp_mpc.model.con_h_expr.size1())
        print("lh size:", len(ocp_mpc.constraints.lh))
        print("uh size:", len(ocp_mpc.constraints.uh))
        print("discrete: ", discrete)
        acados_solver_mpc = AcadosOcpSolver(ocp_mpc, \
                                            json_file=ocp_mpc.solver_options.json_file, build=True, generate=True)
        self.generate_header()
        return acados_solver_mpc

    def generate_header(self):
        rwa_pos: int = 2
        param_length = self.model.param_length

        definitions = {
            "PARAM_LENGTH": param_length,
            "RWA_POS": rwa_pos,
            "USE_DDU_CONTROL": int(self.params.use_ddu_control),
        }

        generate_header(self.generated_folder / self.model_name / "include/mpc_settings.h", \
                        definitions, f'MPC_SETTINGS_{self.model_name}')

class KinematicModel(ODESystem):
    def __init__(self, car_params: CarParams):
        self.gear_ratio = car_params.gear_ratio
        self.wheelbase = car_params.wheelbase
        super().__init__(nx=2, np=1, nu=3)

    def get_derivative(self, state, inp_signals, params) -> SX:
        rwa_cmd, v, c = inp_signals[0], inp_signals[1], inp_signals[2]
        tau, psi = state[0], state[1]
        offset = params[0]
        dtau = np.sin(psi)
        rwa = rwa_cmd + offset
        denominator = 1 - c * tau * v
        dpsi = v * ca.tan(rwa) / self.wheelbase - v * c * np.cos(psi) / denominator
        dx = vertcat(dtau, dpsi)
        return dx


class KinematicModel2dIntegrator(ODESystem):
    def __init__(self, car_params: CarParams):
        self.gear_ratio = car_params.gear_ratio
        self.wheelbase = car_params.wheelbase
        super().__init__(nx=4, np=1, nu=3)

    def get_derivative(self, state, inp_signals, params) -> SX:
        ddu_cmd, v, c = inp_signals[0], inp_signals[1], inp_signals[2]
        tau, psi, rwa, rwa_dot = state[0], state[1], state[2], state[3]
        offset = params[0]
        dtau = np.sin(psi)
        drwa = rwa_dot
        drwa_dot = ddu_cmd

        denominator = 1 - c * tau * v
        dpsi = v * ca.tan(rwa + offset) / self.wheelbase - v * c * np.cos(psi) / denominator
        dx = vertcat(dtau, dpsi, drwa, drwa_dot)
        return dx


class KinematicModel2dIntegratorWithDelay(ODESystem):
    def __init__(self, car_params, delay_order=2):
        self.wheelbase = car_params.wheelbase
        self.delay = DelaySystem(order=delay_order)
        # Состояния: tau, psi, rwa, rwa_dot + состояния задержки
        nx_total = 4 + self.delay.nx
        np_total = 1 + self.delay.np   # offset + tau_d
        nu_total = 3                   # ddu_cmd, v, c
        super().__init__(nx=nx_total, np=np_total, nu=nu_total)

    def get_derivative(self, state, inp_signals, params):
        tau = state[0]
        psi = state[1]
        rwa = state[2]
        rwa_dot = state[3]
        state_delay = state[4:]

        offset = params[1]
        tau_d = params[0]

        ddu_cmd, v, c = inp_signals[0], inp_signals[1], inp_signals[2]

        # Двойной интегратор (без задержки)
        drwa = rwa_dot
        drwa_dot = ddu_cmd

        # Задержка применяется к rwa (с учётом offset)
        rwa_delayed = self.delay.observation(state_delay, [tau_d], rwa)
        rwa_actual = rwa_delayed + offset  # смещение уже учтено

        # Кинематика с задержанным углом
        dtau = ca.sin(psi)
        denominator = 1 - c * tau * v
        dpsi = v * ca.tan(rwa_actual) / self.wheelbase - v * c * ca.cos(psi) / denominator

        # Динамика задержки (вход – rwa_with_offset)
        dx_delay = self.delay.get_derivative(state_delay, [tau_d], rwa)

        return ca.vertcat(dtau, dpsi, drwa, drwa_dot, dx_delay)

    def observation(self, state, inp_signals, params):
        return state[1:2]   # psi


class KinematicMpcCodegenerator(MpcCogeGenerator):
    def __init__(self, params: MpcParams, generated_folder: Path, model_name: str):
        super().__init__(params, generated_folder, model_name)
        #system = KinematicModel(params.car_params)
        #system = KinematicModel2dIntegrator(params.car_params)
        system = KinematicModel2dIntegratorWithDelay(params.car_params)

        self.model: MpcModel = MpcModel(system)
        self.params = params

    def set_ocp_problem(self):
        # Create OCP object
        ocp = AcadosOcp()
        base_model = self.model
        rwa_pos: int = 2
        # model: AcadosModel = base_model.make_discrete_acados_model(self.params.ts, self.params.n_delay, \
        #                                                            self.params.use_ddu_control)

        model = base_model.make_continuous_acados_model()

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
            a_comf = vx * vx * rwa / self.params.car_params.wheelbase
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
