from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

from params import MpcParams
from ocp_utils import generate_header, is_discrete


class MpcModel(ABC):

    @property
    @abstractmethod
    def state_length(self) -> int:
        pass

    @property
    @abstractmethod
    def param_length(self) -> int:
        pass

    @abstractmethod
    def main_dynamics(self, state: SX, rwa: SX, input_signals: SX, input_params: SX) -> SX:
        pass

    def make_sym_variables(self) -> tuple[SX]:
        # State variables
        x = SX.sym('x', self.state_length) # tau, psi ..
        input_params = SX.sym('params', self.param_length)
        rwa = SX.sym('rwa')
        # Control input
        du = SX.sym('du')
        # Parameters
        vx = SX.sym('vx')
        c = SX.sym('c')
        return x, du, rwa, vx, c, input_params

    def make_discrete_acados_model(self, ts: float, n_delay: int, use_ddu: bool) -> AcadosModel:
        delayed_buf_u = SX.sym('delayed_u', n_delay)
        ddu = SX.sym('ddu')
        x, du, rwa, vx, c, inp_params = self.make_sym_variables()
        inp_signals = vertcat(vx, c)
        # RK4 integration
        delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else rwa
        k1 = self.main_dynamics(x, delayed_u, inp_signals, inp_params)
        k2 = self.main_dynamics(x + 0.5 * ts * k1, delayed_u, inp_signals, inp_params)
        k3 = self.main_dynamics(x + 0.5 * ts * k2, delayed_u, inp_signals, inp_params)
        k4 = self.main_dynamics(x + ts * k3, delayed_u, inp_signals, inp_params)
        x_next = x + (ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        delay_dynamics = []
        for i in range(n_delay):
            if i == 0:
                delay_dynamics.append(rwa)
            else:
                delay_dynamics.append(delayed_buf_u[i - 1])

        x = vertcat(x, rwa)
        rwa_next = rwa + du * ts
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
        model.p = vertcat(inp_signals, inp_params)
        model.name = self.model_name
        return model

    def make_continuous_acados_model(self) -> AcadosModel:
        """Continuous bicycle model without input delay for MPC."""
        x, du, rwa, vx, c, inp_params = self.make_sym_variables()
        inp_signals = vertcat(vx, c)
        # Continuous dynamics from base model
        base_dynamics = self.main_dynamics(x, rwa, inp_signals, inp_params)

        # Full dynamics with rwa integration
        x = vertcat(x, rwa)
        dx = vertcat(base_dynamics, du)

        # Create Acados model
        acados_model = AcadosModel()
        acados_model.f_expl_expr = dx
        acados_model.x = x
        acados_model.u = du
        acados_model.p = vertcat(inp_signals, inp_params)
        acados_model.name = self.model_name
        return acados_model


class MpcCogeGenerator(ABC):
    def __init__(self, params: MpcParams, generated_folder: Path, model_name: str):
        self.params = params
        self.generated_folder = generated_folder
        self.model_name = model_name

    @abstractmethod
    def set_ocp_problem(self) -> AcadosOcp:
        pass

    @abstractmethod
    def get_model() -> MpcModel:
        pass

    def generate_code(self):
        ocp_mpc: AcadosOcp = self.set_ocp_problem()
        Tf = self.params.mpc_horizont * self.params.ts  # prediction horizon [s]
        ocp_mpc.solver_options.N_horizon = self.params.mpc_horizont
        ocp_mpc.solver_options.nlp_solver_stats_level = 1
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

        print("con_h_expr size:", ocp_mpc.model.con_h_expr.size1())
        print("lh size:", len(ocp_mpc.constraints.lh))
        print("uh size:", len(ocp_mpc.constraints.uh))

        acados_solver_mhe = AcadosOcpSolver(ocp_mpc, \
                                            json_file=ocp_mpc.solver_options.json_file, build=True, generate=True)
        self.generate_header()

    def generate_header(self):
        base_model = self.get_model()
        rwa_pos: int = base_model.state_length
        state_length = base_model.state_length
        param_length = base_model.param_length
        if (self.params.use_ddu_control):
            state_length = state_length + 1
        definitions = {
            "STATE_LENGTH": state_length,
            "PARAM_LENGTH": param_length,
            "RWA_POS": rwa_pos,
            "USE_DDU_CONTROL": int(self.params.use_ddu_control),
        }
        generate_header(self.generated_folder / self.model_name / "include/mpc_settings.h", \
                        definitions, f'MPC_SETTINGS_{self.model_name}')
