from abc import ABC, abstractmethod

from acados_template import AcadosModel
from casadi import SX, vertcat

from params import MpcParams


class MpcModel(ABC):
    @abstractmethod
    def continuous_dynamics(self, x, rwa, p) -> SX:
        pass

    @abstractmethod
    def get_state(self) -> SX:
        pass


def discrete_bicycle_model_du(params: MpcParams, model_name: str, model: MpcModel) -> AcadosModel:
    rwa = SX.sym('rwa')
    delayed_buf_u = SX.sym('delayed_u', params.n_delay)
    du = SX.sym('du')
    vx = SX.sym('vx')
    c = SX.sym('c')

    p = vertcat(vx, c)
    Ts = params.ts

    x = model.get_state()
    # RK4 integration
    delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else rwa
    k1 = model.continuous_dynamics(x, delayed_u, p)
    k2 = model.continuous_dynamics(x + 0.5 * Ts * k1, delayed_u, p)
    k3 = model.continuous_dynamics(x + 0.5 * Ts * k2, delayed_u, p)
    k4 = model.continuous_dynamics(x + Ts * k3, delayed_u, p)
    x_next = x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    x = vertcat(x, rwa)
    rwa_newt = rwa + du * Ts
    x_next = vertcat(x_next, rwa_newt)

    delay_dynamics = []
    for i in range(params.n_delay):
        if i == 0:
            delay_dynamics.append(rwa)
        else:
            delay_dynamics.append(delayed_buf_u[i - 1])

    x_next = vertcat(x_next, *delay_dynamics)
    x = vertcat(x, delayed_buf_u)

    model = AcadosModel()
    model.disc_dyn_expr = x_next
    model.x = x
    model.u = du
    model.p = p
    model.name = model_name
    return model


def discrete_bicycle_model_ddu(params: MpcParams, model_name: str, model: MpcModel) -> AcadosModel:
    rwa = SX.sym('rwa')
    delayed_buf_u = SX.sym('delayed_u', params.n_delay)
    du = SX.sym('du')
    ddu = SX.sym('ddu')
    vx = SX.sym('vx')
    c = SX.sym('c')

    p = vertcat(vx, c)
    Ts = params.ts

    x = model.get_state()
    # RK4 integration
    delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else rwa
    k1 = model.continuous_dynamics(x, delayed_u, p)
    k2 = model.continuous_dynamics(x + 0.5 * Ts * k1, delayed_u, p)
    k3 = model.continuous_dynamics(x + 0.5 * Ts * k2, delayed_u, p)
    k4 = model.continuous_dynamics(x + Ts * k3, delayed_u, p)
    x_next = x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    x = vertcat(x, rwa, du)
    rwa_newt = rwa + du * Ts
    du_next = du + ddu * Ts
    x_next = vertcat(x_next, rwa_newt, du_next)

    delay_dynamics = []
    for i in range(params.n_delay):
        if i == 0:
            delay_dynamics.append(rwa)
        else:
            delay_dynamics.append(delayed_buf_u[i - 1])

    x_next = vertcat(x_next, *delay_dynamics)
    x = vertcat(x, delayed_buf_u)

    model = AcadosModel()
    model.disc_dyn_expr = x_next
    model.x = x
    model.u = ddu
    model.p = p
    model.name = model_name
    return model


def continuous_bicycle_model_du(params: MpcParams, model_name: str, model: MpcModel) -> AcadosModel:
    """Continuous bicycle model without input delay for MPC."""

    # State variables
    x_base = model.get_state()  # tau, psi ..
    rwa = SX.sym('rwa')
    x = vertcat(x_base, rwa)

    # Control input
    du = SX.sym('du')

    # Parameters
    vx = SX.sym('vx')
    c = SX.sym('c')
    p = vertcat(vx, c)

    # Continuous dynamics from base model
    base_dynamics = model.continuous_dynamics(x_base, rwa, p)

    # Full dynamics with rwa integration
    dx_base = base_dynamics
    drwa = du

    # Combine all dynamics
    dx = vertcat(dx_base, drwa)

    # Create Acados model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = dx
    acados_model.x = x
    acados_model.u = du
    acados_model.p = p
    acados_model.name = model_name

    return acados_model
