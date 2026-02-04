
from abc import ABC, abstractmethod

from acados_template import AcadosModel
from casadi import SX, vertcat

from params import MheParams


class MheModel(ABC):
    @abstractmethod
    def continuous_dynamics(self, x, noise, p) -> SX:
        pass

    @abstractmethod
    def get_state(self) -> SX:
        pass


def make_continious_mhe_model(model_name: str, params: MheParams, model: MheModel) -> AcadosModel:
    """Continuous bicycle model without input delay for MHE."""
    # State variables.
    steering = SX.sym('steering')
    x = model.get_state()  # heading, params ..
    w_noise = SX.sym('w_noise')
    # Parameters
    vx = SX.sym('vx')
    p = vertcat(vx, steering)
    dx = model.continuous_dynamics(x, w_noise, p)
    # Create Acados model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = dx
    acados_model.x = x
    acados_model.u = w_noise
    acados_model.param_length = model.param_length
    acados_model.state_length = model.state_length
    acados_model.p = p
    acados_model.name = model_name
    return acados_model


def make_discrete_mhe_model(model_name: str, params: MheParams, model: MheModel) -> AcadosModel:
    """Continuous bicycle model without input delay for MHE."""
    # State variables.
    steering = SX.sym('steering')
    delayed_buf_u = SX.sym('delayed_u', params.delay)
    x = model.get_state()  # heading, params ..
    w_noise = SX.sym('w_noise')
    # Parameters
    vx = SX.sym('vx')
    Ts = params.dt
    # Continuous dynamics from base model
    delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else steering
    p = vertcat(vx, delayed_u)
    k1 = model.continuous_dynamics(x, w_noise, p)
    k2 = model.continuous_dynamics(x + 0.5 * Ts * k1, w_noise, p)
    k3 = model.continuous_dynamics(x + 0.5 * Ts * k2, w_noise, p)
    k4 = model.continuous_dynamics(x + Ts * k3, w_noise, p)

    x_next = x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    if (delayed_buf_u.shape[0] > 0):
        x = vertcat(x, delayed_buf_u)
        buf_next = vertcat(steering, delayed_buf_u[:-1])
        x_next = vertcat(x_next, buf_next)

    # Create Acados model
    acados_model = AcadosModel()
    acados_model.disc_dyn_expr = x_next
    acados_model.x = x
    acados_model.u = w_noise
    acados_model.param_length = model.param_length
    acados_model.state_length = model.state_length
    acados_model.p = vertcat(vx, steering)
    acados_model.name = model_name
    return acados_model
