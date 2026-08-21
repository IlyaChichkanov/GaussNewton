"""Example ODE models. Each one only declares its sizes and its symbolic
right-hand side, observation and input signals; see docs/api-reference.md.
"""
import casadi as ca
from casadi import vertcat
from jax import numpy as jnp

from commom_utils.ode_system import ODESystem


class Pendulum(ODESystem):
    """Cart-pole. State [x, phi, v, dphi], parameters [M, m, length]."""

    def __init__(self, order=1):
        super().__init__(nx=4, n_theta=3, nu=1)

    def get_derivative(self, state, theta, u):
        g = 10
        M, m, length = theta[0], theta[1], theta[2]
        phi, v1, dphi = state[1], state[2], state[3]
        F = u[0]
        denominator = M + m - m * ca.cos(phi) * ca.cos(phi)
        return vertcat(
            v1,
            dphi,
            (-m * length * ca.sin(phi) * dphi * dphi
             + m * g * ca.cos(phi) * ca.sin(phi) + F) / denominator,
            (-m * length * ca.cos(phi) * ca.sin(phi) * dphi * dphi
             + F * ca.cos(phi) + (M + m) * g * ca.sin(phi))
            / (length * denominator))

    def observation(self, state, theta, u):
        x1, phi = state[0], state[1]
        return vertcat(x1, phi)

    def get_input_signals(self, t):
        return [jnp.sin(0.2 * t)]


class DelayOffset(ODESystem):
    """DelaySystem plus a constant output offset. Parameters [tau_d, offset]."""

    def __init__(self, order=2):
        self.order = order
        self.delay = DelaySystem(order=order)
        super().__init__(nx=self.delay.nx, n_theta=1 + self.delay.n_theta, nu=1)

    def get_derivative(self, state, theta, u):
        return self.delay.get_derivative(state, theta, u)

    def observation(self, state, theta, u):
        offset = theta[1]
        return self.delay.observation(state, theta[:1], u) + offset


class DelaySystem(ODESystem):
    """Pade approximation of a pure delay tau; order 1 or 2."""

    def __init__(self, order=1):
        self.order = order
        if order == 1:
            super().__init__(nx=1, n_theta=1, nu=1)
        else:
            super().__init__(nx=2, n_theta=1, nu=1)

    def get_derivative(self, state, theta, u):
        tau = theta[0]
        tau_safe = ca.fmax(tau, 1e-6)
        if self.order == 1:
            x = state[0]
            return ca.vertcat((2.0 / tau_safe) * (u - x))
        x1, x2 = state[0], state[1]
        dx1 = x2
        dx2 = (-(12.0 / tau_safe**2) * x1 - (6.0 / tau_safe) * x2
               + (12.0 / tau_safe**2) * u)
        return ca.vertcat(dx1, dx2)

    def observation(self, state, theta, u):
        tau_safe = ca.fmax(theta[0], 1e-6)
        if self.order == 1:
            return 2 * u - state[0]
        return u - tau_safe * state[1]


class LateralCarDynamic(ODESystem):
    """Linear bicycle lateral dynamics. State [vy, wz], inputs [vx, steering]."""

    GEAR_RATIO = 20

    def __init__(self, wheelbase):
        super().__init__(2, 4, 2)
        self.wheelbase = wheelbase

    def get_derivative(self, state, theta, u):
        vy, wz = state[0], state[1]
        a0, a1, b0, b1 = theta[0], theta[1], theta[2], theta[3]
        vx, steering = u[0], u[1]
        alpha_f = (vy + self.wheelbase * wz) / vx
        alpha_r = vy / vx
        rwa = steering / self.GEAR_RATIO
        vy_dot = a0 * (rwa - alpha_f) + a1 * alpha_r - vx * wz
        wz_dot = b0 * (rwa - alpha_f) + b1 * alpha_r
        return vertcat(vy_dot, wz_dot)

    def observation(self, state, theta, u):
        return state


class LotkaVoltera(ODESystem):
    """Predator-prey. State [x, y], parameters [alpha, beta, gamma, delta]."""

    def __init__(self):
        super().__init__(2, 4, 0)

    def get_derivative(self, state, theta, u):
        x, y = state[0], state[1]
        alpha, beta, gamma, delta = theta[0], theta[1], theta[2], theta[3]
        return vertcat(alpha * x - beta * x * y,
                       delta * x * y - gamma * y)

    def get_input_signals(self, t):
        return []

    def observation(self, state, theta, u):
        return vertcat(state[0], state[1])


class Attractor(ODESystem):
    """Lorenz attractor. State [x, y, z], parameters [sigma, rho, beta]."""

    def __init__(self):
        super().__init__(3, 3, 0)

    def get_derivative(self, state, theta, u):
        x, y, z = state[0], state[1], state[2]
        alpha, beta, gamma = theta[0], theta[1], theta[2]
        return vertcat(alpha * (y - x),
                       x * (beta - z) - y,
                       x * y - gamma * z)

    def observation(self, state, theta, u):
        return vertcat(state[0], state[1], state[2])


class OscillatorModel(ODESystem):
    """Damped oscillator. Parameters [omega, zeta]."""

    def __init__(self):
        super().__init__(nx=2, nu=0, n_theta=2)

    def get_derivative(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        omega, zeta = params[0], params[1]
        return vertcat(x2, -omega**2 * x1 - 2 * zeta * omega * x2)

    def observation(self, state, theta, u):
        return state


class MassSpringDamper(ODESystem):
    """Mass-spring-damper driven by one input. Parameters [k, c]."""

    def __init__(self, m=1):
        self.mass = m
        super().__init__(nx=2, nu=1, n_theta=2)

    def get_derivative(self, state, params, input_signals):
        x1, x2 = state[0], state[1]
        k, c = params[0], params[1]
        u = input_signals[0] if input_signals.shape[0] > 0 else 0.0
        return vertcat(x2, (u - k * x1 - c * x2) / self.mass)

    def get_input_signals(self, t):
        # jnp.where rather than `if`: t can be a float, a jax tracer inside
        # odeint, or an array of times in the collocation path
        w = 0.7
        u = 0.8 * jnp.cos(t * 0.25 * w) * jnp.sin(w * t)
        return [jnp.where(t < 1.0, 0.0, u)]


class KinematicBycicleErrors(ODESystem):
    """Path-tracking errors [d, psi] of a kinematic bicycle. Parameters [vx, curvature]."""

    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        super().__init__(nx=2, nu=1, n_theta=2)

    def get_derivative(self, state, params, input_signals):
        d, psi = state[0], state[1]
        rwa = input_signals[0]
        vx, c = params[0], params[1]
        dd = vx * ca.sin(psi)
        dpsi = vx * ca.tan(rwa) / self.wheelbase - vx * c * ca.cos(psi) / (1 - c * d)
        return ca.vertcat(dd, dpsi)


class KinematicModel(ODESystem):
    """Heading of a kinematic bicycle; identifies the steering gear ratio.

    Inputs [vx, steering, wheelbase]; parameters [GR] or [GR, offset].
    """

    def __init__(self, use_offset: bool):
        self.use_offset = use_offset
        super().__init__(nx=1, n_theta=2 if use_offset else 1, nu=3)

    def _road_wheel_angle(self, params, input_signals):
        GR = params[0]
        offset = params[1] if self.use_offset else 0
        return GR * input_signals[1] - offset

    def get_derivative(self, state, params, input_signals):
        vx = input_signals[0]
        wheelbase = input_signals[2]
        rwa = self._road_wheel_angle(params, input_signals)
        return ca.vertcat(vx * ca.tan(rwa) / wheelbase)

    def observation(self, state, params, input_signals):
        v = input_signals[0]
        wheelbase = input_signals[2]
        rwa = self._road_wheel_angle(params, input_signals)
        return vertcat(state, v * ca.tan(rwa) / wheelbase)


class OffsetEstimator(ODESystem):
    """Steering offset of a kinematic bicycle with a known gear ratio."""

    def __init__(self, wheelbase, gear_ratio):
        self.wheelbase = wheelbase
        self.GR = 1 / gear_ratio
        super().__init__(nx=1, nu=2, n_theta=1)

    def get_derivative(self, state, params, input_signals):
        offset = params[0]
        v, steering = input_signals[0], input_signals[1]
        rwa = self.GR * steering + offset
        return v * ca.tan(rwa) / self.wheelbase

    def observation(self, state, params, input_signals):
        offset = params[0]
        v, steering = input_signals[0], input_signals[1]
        rwa = self.GR * steering + offset
        return v * ca.tan(rwa) / self.wheelbase


class KinematicBycicleActuator(ODESystem):
    """Kinematic bicycle with second-order steering actuator dynamics.

    State [psi, delta, delta_dot]; only the heading is measured.
    """

    def __init__(self, wheelbase, kp=80.9, kv=80.61):
        super().__init__(nx=3, nu=2, n_theta=2)
        self.wheelbase = wheelbase
        self.kp = kp
        self.kv = kv

    def get_derivative(self, state, params, input_signals):
        delta, delta_dot = state[1], state[2]
        GR, offset = params[0], params[1]
        vx, steering = input_signals[0], input_signals[1]
        dpsi = vx * ca.tan(delta) / self.wheelbase
        rwa = GR * steering + offset
        ddelta_dot = self.kp * (rwa - delta) - self.kv * delta_dot
        return ca.vertcat(dpsi, delta_dot, ddelta_dot)

    def observation(self, state, theta, u):
        return state[0]


class KinematicModelDelay(ODESystem):
    """Kinematic bicycle whose steering command passes through a delay.

    State [psi, delay states]; parameters [GR, offset, tau_d].
    """

    def __init__(self, wheelbase: float, order: int):
        self.wheelbase = wheelbase
        self.delay = DelaySystem(order=order)
        super().__init__(nx=1 + self.delay.nx,
                         n_theta=2 + self.delay.n_theta, nu=2)

    def get_derivative(self, state, params, input_signals):
        state_delay = state[1:]
        tau_d = params[2]
        steering_cmd = input_signals[1]
        dx_delay = self.delay.get_derivative(state_delay, [tau_d], steering_cmd)
        dpsi = self.bycicle_dynamics(state, params, input_signals)
        return ca.vertcat(dpsi, dx_delay)

    def bycicle_dynamics(self, state, params, input_signals):
        """Yaw rate from the delayed steering command."""
        GR, offset, tau_d = params[0], params[1], params[2]
        state_delay = state[1:]
        vx, steering_cmd = input_signals[0], input_signals[1]
        steering_cmd_delayed = self.delay.observation(state_delay, [tau_d],
                                                      steering_cmd)
        rwa = GR * steering_cmd_delayed + offset
        return vx * rwa / self.wheelbase

    def observation(self, state, params, input_signals):
        psi = state[0]
        w = self.bycicle_dynamics(state, params, input_signals)
        return vertcat(psi, w)


class RosenzweigMacArthur(ODESystem):
    """Predator-prey with a type III functional response.

    Parameters [r, K, a, h, e, m].
    """

    def __init__(self):
        super().__init__(nx=2, nu=0, n_theta=6)

    def get_derivative(self, state, params, input_signals):
        x, y = state[0], state[1]
        r, K, a, h, e, m = (params[0], params[1], params[2],
                            params[3], params[4], params[5])
        dx = r * x * (1 - x / K) - (a * x**2 * y) / (1 + a * h * x**2)
        dy = e * (a * x**2 * y) / (1 + a * h * x**2) - m * y
        return vertcat(dx, dy)


class Quadrotor2D(ODESystem):
    """Planar quadrotor in the X-Z plane.

    State [x, z, phi, x_dot, z_dot, phi_dot]; the input is the thrust
    difference between the rotors. Only the thrust coefficient k_T is
    identified; mass, inertia and the moment coefficient are fixed.
    """

    def __init__(self):
        super().__init__(nx=6, nu=1, n_theta=1)

    def get_derivative(self, state, params, input_signals):
        phi, x_dot, z_dot, phi_dot = state[2], state[3], state[4], state[5]
        k_T = params[0]
        J, m, k_M = 10, 1, 1
        u = input_signals[0]
        g = 10
        F_total = k_T * u
        return vertcat(x_dot, z_dot, phi_dot,
                       -(F_total / m) * ca.sin(phi),
                       (F_total / m) * ca.cos(phi) - g,
                       (k_M / J) * u)

    def get_input_signals(self, t):
        # jnp keeps the signal traceable in jax mode (see MassSpringDamper)
        return [0.5 * jnp.sin(0.05 * t) + t]


class Integrator(ODESystem):
    """Double integrator with an unknown gain k; only the position is measured."""

    def __init__(self):
        super().__init__(nx=2, nu=1, n_theta=1)

    def get_derivative(self, state, params, input_signals):
        v = state[1]
        k = params[0]
        u = input_signals[0]
        return vertcat(v, k * u)

    def get_input_signals(self, t):
        # jnp keeps the signal traceable in jax mode (see MassSpringDamper)
        return [0.5 * jnp.sin(0.5 * t) + t * 0.001 * jnp.sin(t)]

    def observation(self, state, theta, u):
        return state[0]
