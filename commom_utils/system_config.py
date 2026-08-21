"""Ready-made configurations of the example systems: initial states, true
parameters, perturbations, and the MHE weights that go with them.
"""
import numpy as np
from casadi import vertcat
from jax import numpy as jnp

from commom_utils.systems import (Attractor, DelaySystem, Integrator,
                                  KinematicModel, KinematicModelDelay,
                                  LateralCarDynamic, LotkaVoltera,
                                  MassSpringDamper, OffsetEstimator, Pendulum)
from mhe.params import MheParams

def get_input_signals_bycicle(t):
    w = 2.7
    steering = 0.8 * jnp.cos(t * 0.25 * w) * jnp.sin(w * t)
    v = 10.0
    # jnp.where rather than `if t < 10`: time is traced inside jax odeint and
    # a Python comparison raises TracerBoolConversionError
    steering = jnp.where(t < 10.0, 0.0, steering)
    return [v, steering, 2.65]          # order: vx, steering, wheelbase

def harmonic(t):
    return [jnp.cos(0.3 * t) * jnp.sin(0.1 * t + np.pi / 2)]


# Seeded so that a module import always yields the same perturbations
_CFG_RNG = np.random.default_rng(0)


SYSTEM_CONFIGS = {
    # ------------------------------------------------------------------
    # Models used in the Gauss-Newton experiments
    # ------------------------------------------------------------------
    "LotkaVoltera": {
        "class": LotkaVoltera,
        "args": [],
        "c0": np.array([6.0, 5.0]),
        "theta_true": np.array([1.2, 0.4, 0.3, 0.1]),
        "delta_theta": np.array([0.2, -0.11, 0.05, 0.01]) * 0.7 + (_CFG_RNG.random(4) - 0.5) * 0.05,
        "input_signal": None,                          # no inputs
        "observation": lambda state, theta, u: vertcat(state[0], state[1]),
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "LateralCarDynamic": {
        "class": LateralCarDynamic,
        "args": [2.5],                                 # wheelbase
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([3.90697911, -3.61844364, 11.46438743, 10.16318852]),
        "delta_theta": (_CFG_RNG.random(4) - 0.5) * 2,
        "input_signal": get_input_signals_bycicle,  # vx, steering, wheelbase
        "observation": lambda state, theta, u: vertcat(state[0], state[1]),
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "Attractor": {
        "class": Attractor,
        "args": [],
        "c0": np.array([-10.0, 10.0, 30.0]),
        "theta_true": np.array([10.0, 28.0, 8/3]),
        "delta_theta": (_CFG_RNG.random(3) - 0.5) * 10.0,
        "input_signal": None,
        "observation": lambda state, theta, u: state,  # every coordinate
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "Pendulum": {
        "class": Pendulum,
        "args": [],
        "c0": np.array([0.0, np.pi, 0.0, 0.1]),
        "theta_true": np.array([10.0, 1.0, 1.0]),
        "delta_theta": np.array([4.0, 0.5, 0.3]),
        "input_signal": lambda t: [jnp.sin(t)],         # one input
        "observation": lambda state, theta, u: state[:3],
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },

    # ------------------------------------------------------------------
    # Models used in the MHE experiments
    # ------------------------------------------------------------------
    "MassSpringDamper": {
        "class": MassSpringDamper,
        "args": [],
        "c0": np.array([1.0, 10.0]),
        "theta_true": np.array([3.0, 1.0]),
        "delta_theta": None,                           # not perturbed
        "input_signal": None,
        "observation": lambda state, theta, u: state,
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "KinematicBycicle": {
        "class": KinematicModel,
        "args": [True],                                # use_offset
        "c0": np.array([0.0]),
        "theta_true": np.array([0.05, np.deg2rad(-0.1)]),
        "delta_theta": np.array([0.04, np.deg2rad(5.0)]),
        "input_signal": get_input_signals_bycicle,
        "get_initial_state": lambda y_meas, u, theta: y_meas[0:1],
    },
    "KinematicModelDelay": {
        "class": KinematicModelDelay,
        "args": [2.65, 2],                             # wheelbase, delay order
        "c0": np.zeros(3),                             # [psi, delay states]
        "theta_true": np.array([0.05, np.deg2rad(-1.0), 0.2]),
        "delta_theta": np.array([0.07, np.deg2rad(-1.0), 0.1]),
        "input_signal": get_input_signals_bycicle,
        "observation": None,
        "get_initial_state": lambda y_meas, u, theta: np.hstack((y_meas[0], u[1], 0)),
    },
    "Integrator": {
        "class": Integrator,
        "args": [],
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([1.0]),
        "delta_theta": np.array([0.1]),
        "input_signal": None,
        "observation": lambda state, theta, u: state,
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "DelaySystem": {
        "class": DelaySystem,
        "args": [2],
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([0.4]),
        "delta_theta": np.array([0.2]),
        "input_signal": harmonic,
        "observation": None,
        "get_initial_state": lambda y_meas, u, theta: np.hstack((u, 0)),
    },
    "OffsetEstimator": {
        "class": OffsetEstimator,
        "args": [2.65, 1],
        "c0": np.array([0.0]),
        "theta_true": np.array([0.4]),
        "delta_theta": np.array([-0.2]),
        "input_signal": get_input_signals_bycicle,
        "observation": lambda state, theta, u: state,
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
}


MHE_CONFIGS = {
    "LotkaVoltera": {
        "measurements_residual_r": np.diag([1.0, 1.0]),
        "state_prior_q0": np.diag([1.0, 1.0]),
        "noise_peanlty_w": np.eye(2) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01]] * 2,
        "bounds_state": [[-np.inf, np.inf]] * 2,
        "bounds_param": [[-2000, 2000]] * 4,
    },

    "MassSpringDamper": {
        "measurements_residual_r": np.diag([1.0, 1.0]),
        "state_prior_q0": np.diag([1.0, 1.0]),
        "noise_peanlty_w": np.eye(2) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01]] * 2,
        "bounds_state": [[-np.inf, np.inf]] * 2,
        "bounds_param": [[-2000, 2000]] * 2,
    },
    "LateralCarDynamic": {
        "measurements_residual_r": np.diag([1.0, 1.0]),
        "state_prior_q0": np.diag([1.0, 1.0]),
        "noise_peanlty_w": np.eye(2) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01], [-0.01, 0.01]],
        "bounds_state": [[-np.inf, np.inf], [-100, 100]],
        "bounds_param": [[-100, 100], [-100, 100], [-100, 100], [-100, 100]],
    },
    "KinematicBycicle": {
        "measurements_residual_r": np.diag([1.0, 1.0]),
        "state_prior_q0": np.diag([1.0]),
        "noise_peanlty_w": np.eye(1) * 1e3,
        "fim_scaler": 0.5,
        "bounds_noise": [[-0.01, 0.01]],
        "bounds_state": [[-np.inf, np.inf]],
        "bounds_param": [[0.01, 0.2], np.deg2rad([-5, 5])],
    },
    "KinematicModelDelay": {
        "measurements_residual_r": np.diag([1.0, 3.0]),
        "state_prior_q0": np.eye(3),
        "noise_peanlty_w": np.eye(3) * 1e1,
        "fim_scaler": 0.5,
        "bounds_noise": [[-1, 1]] * 3,
        "bounds_state": [[-50, 50]] * 3,
        "bounds_param": [[0.01, 0.5], np.deg2rad([-5, 5]), [0.02, 1.6]],
    },
    "Integrator": {
        "measurements_residual_r": np.diag([1.0]),
        "state_prior_q0": np.diag([1.0, 1.0]),
        "noise_peanlty_w": np.eye(2) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01]] * 2,
        "bounds_state": [[-1e5, 1e5]] * 2,
        "bounds_param": [[0, 20]],
    },
    "DelaySystem": {
        "measurements_residual_r": np.diag([1.0]),
        "state_prior_q0": np.diag([1, 1]),
        "noise_peanlty_w": np.eye(2)*10 ,
        "fim_scaler": 1.0,
        "bounds_noise": [[-0.1, 0.1]] * 2,
        "bounds_state": [[-1e5, 1e5]] * 2,
        "bounds_param": [[0, 0.7]],
    },
    "OffsetEstimator": {
        "measurements_residual_r": np.diag([1.0]),
        "state_prior_q0": np.zeros((1, 1)),
        "noise_peanlty_w": np.eye(1) * 1e3,
        "fim_scaler": 0.1,
        "bounds_noise": [[-0.01, 0.01]],
        "bounds_state": [[-1e5, 1e5]],
        "bounds_param": [[0, 20]],
    },
    # Models used only in the Gauss-Newton experiments (Attractor, Pendulum,
    # ...) need no MHE entry; add one by analogy if MHE is applied to them.
}


def create_system(cfg: dict):
    """Build a configured system -> (system, c0, theta_true, delta_theta)."""
    class ConfiguredSystem(cfg["class"]):
        pass

    # Fall back to the parent behaviour when a hook is not configured
    if "observation" in cfg and cfg["observation"] is not None:
        ConfiguredSystem.observation = lambda self, state, theta, u: cfg["observation"](state, theta, u)

    if cfg.get("input_signal") is not None:
        ConfiguredSystem.get_input_signals = lambda self, t: cfg["input_signal"](t)

    if "get_initial_state" in cfg and cfg["get_initial_state"] is not None:
        ConfiguredSystem.get_initial_state = lambda self, y_meas, u, theta: cfg["get_initial_state"](y_meas, u, theta)
    else:
        ConfiguredSystem.get_initial_state = lambda self, y_meas, u, theta: y_meas

    system = ConfiguredSystem(*cfg["args"])
    # delta_theta may be None, meaning "do not perturb"
    delta_theta = cfg.get("delta_theta")
    if delta_theta is not None:
        delta_theta = np.asarray(delta_theta).copy()
    return system, cfg["c0"].copy(), cfg["theta_true"].copy(), delta_theta


def create_mhe_params(mhe_cfg: dict, dt: float, mhe_horizont: int):
    """MheParams from an MHE_CONFIGS entry."""
    return MheParams(
        dt=dt,
        mhe_horizont=mhe_horizont,
        state_prior_q0=mhe_cfg["state_prior_q0"],
        noise_peanlty_w=mhe_cfg["noise_peanlty_w"],
        measurements_residual_r=mhe_cfg["measurements_residual_r"],
        bounds_noise=mhe_cfg["bounds_noise"],
        bounds_state=mhe_cfg["bounds_state"],
        bounds_param=mhe_cfg["bounds_param"],
        use_noise=0
    )