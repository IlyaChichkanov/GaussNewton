from commom_utils.systems import *
import numpy as np
from jax import numpy as jnp

def get_input_signals_bycicle(t):
    w = 2.7
    steering = 0.8 * jnp.cos(t * 0.25 * w) * jnp.sin(w * t)
    v = 10.0
    return [v, steering]          # порядок: steering, vx (как ожидает LateralCarDynamic)

def harmonic(t):
    return [jnp.cos(0.3 * t) * jnp.sin(0.1 * t + np.pi / 2)]


SYSTEM_CONFIGS = {
    # ------------------------------------------------------------------
    # Модели, использовавшиеся в Gauss‑Newton (models_config)
    # ------------------------------------------------------------------
    "LotkaVoltera": {
        "class": LotkaVoltera,
        "args": [],
        "c0": np.array([6.0, 5.0]),
        "theta_true": np.array([1.2, 0.4, 0.3, 0.1]),
        "delta_theta": np.array([0.2, -0.11, 0.05, 0.01]) * 0.7 + (np.random.rand(4) - 0.5) * 0.05,
        "input_signal": None,                          # нет входа
        "observation": lambda state, theta, u: vertcat(state[0], state[1]),
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "LateralCarDynamic": {
        "class": LateralCarDynamic,
        "args": [2.5],                                 # wheelbase
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([3.90697911, -3.61844364, 11.46438743, 10.16318852]),
        "delta_theta": (np.random.rand(4) - 0.5) * 2,
        "input_signal": get_input_signals_bycicle,  # steering, vx
        "observation": lambda state, theta, u: vertcat(state[0], state[1]),
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "Attractor": {
        "class": Attractor,
        "args": [],
        "c0": np.array([-10.0, 10.0, 30.0]),
        "theta_true": np.array([10.0, 28.0, 8/3]),
        "delta_theta": (np.random.rand(3) - 0.5) * 10.0,
        "input_signal": None,
        "observation": lambda state, theta, u: state,  # все координаты
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "Pendulum": {
        "class": Pendulum,
        "args": [],
        "c0": np.array([0.0, np.pi, 0.0, 0.1]),
        "theta_true": np.array([10.0, 1.0, 1.0]),
        "delta_theta": np.array([4.0, 0.5, 0.3]),
        "input_signal": lambda t: [jnp.sin(t)],         # одномерный вход
        "observation": lambda state, theta, u: state[:3],
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "Delay": {                                         # соответствует GN‑модели "Delay"
        "class": DelaySystem,
        "args": [2],
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([0.4]),
        "delta_theta": np.array([0.2]),
        "input_signal": lambda t: [jnp.sin(t)],         # одномерный вход
        "observation": lambda state, theta, u: state,  # по умолчанию весь state
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },

    # ------------------------------------------------------------------
    # Модели, использовавшиеся только в MHE (configs), дополненные
    # ------------------------------------------------------------------
    "MassSpringDamper": {
        "class": MassSpringDamper,
        "args": [],
        "c0": np.array([1.0, 10.0]),
        "theta_true": np.array([3.0, 1.0]),
        "delta_theta": None,                           # не возмущаем
        "input_signal": None,
        "observation": lambda state, theta, u: state,
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "KinematicBycicle": {
        "class": KinematicBycicle,                     # модель из MHE (возможно, упрощённая)
        "args": [2.65],                                # wheelbase
        "c0": np.array([0.0]),                         # одномерное состояние? Уточните
        "theta_true": np.array([0.05, np.deg2rad(-1.0)]),
        "delta_theta": np.array([0.01, np.deg2rad(2.0)]),
        "input_signal": get_input_signals_bycicle,
        "get_initial_state": lambda y_meas, u, theta: y_meas[0:1],
    },
    "KinematicModelDelay": {
        "class": KinematicModelDelay,
        "args": [2.65, 2],                             # wheelbase, delay_order
        "c0": np.zeros(3),                             # [d, psi, ...]theta
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
        "delta_theta": None,
        "input_signal": None,
        "observation": lambda state, theta, u: state,
        "get_initial_state": lambda y_meas, u, theta: y_meas,
    },
    "DelaySystem": {                                   # MHE-вариант системы с задержкой
        "class": DelaySystem,
        "args": [2],
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([0.4]),
        "delta_theta": None,
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


def create_system(model_key: str):
    """
    Возвращает: system (объект ODESystem), c0 (np.array), theta_true (np.array),
                delta_theta (np.array или None).
    """
    cfg = SYSTEM_CONFIGS[model_key]

    class ConfiguredSystem(cfg["class"]):
        pass

    # observation (если не задано, оставляем стандартное поведение родителя)
    if "observation" in cfg and cfg["observation"] is not None:
        ConfiguredSystem.observation = lambda self, state, theta, u: cfg["observation"](state, theta, u)

    # input_signals (если None, оставляем стандартный get_input_signals -> [])
    if cfg.get("input_signal") is not None:
        ConfiguredSystem.get_input_signals = lambda self, t: cfg["input_signal"](t)

    # Дополнительный метод get_initial_state (для MHE, не мешает GN)
    if "get_initial_state" in cfg and cfg["get_initial_state"] is not None:
        ConfiguredSystem.get_initial_state = lambda self, y_meas, u, theta: cfg["get_initial_state"](y_meas, u, theta)
    else:
        ConfiguredSystem.get_initial_state = lambda self, y_meas, u, theta: y_meas

    system = ConfiguredSystem(*cfg["args"])
    return system, cfg["c0"].copy(), cfg["theta_true"].copy(), cfg.get("delta_theta")



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
    "KinematicBycicle": {
        "measurements_residual_r": np.diag([1.0]),
        "state_prior_q0": np.diag([1.0]),
        "noise_peanlty_w": np.eye(1) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01]],
        "bounds_state": [[-np.inf, np.inf]],
        "bounds_param": [np.deg2rad([-5, 5]), [-1, 1]],
    },
    "KinematicModelDelay": {
        "measurements_residual_r": np.diag([1.0, 3.0]),
        "state_prior_q0": np.eye(3),            # нулевая априорная точность по состоянию
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
    # Для моделей, которые используются только в GN (LotkaVoltera, LateralCarDynamic,
    # Attractor, Pendulum), MHE_CONFIGS не обязателен, но если вы захотите применить
    # MHE к ним позже, добавьте записи по аналогии.
}

from mhe.params import MheParams

def create_mhe_params(model_key: str, dt: float, mhe_horizont: int):
    """Создаёт объект MheParams на основе MHE_CONFIGS."""
    mhe_cfg = MHE_CONFIGS[model_key]
    return MheParams(
        dt=dt,
        mhe_horizont=mhe_horizont,
        state_prior_q0=mhe_cfg["state_prior_q0"],
        noise_peanlty_w=mhe_cfg["noise_peanlty_w"],
        measurements_residual_r=mhe_cfg["measurements_residual_r"],
        bounds_noise=mhe_cfg["bounds_noise"],
        bounds_state=mhe_cfg["bounds_state"],
        bounds_param=mhe_cfg["bounds_param"],
        fim_scaler=mhe_cfg["fim_scaler"],
    )