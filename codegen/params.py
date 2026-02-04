from dataclasses import dataclass

import numpy as np


@dataclass
class CarParams:
    """Class for keeping track of an item in inventory."""
    wheelbase: float
    length: float
    rear_overhang: float
    u_max: float
    du_max: float
    ddu_max: float

    def __init__(self, wheelbase: float,
                length: float,
                rear_overhang: float,
                u_max: float,
                du_max: float,
                ddu_max: float):
        self.wheelbase = wheelbase
        self.length = length
        self.rear_overhang = rear_overhang
        self.u_max = u_max
        self.du_max = du_max
        self.ddu_max = ddu_max


@dataclass
class MpcParams:
    """Class for keeping track of an item in inventory."""
    car_params: CarParams
    use_dynamic_model: bool
    use_ddu_control: bool
    mpc_horizont: int
    n_delay: int
    ts: float
    r_dist: float
    r_ang: float
    r_w: float
    r_u: float
    r_u_diff: float
    r_ddu: float
    r_jerk: float
    a_comf: float
    jerk_max: float
    final_cost: float

    def __init__(self, car_params: CarParams,
                use_dynamic_model: bool,
                use_ddu_control: bool,
                mpc_horizont: int,
                n_delay: int,
                ts: float,
                r_dist: float,
                r_ang: float,
                r_w: float,
                r_u: float,
                r_du: float,
                r_ddu: float,
                r_jerk: float,
                a_comf: float,
                jerk_max: float,
                jerk_comf: float,
                final_cost: float):
        self.car_params = car_params
        self.use_dynamic_model = use_dynamic_model
        self.use_ddu_control = use_ddu_control
        self.mpc_horizont = mpc_horizont
        self.n_delay = n_delay
        self.ts = ts
        self.r_dist = r_dist
        self.r_ang = r_ang
        self.r_w = r_w
        self.r_u = r_u
        self.r_du = r_du
        self.r_ddu = r_ddu
        self.r_jerk = r_jerk
        self.a_comf = a_comf
        self.jerk_max = jerk_max
        self.jerk_comf = jerk_comf
        self.final_cost = final_cost


@dataclass
class MheParams:
    """Class for keeping track of an item in inventory."""
    dt: float
    mhe_horizont: int
    state_prior_q0: np.array
    params_prior_p0: np.array
    noise_peanlty_w: np.array
    measurements_residual_r: np.array
    wheelbase: float
    use_offset: bool
    delay: int

    def __init__(self, dt: float,
                mhe_horizont: float,
                state_prior_q0: np.array,
                params_prior_p0: np.array,
                noise_peanlty_w: np.array,
                measurements_residual_r: np.array,
                wheelbase: float,
                use_offset: bool,
                delay: int):
        self.dt = dt
        self.mhe_horizont = mhe_horizont
        self.state_prior_q0 = state_prior_q0
        self.params_prior_p0 = params_prior_p0
        self.noise_peanlty_w = noise_peanlty_w
        self.measurements_residual_r = measurements_residual_r
        self.wheelbase = wheelbase
        self.use_offset = use_offset
        self.delay = delay

    def print(self):
        print("dt: ", self.dt)
        print("mhe_horizont: ", self.mhe_horizont)
        print("noise_peanlty_w: ", self.noise_peanlty_w)
        print("state_prior_q0: ", self.state_prior_q0)
        print("params_prior_p0: ", self.params_prior_p0)
        print("measurements_residual_r: ", self.measurements_residual_r)
        print("wheelbase: ", self.wheelbase)
        print("use_offset: ", self.use_offset)
        print("delay: ", self.delay)
