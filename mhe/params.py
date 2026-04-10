from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MheParams:
    dt: float
    mhe_horizont: int
    state_prior_q0: np.ndarray
    noise_peanlty_w: np.ndarray
    measurements_residual_r: np.ndarray
    bounds_param: np.ndarray
    bounds_state: np.ndarray
    bounds_noise: np.ndarray
    fim_scaler: float


    def __post_init__(self):
        self.check()

    def check(self):
        pass


    def print(self):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            print(f"{field}: {value}")
