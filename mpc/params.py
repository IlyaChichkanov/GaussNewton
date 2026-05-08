from dataclasses import asdict, dataclass

import numpy as np

@dataclass
class CarParams:
    wheelbase: float
    gear_ratio: float
    length: float
    rear_overhang: float
    u_max: float
    du_max: float
    ddu_max: float


@dataclass
class MpcParams:
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
    r_du: float
    r_ddu: float
    r_jerk: float
    jerk_comf: float
    jerk_max: float
    a_comf_max: float
    final_cost: float

    def print(self):
        data = asdict(self)
        print("MpcParams:")
        for key, value in data.items():
            if key == "car_params":
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

