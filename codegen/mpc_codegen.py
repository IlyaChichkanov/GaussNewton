import argparse
from pathlib import Path

import numpy as np
import yaml

import mpc.mpc_dynamic_model as mpc_dynamic_model
import mpc.mpc_kinematic_model as mpc_kinematic_model
import sat_sda_config
from params import CarParams, MpcParams


def read_mpc_params(config_path: Path) -> MpcParams:
    ros_parameters = config_path["/**"]["ros__parameters"]
    mpc_params = ros_parameters["control"]
    steering_params = ros_parameters["steering"]
    chassis_params = ros_parameters["chassis"]

    mpc_horizont = mpc_params["mpc_horizont"]
    ts = mpc_params["mpc_sample_time_ms"] / 1000.0
    n_delay = mpc_params["mpc_delay"]

    steering_gear_ratio = chassis_params["steering_gear_ratio"]
    wheel_base = chassis_params["wheel_base"]
    length = chassis_params["length"]
    rear_overhang = chassis_params["rear_overhang"]
    max_road_wheel_angle = steering_params["max_wheel_angle"] / steering_gear_ratio
    max_steering_wheel_angle_rate = steering_params["max_steering_wheel_rotation_speed"]

    u_max = np.tan(np.deg2rad(max_road_wheel_angle))
    du_max = np.deg2rad(max_steering_wheel_angle_rate / steering_gear_ratio)
    ddu_max = mpc_params["ddu_max"]

    # Cost Penalties
    r_dist = (mpc_params["r_dist"] / wheel_base)
    r_ang = (np.rad2deg(mpc_params["r_ang"]) / wheel_base)
    r_w = np.rad2deg(mpc_params["r_w"])
    r_u = mpc_params["r_u"]
    r_du = mpc_params["r_du"]
    r_ddu = mpc_params["r_ddu"]
    r_jerk = mpc_params["r_jerk"]
    a_comf = 1.5
    jerk_max = mpc_params["jerk_max"]
    jerk_comf = mpc_params["jerk_comf"]
    use_dynamic_model = mpc_params["use_dynamic_model"]
    use_ddu_control = mpc_params["use_ddu_control"]

    if (not use_dynamic_model):
        u_max = u_max / wheel_base
        du_max = du_max / wheel_base
        ddu_max = ddu_max / wheel_base

    final_cost = 2

    car_params = CarParams(wheel_base,
                              length,
                              rear_overhang,
                              u_max,
                              du_max,
                              ddu_max)

    params = MpcParams(car_params,
                        use_dynamic_model,
                        use_ddu_control,
                        mpc_horizont,
                        n_delay,
                        ts,
                        r_dist,
                        r_ang,
                        r_w,
                        r_u,
                        r_du,
                        r_ddu,
                        r_jerk,
                        a_comf,
                        jerk_max,
                        jerk_comf,
                        final_cost)
    return params


def print_params(cfg: MpcParams):
    print("use_dynamic_model: ", cfg.use_dynamic_model)
    print("use_ddu_control: ", cfg.use_ddu_control)
    print("mpc_horizont: ", cfg.mpc_horizont)
    print("n_delay: ", cfg.n_delay)
    print("ts", cfg.ts)
    print("length: ", cfg.car_params.length)
    print("rear_overhang: ", cfg.car_params.rear_overhang)
    print("wheelbase: ", cfg.car_params.wheelbase)
    print("u_max: ", cfg.car_params.u_max)
    print("du_max: ", cfg.car_params.du_max)
    print("ddu_max: ", cfg.car_params.ddu_max)
    print("jerk_comf: ", cfg.jerk_comf)
    print("r_dist: ", cfg.r_dist)
    print("r_ang: ", cfg.r_ang)
    print("r_w: ", cfg.r_w)
    print("r_u: ", cfg.r_u)
    print("r_du: ", cfg.r_du)
    print("r_ddu: ", cfg.r_ddu)
    print("r_jerk: ", cfg.r_jerk)


def main(agrs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, help="Input dir for generated files", required=True)

    parser.set_defaults(save_sep=True)
    args = parser.parse_args()
    generated_folder = Path(args.output_dir)

    ROVERS = [("ceed", "ceed"), ("cargo", "cargo"), ("shuttle_poc1", "shuttle"), ("shuttle_poc2", "shuttle")]
    for rover_model_name, rover_config_name in ROVERS:
        print(f'PARAMETERS_____________________________{rover_config_name}_______________________________________')
        ceed_param_filename = sat_sda_config.get_config_path()  \
            / 'rover_type_specific' / f'{rover_config_name}.param.yaml'
        with open(ceed_param_filename, 'r') as file:
            ceed_param_file = yaml.safe_load(file)
            ceed_params = read_mpc_params(ceed_param_file)
        print_params(ceed_params)
        if (ceed_params.use_dynamic_model):
            mpc_dynamic_model.set_ocp_problem(ceed_params, f'{rover_model_name}_model', generated_folder)
        else:
            mpc_kinematic_model.set_ocp_problem(ceed_params, f'{rover_model_name}_model', generated_folder)


if __name__ == "__main__":
    main()
