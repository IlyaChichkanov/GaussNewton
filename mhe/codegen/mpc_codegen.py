import argparse
import os
import pathlib
from pathlib import Path

import numpy as np
import yaml

import mpc.mpc_dynamic_model as mpc_dynamic_model
import mpc.mpc_kinematic_model as mpc_kinematic_model

#import sat_sda_config
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

    u_max = np.deg2rad(max_road_wheel_angle)
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
    jerk_comf = mpc_params["jerk_comf"]
    jerk_max = 5
    a_comf_max = 5.0
    use_dynamic_model = mpc_params["use_dynamic_model"]
    use_ddu_control = mpc_params["use_ddu_control"]

    final_cost = 2

    car_params = CarParams(
        gear_ratio=steering_gear_ratio,
        wheelbase=wheel_base,
        length=length,
        rear_overhang=rear_overhang,
        u_max=u_max,
        du_max=du_max,
        ddu_max=ddu_max
    )

    params = MpcParams(
        car_params=car_params,
        use_dynamic_model=use_dynamic_model,
        use_ddu_control=1,
        mpc_horizont=mpc_horizont,
        n_delay=n_delay,
        ts=ts,
        r_dist=r_dist,
        r_ang=r_ang,
        r_w=r_w,
        r_u=r_u,
        r_du=r_du,
        r_ddu=r_ddu,
        r_jerk=r_jerk,
        jerk_comf=jerk_comf,
        jerk_max=jerk_max,
        a_comf_max=a_comf_max,
        final_cost=final_cost
    )
    return params


def main(agrs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, help="Input dir for generated files", required=True)

    parser.set_defaults(save_sep=True)
    args = parser.parse_args()
    generated_folder = Path(args.output_dir)

    ROVERS = [("ceed", "ceed"), ("cargo", "cargo"), ("shuttle_poc1", "shuttle"), ("shuttle_poc2", "shuttle")]
    for rover_model_name, rover_config_name in ROVERS:
        print(f'PARAMETERS_____________________________{rover_model_name}_______________________________________')
        rover_param_filename = pathlib.Path(os.getenv("CONFIG_PATH", "sda_config")) \
            / 'rover_type_specific' / f'{rover_config_name}.param.yaml'
        with open(rover_param_filename, 'r') as file:
            param_file = yaml.safe_load(file)
            params = read_mpc_params(param_file)
            params.print()

        if (params.use_dynamic_model):
            code_generator = \
                mpc_dynamic_model.DynamicMheCodegenerator(params, generated_folder, f'{rover_model_name}_model')
            code_generator.generate_code()
        else:
            code_generator =\
                  mpc_kinematic_model.KinematicMheCodegenerator(params, generated_folder, f'{rover_model_name}_model')
            code_generator.generate_code()


if __name__ == "__main__":
    main()
