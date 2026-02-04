import argparse
from pathlib import Path

import numpy as np
import yaml

import mhe.mhe_kinematic_model as mhe_kinematic_model
import sat_sda_config
from params import MheParams


def read_mhe_params(config_path) -> MheParams:
    ros_parameters = config_path["/**"]["ros__parameters"]
    mhe_params = ros_parameters["control"]
    steering_params = ros_parameters["steering"]
    chassis_params = ros_parameters["chassis"]

    mhe_horizont = mhe_params["mhe_horizont"]
    ts = mhe_params["mhe_sample_time_ms"] / 1000.0
    Q0_mhe = np.diag(mhe_params["mhe_Q0"])
    P0_mhe = np.diag(mhe_params["mhe_P0"])
    wheel_base = chassis_params["wheel_base"]
    state_noise_stds_mhe = np.array(mhe_params["mhe_state_noise_stds"])
    R_mhe = np.diag(mhe_params["mhe_R"])   #x
    Q_mhe = np.diag(1. / state_noise_stds_mhe**2) # w_noise
    use_offset = 1
    delay = 0  #is not used
    if (not use_offset):
        P0_mhe = P0_mhe[:1, :1]   #only GR

    mhe_params = MheParams(ts,
                              mhe_horizont,
                              Q0_mhe,
                              P0_mhe,
                              Q_mhe,
                              R_mhe,
                              wheel_base,
                              use_offset,
                              delay)

    return mhe_params


def main(agrs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, help="Input dir for generated files", required=True)
    parser.set_defaults(save_sep=True)
    args = parser.parse_args()
    generated_folder = Path(args.output_dir)
    rover_config_name = 'ceed'
    ceed_param_filename = sat_sda_config.get_config_path()  \
        / 'rover_type_specific' / f'{rover_config_name}.param.yaml'
    with open(ceed_param_filename, 'r') as file:
        ceed_param_file = yaml.safe_load(file)
        mhe_params = read_mhe_params(ceed_param_file)
        mhe_params.print()
        mhe_kinematic_model.set_ocp_problem(mhe_params, 'bicycle_mhe', generated_folder)


if __name__ == "__main__":
    main()
