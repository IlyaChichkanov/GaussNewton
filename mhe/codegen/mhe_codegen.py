# ruff: noqa: I001
import argparse
import os
import pathlib
from pathlib import Path

import numpy as np
import yaml

import mhe.mhe_kinematic_model as mhe_kinematic_model
import mhe.mhe_dynamic_model as mhe_dynamic_model
from params import MheParams

from enum import Enum


class MheModelType(Enum):
    Kinematic = 1
    KinematicOffset = 2
    Dynamic = 3


def read_mhe_params(config_path, model_type: MheModelType) -> MheParams:
    ros_parameters = config_path["/**"]["ros__parameters"]
    chassis_params = ros_parameters["chassis"]
    wheel_base = chassis_params["wheel_base"]

    # Common parameters (could be read from config)
    dt = 10.0 / 1000.0          # 20 ms
    mhe_horizont = 500
    fim_scaler = 1.0

    # Default flags
    use_offset = False
    use_only_offset = False

    if model_type == MheModelType.Dynamic:
        # 2 states, 4 parameters
        measurements_residual_r = np.diag([0.1, 1.0])
        state_prior_q0 = np.eye(2)
        noise_peanlty_w = np.diag([1e3, 1e3])
        params_prior_p0 = np.eye(3) * 0.1                 #   (placeholder)

        # measurements_residual_r = np.diag([0.1, 1.0, 1.0])
        # state_prior_q0 = np.eye(3)
        # noise_peanlty_w = np.diag([1e3, 1e3, 1e3])
        # params_prior_p0 = np.eye(3) * 0.1                 #   (placeholder)
        
    elif model_type == MheModelType.Kinematic:
        # 1 state, 1 or 2 parameters (GR and optional offset)
        use_offset = True   # assume offset is used; adjust later
        measurements_residual_r = np.diag([1, 0.0])                 # R: 1x1
        state_prior_q0 = np.diag([1, 0.001])                          # Q0: 1x1
        noise_peanlty_w = np.eye(2) * 1e3                    # Q: 1x1
        if use_offset:
            params_prior_p0 = np.eye(2)                     # P0: 2x2 (GR, offset)
        else:
            params_prior_p0 = np.eye(1)                     # P0: 1x1 (only GR)
        fim_scaler = 0.2

    elif model_type == MheModelType.KinematicOffset:
        # 1 state, 1 parameter (only offset)
        measurements_residual_r = np.eye(1)
        state_prior_q0 = np.eye(1)
        noise_peanlty_w = np.eye(1) * 1e3
        params_prior_p0 = np.eye(1)                         # only offset
        fim_scaler = 0.2
        use_only_offset = True

    else:
        raise TypeError(f"Unknown model type: {model_type}")

    return MheParams(
        dt=dt,
        mhe_horizont=mhe_horizont,
        state_prior_q0=state_prior_q0,
        params_prior_p0=params_prior_p0,
        noise_peanlty_w=noise_peanlty_w,
        measurements_residual_r=measurements_residual_r,
        fim_scaler=fim_scaler,
        wheelbase=wheel_base,
        use_offset=use_offset,
        use_only_offset=use_only_offset
    )


def main(agrs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, help="Input dir for generated files", required=True)
    parser.set_defaults(save_sep=True)
    args = parser.parse_args()
    generated_folder = Path(args.output_dir)
    rover_config_name = 'ceed'
    ceed_param_filename = pathlib.Path(os.getenv("CONFIG_PATH", "sda_config"))  \
        / 'rover_type_specific' / f'{rover_config_name}.param.yaml'

    with open(ceed_param_filename, 'r') as file:
        ceed_param_file = yaml.safe_load(file)
        print(f'PARAMETERS_____________________________{"Offset MHE model"}_______________________________________')
        mhe_params = read_mhe_params(ceed_param_file, MheModelType.KinematicOffset)
        mhe_params.print()
        generator = mhe_kinematic_model.KinematicMheCodegenerator(mhe_params, generated_folder, 'offset_mhe')
        generator.generate_code()
        print(f'PARAMETERS_____________________________{"Kinematic MHE model"}_______________________________________')
        mhe_params = read_mhe_params(ceed_param_file, MheModelType.Kinematic)
        mhe_params.print()
        generator = mhe_kinematic_model.KinematicMheCodegenerator(mhe_params, generated_folder, 'bicycle_mhe')
        generator.generate_code()
        print(f'PARAMETERS_____________________________{"Dynamic MHE model"}_______________________________________')
        mhe_params = read_mhe_params(ceed_param_file, MheModelType.Dynamic)
        mhe_params.print()
        generator = mhe_dynamic_model.DynamicMheCodegenerator(mhe_params, generated_folder, 'dynamic_mhe')
        generator.generate_code()

    file_path = generated_folder / "codegen.txt"
    content = "Hello, this is a new file for checking generation is completed"

    with open(file_path, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    main()
