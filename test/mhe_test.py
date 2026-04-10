import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
import numpy as np
import pytest
from acados_template import AcadosOcp
from commom_utils.ode_system import check_system_ok, MHESyntheticDataGenerator
from commom_utils.systems import KinematicBycicle  # add other systems as needed
from mhe.mhe_base_model_interface import MheCogeGenerator
from mhe.params import MheParams
from mhe.mhe_utils import run_mhe_estimation

# ------------------------------------------------------------
# Configuration for different systems
# ------------------------------------------------------------
SYSTEM_CONFIGS = {
    "KinematicBicycle": {
        "system_class": KinematicBycicle,
        "system_kwargs": {"wheelbase": 2.65},
        "true_params": np.array([0.05, np.deg2rad(-0.3)]),  # [v, steering_angle]
        "initial_state": np.array([0.0]),                   # [x] or [position]
        "bounds_param": [np.deg2rad([-5, 5]), [-1, 1]],
        "bounds_state": [[-np.inf, np.inf]],
        "bounds_noise": [[-0.01, 0.01]],
        "state_prior_q0": np.diag([1.0]),
        "noise_penalty_w": np.eye(1) * 1e3,
        "measurements_residual_r": np.diag([1.0]),
        "fim_scaler": 0.2,
        "initial_params": np.array([0.02, np.deg2rad(2.3)])
    },
    # Add more systems here, e.g.:
    # "AnotherSystem": {...}
}

@pytest.fixture(params=SYSTEM_CONFIGS.keys())
def system_config(request):
    """Fixture that returns configuration for each system."""
    return SYSTEM_CONFIGS[request.param], request.param

def test_mhe_identification(system_config, tmp_path):
    """
    Test MHE parameter identification for a given system.
    Verifies that the estimated parameters converge to the true values.
    """
    config, system_name = system_config

    # Create system instance
    system = config["system_class"](**config.get("system_kwargs", {}))
    check_system_ok(system)

    # MHE parameters
    mhe_params = MheParams(
        dt=0.02,
        mhe_horizont=400,
        state_prior_q0=config["state_prior_q0"],
        noise_peanlty_w=config["noise_penalty_w"],
        measurements_residual_r=config["measurements_residual_r"],
        fim_scaler=config["fim_scaler"],
        bounds_noise=config["bounds_noise"],
        bounds_state=config["bounds_state"],
        bounds_param=config["bounds_param"],
    )

    # Generator for acados code
    class TestGenerator(MheCogeGenerator):
        def __init__(self):
            # Use temporary directory for generated code
            generated_dir = tmp_path / "generated" / system_name
            super().__init__(system, mhe_params, generated_dir, f"{system_name}_mhe")

        def modify_ocp_problem(self, ocp_mhe: AcadosOcp) -> AcadosOcp:
            ocp_mhe.solver_options.print_level = 0
            ocp_mhe.solver_options.nlp_solver_stats_level = 0
            ocp_mhe.solver_options.nlp_solver_max_iter = 150
            return ocp_mhe

    generator = TestGenerator()
    acados_solver = generator.generate_code()

    # Synthetic data generation
    data_gen = MHESyntheticDataGenerator(system, sigma=0.0)   # no noise for test

    t0 = 0.0
    T_f = mhe_params.dt * mhe_params.mhe_horizont
    N_meas = mhe_params.mhe_horizont
    overlap_points = int(N_meas * 0.5)
    num_windows = 30

    t_windows, u_windows, meas_windows, _ = data_gen.generate_sliding_windows_exact(
        c0=config["initial_state"],
        theta=config["true_params"],
        t0=t0,
        tf=T_f,
        num_windows=num_windows,
        n_measurement=N_meas,
        overlap_points=overlap_points
    )

    def get_window(i):
        return t_windows[i], u_windows[i], meas_windows[i]

    # Initial guess for parameters (slightly perturbed)
    initial_theta = config["initial_params"]

    # Run MHE estimation
    results = run_mhe_estimation(
        mhe_model=generator.get_model(),
        acados_solver_factory=acados_solver,
        get_window_func=get_window,
        overlap_points=overlap_points,
        initial_theta=initial_theta,
        mhe_params=mhe_params,
        num_windows=num_windows,
        R_inv=mhe_params.measurements_residual_r,
        forgetting_factor=0.01,
        compute_advanced_fim=True,
        plot=False
    )

    # Check that the last window's parameter estimate is close to true values
    final_theta_est = results[-1].param_est
    true_theta = config["true_params"]

    # Relative error tolerance (10% for now, can be tightened)
    rel_error = np.abs((final_theta_est - true_theta) / true_theta)
    assert np.all(rel_error < 1e-2), f"System {system_name}: final estimate {final_theta_est} differs from true {true_theta} by {rel_error}"
