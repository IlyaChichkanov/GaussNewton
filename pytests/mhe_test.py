import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
import numpy as np
import pytest
from acados_template import AcadosOcp
from commom_utils.ode_system import check_system_ok, MHESyntheticDataGenerator
from commom_utils.systems import KinematicBycicle, DelaySystem  # add other systems as needed
from commom_utils.system_config import create_system, create_mhe_params
from mhe.mhe_base_model_interface import MheCogeGenerator
from mhe.params import MheParams
from mhe.mhe_utils import run_mhe_estimation, reset_mhe_solver

# ------------------------------------------------------------
# Configuration for different systems
# ------------------------------------------------------------

def get_input_signals_bycicle(t):
    import math
    w = 0.7
    steering = 0.8 * math.cos(t * 0.25 * w) * math.sin(w * t)
    if t < 25:
        steering = 0
    v = 10.0
    return [v, steering]

def harmonic(t):
    u  = np.sin(0.3*t)
    return [u]


SYSTEM_CONFIGS = {
    "DelaySystem": {                                         # соответствует GN‑модели "Delay"
        "class": DelaySystem,
        "args": [2],
        "c0": np.array([0.0, 0.0]),
        "theta_true": np.array([0.4]),
        "delta_theta": np.array([0.2]),
        "input_signal": lambda t: harmonic(t),         # одномерный вход
        #"observation": lambda state, theta, u: state,  # по умолчанию весь state
        "get_initial_state": lambda y_meas, u, theta: np.hstack((u, 0))
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

}


MHE_CONFIGS = {
    "KinematicBycicle": {
        "measurements_residual_r": np.diag([1.0, 1.0]),
        "state_prior_q0": np.diag([1.0]),
        "noise_peanlty_w": np.eye(1) * 1e3,
        "fim_scaler": 0.2,
        "bounds_noise": [[-0.01, 0.01]],
        "bounds_state": [[-np.inf, np.inf]],
        "bounds_param": [np.deg2rad([-5, 5]), [-1, 1]],
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
    # Для моделей, которые используются только в GN (LotkaVoltera, LateralCarDynamic,
    # Attractor, Pendulum), MHE_CONFIGS не обязателен, но если вы захотите применить
    # MHE к ним позже, добавьте записи по аналогии.
}


@pytest.fixture(params=SYSTEM_CONFIGS.keys())
def system_config(request):
    """Fixture that returns configuration for each system."""
    return SYSTEM_CONFIGS[request.param], MHE_CONFIGS[request.param], request.param

def test_mhe_identification(system_config, tmp_path):
    """
    Test MHE parameter identification for a given system.
    Verifies that the estimated parameters converge to the true values.
    """
    system_config, mhe_config, system_name = system_config


    system, c0, theta_true, delta_theta = create_system(system_config)
    check_system_ok(system)
    mhe_params = create_mhe_params(mhe_config, dt=0.02, mhe_horizont=400)
    mhe_params.print()
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
    num_windows = 20

    t_windows, u_windows, meas_windows, _ = data_gen.generate_sliding_windows_exact(
        c0=c0,
        theta=theta_true,
        t0=t0,
        tf=T_f,
        num_windows=num_windows,
        n_measurement=N_meas,
        overlap_points=overlap_points
    )

    def get_window(i):
        return t_windows[i], u_windows[i], meas_windows[i], _

    # Initial guess for parameters (slightly perturbed)
    initial_theta = delta_theta + theta_true

    # Run MHE estimation
    reset_mhe_solver(generator.get_model(), 
                    acados_solver,
                    u_windows[0],
                    system.get_initial_state(meas_windows[0][0], u_windows[0][0], initial_theta),
                    initial_theta,
                    N_meas
                    )
    
    results = run_mhe_estimation(
        mhe_model=generator.get_model(),
        acados_solver_factory=acados_solver,
        get_window_func=get_window,
        get_initial_state_func=system.get_initial_state,
        overlap_points=overlap_points,
        initial_theta=initial_theta,
        mhe_params=mhe_params,
        num_windows=num_windows,
        ridge_reg = 1e-0,
        R_inv=mhe_params.measurements_residual_r,
        forgetting_factor=0.01,
        compute_advanced_fim=True,
        plot=False
    )


    # Check that the last window's parameter estimate is close to true values
    final_theta_est = results[-1].param_est

    # Relative error tolerance (10% for now, can be tightened)
    rel_error = np.abs((final_theta_est - theta_true) / theta_true)
    assert np.all(rel_error < 1e-2), f"System {system_name}: final estimate {final_theta_est} differs from true {theta_true} by {rel_error}"
