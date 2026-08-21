"""Identification through CollocationShootingAccum.

Same shape as gauss_newton_test.py, but the problem is Radau IIA collocation
with accumulated H and g, so neither the big J nor a manual mu is involved.
"""
import pytest
import numpy as np
from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.normal_equations import CollocationShootingAccum
from gauss_newton.adaptive import run_optimization_adaptive
from gauss_newton.utils import plot_solution

import os

# Figures stay closed by default: fig.show() hangs CI and any headless run.
# Enable locally with GN_TEST_PLOT=1 pytest ...
PLOT = os.environ.get("GN_TEST_PLOT", "0") not in ("0", "", "false", "False")

SYSTEMS_CONFIG = {
    "LotkaVolterra": {
        "class": LotkaVoltera,
        "true_params": np.array([1.2, 0.4, 0.3, 0.1]),
        "initial_state": np.array([6.0, 5.0]),
        "time_interval": (0.0, 4.0),
        "n_measurements": 50,
        "noise_sigma": 1e-15,
        "N_shoot": 1,
        "n_sub": 1,          # collocation elements per measurement interval
        "theta_init": np.array([1.0, 0.5, 0.2, 0.05]),  # deliberately off
    },
    "Attractor": {
        "class": Attractor,
        "true_params": np.array([10.0, 28.0, 8.0 / 3.0]),
        "initial_state": np.array([1.0, 1.0, 1.0]),
        "time_interval": (0.0, 5.0),
        "n_measurements": 100,
        "noise_sigma": 0.01,
        "N_shoot": 20,
        "n_sub": 2,          # chaotic system, so a twice finer grid
        "theta_init": np.array([0.0, 0.0, 0.0]),
    },
}


def pytest_generate_tests(metafunc):
    """One test invocation per system in SYSTEMS_CONFIG."""
    if "system_name" in metafunc.fixturenames:
        metafunc.parametrize("system_name", list(SYSTEMS_CONFIG.keys()),
                             scope="function")


@pytest.fixture
def system_config(system_name):
    return SYSTEMS_CONFIG[system_name]


@pytest.fixture
def system(system_config):
    return system_config["class"]()


@pytest.fixture
def true_params(system_config):
    return system_config["true_params"]


@pytest.fixture
def initial_state(system_config):
    return system_config["initial_state"]


@pytest.fixture
def time_interval(system_config):
    return system_config["time_interval"]


@pytest.fixture
def synthetic_data(system, true_params, initial_state, time_interval,
                   system_config):
    np.random.seed(0)   # reproducible noise
    gen = SyntheticDataGenerator(
        system,
        sigma=system_config["noise_sigma"],
        perturb_initial=True,
        perturbation_scale=0.0,
        use_jax=True
    )
    t_batches, meas_batches, state_batches, _ = gen.generate(
        c0=initial_state,
        theta=true_params,
        time_intervals=[time_interval],
        n_measurements=system_config["n_measurements"]
    )
    return t_batches[0], meas_batches[0], state_batches[0]


def test_identification(system, true_params, synthetic_data, system_config):
    t_meas, meas_batch, state_true_batch = synthetic_data

    problem = CollocationShootingAccum(
        system=system,
        N_shoot=system_config["N_shoot"],
        gamma=np.ones(system.n_obs),
        c0_cost=1.0,
        n_sub=system_config["n_sub"],
    )
    problem.add_batch(meas_batch, t_meas)

    theta_full = problem.make_full_theta(system_config["theta_init"])
    theta_full_opt, hist = run_optimization_adaptive(
        problem, theta_full, n_iter=40)

    n_theta = len(true_params)
    theta_est = theta_full_opt[:n_theta]
    rel_error = np.abs((theta_est - true_params) / true_params)

    print(f"\nSystem: {system_config['class'].__name__} (collocation + accum)")
    print("True parameters:     ", true_params)
    print("Estimated parameters:", theta_est)
    print("Relative error:      ", rel_error)
    print(f"Iterations: {len(hist['mu']) - 1}, accepted: {len(hist['accepted'])}, "
          f"solves: {hist['n_solves']}, mu: {hist['mu'][0]:.2e} -> {hist['mu'][-1]:.2e}")

    assert np.all(rel_error < 0.05), \
        f"Estimation error too high: {rel_error}"

    if PLOT:
        fig = plot_solution(
            problem=problem,
            theta_hist=hist['theta'],
            plot_xy=1,
            plot_theta=True,
            plot_trajectory=0,
            plot_true_solution=False,
            plot_residuals=True,
            plot_measurements=1,
            r_meas_hist=hist['r_meas'],
            r_cont_hist=hist['r_cont'],
            ci_low_hist=hist['ci_low'],
            ci_high_hist=hist['ci_high'],
            index=-1,
            theta_true=true_params,
        )
        fig.show()
