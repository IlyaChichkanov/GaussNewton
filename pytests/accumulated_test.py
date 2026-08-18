from pathlib import Path
import sys

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.problem import MultipleShooting
from gauss_newton.adaptive import gn_step, run_optimization_adaptive
from gauss_newton.normal_equations import (NormalEquations,
                                           MultipleShootingAccum,
                                           CollocationShootingAccum)

SYSTEMS_CONFIG = {
    "LotkaVolterra": {
        "class": LotkaVoltera,
        "true_params": np.array([1.2, 0.4, 0.3, 0.1]),
        "initial_state": np.array([6.0, 5.0]),
        "time_interval": (0.0, 4.0),
        "n_measurements": 50,
        "noise_sigma": 0.01,
        "N_shoot": 5,
        "theta_init": np.array([1.0, 0.5, 0.2, 0.05]),
    },
    "Attractor": {
        "class": Attractor,
        "true_params": np.array([10.0, 28.0, 8.0 / 3.0]),
        "initial_state": np.array([1.0, 1.0, 1.0]),
        "time_interval": (0.0, 5.0),
        "n_measurements": 100,
        "noise_sigma": 0.01,
        "N_shoot": 20,
        "theta_init": np.array([0.0, 0.0, 0.0]),
    },
}


def generate_data(config, seed=0):
    np.random.seed(seed)
    system = config["class"]()
    gen = SyntheticDataGenerator(system, sigma=config["noise_sigma"],
                                 perturb_initial=True, perturbation_scale=0.0,
                                 use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=config["initial_state"],
                                     theta=config["true_params"],
                                     time_intervals=[config["time_interval"]],
                                     n_measurements=config["n_measurements"])
    return system, t_b[0], meas_b[0]


def make_problem(config, t_meas, meas, cls=MultipleShootingAccum, **kwargs):
    system = config["class"]()
    if cls in (MultipleShooting, MultipleShootingAccum):
        # у коллокационных классов use_jax выставляется внутри
        kwargs.setdefault("use_jax", True)
    prob = cls(system=system, N_shoot=config["N_shoot"],
               gamma=np.ones(system.n_obs), c0_cost=1.0, **kwargs)
    prob.add_batch(meas, t_meas)
    return prob


def rel_err(theta, true):
    return np.max(np.abs((theta[:len(true)] - true) / true))


# ---------------------------------------------------------------------------
# Накопленные H, g совпадают с J^T J, J^T R (J не строится, но математика та же)
# ---------------------------------------------------------------------------
def test_accumulated_matches_dense():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    J, R, J_G, R_G = prob.solve(theta_full)              # путь через большую J
    dense = NormalEquations.from_jacobian(J, R, J_G, R_G)
    accum = prob.normal_equations(theta_full)            # накопление

    H_ref = dense.H.toarray()
    assert np.abs(accum.H.toarray() - H_ref).max() < 1e-9 * np.abs(H_ref).max()
    assert np.abs(accum.g - dense.g).max() < 1e-9 * (np.abs(dense.g).max() + 1e-300)
    assert abs(accum.rss - dense.rss) < 1e-9 * dense.rss
    assert accum.n_rows == dense.n_rows == len(R)

    # стыковки собираются тем же кодом (continuity_rows)
    assert np.abs((accum.J_G - J_G).toarray()).max() < 1e-15
    assert np.abs(accum.R_G - R_G).max() < 1e-15


# ---------------------------------------------------------------------------
# Шаг не зависит от того, откуда пришли H и g
# ---------------------------------------------------------------------------
def test_step_matches_dense():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    dense = NormalEquations.from_jacobian(*prob.solve(theta_full))
    accum = prob.normal_equations(theta_full)

    for mu, lam in [(1.0, 1e-3), (1e-4, 1e-1), (1e2, 1e-6)]:
        d_ref, p_ref = gn_step(dense, mu, lam)
        d_acc, p_acc = gn_step(accum, mu, lam)
        assert np.abs(d_acc - d_ref).max() < 1e-8 * np.abs(d_ref).max()
        assert abs(p_acc - p_ref) < 1e-8 * p_ref


# ---------------------------------------------------------------------------
# Ковариация из H совпадает с ковариацией из [J; J_G]
# ---------------------------------------------------------------------------
def test_covariance_matches_dense():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])
    n_theta = len(config["true_params"])

    J, R, J_G, R_G = prob.solve(theta_full)
    # Эталон — та же формула, но из построенной J (NormalEquations.from_jacobian
    # сохранена именно как эталонная сборка, в цикле оптимизации не участвует)
    cov_ref, sigma2_ref, dof_ref = NormalEquations.from_jacobian(
        J, R, J_G, R_G).covariance_theta(n_theta)
    cov, sigma2, dof = prob.normal_equations(theta_full).covariance_theta(n_theta)

    assert dof == dof_ref
    assert abs(sigma2 - sigma2_ref) < 1e-9 * sigma2_ref
    assert np.abs(cov - cov_ref).max() < 1e-8 * np.abs(cov_ref).max()


# ---------------------------------------------------------------------------
# Сквозная идентификация на накопленном пути
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_name", list(SYSTEMS_CONFIG.keys()))
def test_identification_accum(system_name):
    config = SYSTEMS_CONFIG[system_name]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=40)

    err = rel_err(theta_opt, config["true_params"])
    print(f"\n{system_name} (accum): est={theta_opt[:len(config['true_params'])]}, "
          f"rel err={err:.2e}, accepted={len(hist['accepted'])}, "
          f"solves={hist['n_solves']}")
    assert err < 0.05, f"Estimation error too high: {err}"
    assert len(hist["accepted"]) > 0
    # история пригодна для plot_solution: длины совпадают с theta
    assert len(hist["r_meas"]) == len(hist["theta"]) == len(hist["ci_low"])


# ---------------------------------------------------------------------------
# Оба пути (J и накопление) дают один и тот же результат — цикл общий
# ---------------------------------------------------------------------------
def test_agrees_with_dense_path():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)

    prob_j = make_problem(config, t_meas, meas, cls=MultipleShooting)
    theta_j, _ = run_optimization_adaptive(
        prob_j, prob_j.make_full_theta(config["theta_init"]), n_iter=40)

    prob_a = make_problem(config, t_meas, meas)
    theta_a, _ = run_optimization_adaptive(
        prob_a, prob_a.make_full_theta(config["theta_init"]), n_iter=40)

    diff = np.abs(theta_j[:4] - theta_a[:4])
    assert diff.max() < 1e-6, f"dense vs accumulated mismatch: {diff}"


# ---------------------------------------------------------------------------
# Работает и с коллокационным интегратором
# ---------------------------------------------------------------------------
def test_collocation_accum():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas,
                        cls=CollocationShootingAccum, n_sub=1)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, _ = run_optimization_adaptive(prob, theta_full, n_iter=40)

    err = rel_err(theta_opt, config["true_params"])
    assert err < 0.05, f"Estimation error too high: {err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
