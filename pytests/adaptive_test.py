import types
from pathlib import Path
import sys

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.gauss_newton_math import (MultipleShooting, run_optimization,
                                            compute_delta_gn)
from gauss_newton.adaptive import gn_step, run_optimization_adaptive
from gauss_newton.normal_equations import NormalEquations
from gauss_newton.collocation_shooting import CollocationShooting

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
        # заведомо плохое начальное приближение: baseline с неудачным mu0
        # (например 0.01) отсюда расходится - адаптивная схема должна пройти
        # без какого-либо ручного mu
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


def make_problem(config, t_meas, meas, cls=MultipleShooting, **kwargs):
    system = config["class"]()
    if cls is MultipleShooting:
        kwargs.setdefault("use_jax", True)
    prob = cls(system=system, N_shoot=config["N_shoot"],
               gamma=np.ones(system.n_obs), c0_cost=1.0, **kwargs)
    prob.add_batch(meas, t_meas)
    return prob


def rel_err(theta, true):
    return np.max(np.abs((theta[:len(true)] - true) / true))


# ---------------------------------------------------------------------------
# Шаг: совпадение с compute_delta_gn при одинаковых (mu, lambda) и pred > 0
# ---------------------------------------------------------------------------
def test_step_matches_compute_delta_gn():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])
    J, R, J_G, R_G = prob.solve(theta_full)

    mu, lam, lam_reg = 1.0, 1e-3, 1e-6
    delta, pred = gn_step(NormalEquations.from_jacobian(J, R, J_G, R_G),
                          mu, lam, lam_reg)
    delta_ref, _ = compute_delta_gn(J, R, J_G, R_G, mu, lam, lam_reg,
                                    theta_full, mu_dec=0.7)
    # одна и та же седловая система - решения совпадают до допуска решателя
    scale = np.abs(delta_ref).max()
    assert np.abs(delta - delta_ref).max() < 1e-9 * scale
    assert pred > 0


def test_pred_positive_across_regimes():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])
    ne = NormalEquations.from_jacobian(*prob.solve(theta_full))

    for mu in [1e-6, 1e-2, 1.0, 1e2]:
        for lam in [1e-6, 1e-3, 1.0]:
            delta, pred = gn_step(ne, mu, lam)
            assert np.all(np.isfinite(delta))
            assert pred > 0, f"pred <= 0 при mu={mu}, lam={lam}"
    # merit согласован с pred по построению: Phi_mu >= 0
    assert ne.merit(1.0) >= 0


# ---------------------------------------------------------------------------
# Сквозная идентификация без ручного mu (у Аттрактора - из theta = 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_name", list(SYSTEMS_CONFIG.keys()))
def test_identification_adaptive(system_name):
    config = SYSTEMS_CONFIG[system_name]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=40)

    err = rel_err(theta_opt, config["true_params"])
    print(f"\n{system_name} (adaptive): est={theta_opt[:len(config['true_params'])]}, "
          f"rel err={err:.2e}, accepted={len(hist['accepted'])}, "
          f"solves={hist['n_solves']}")
    assert err < 0.05, f"Estimation error too high: {err}"
    assert len(hist["accepted"]) > 0
    # mu стартует по кривизне и не растёт при отказах
    assert hist["mu"][0] > 0
    assert hist["mu"][-1] <= hist["mu"][0] + 1e-15


# ---------------------------------------------------------------------------
# Согласие с run_optimization (обе схемы - один и тот же минимум)
# ---------------------------------------------------------------------------
def test_agrees_with_baseline():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)

    prob_b = make_problem(config, t_meas, meas)
    th0 = prob_b.make_full_theta(config["theta_init"])
    cfg = types.SimpleNamespace(mu=1.0, n_iter=40, lambda_=1e-3,
                                lambda_reg=0.0, mu_dec=0.7, mu_min=1e-6)
    out = run_optimization(prob_b, cfg, th0, config["class"](), verbose=False)
    theta_baseline = out[3][:4]

    prob_a = make_problem(config, t_meas, meas)
    theta_adaptive, _ = run_optimization_adaptive(
        prob_a, prob_a.make_full_theta(config["theta_init"]), n_iter=40)

    diff = np.abs(theta_baseline - theta_adaptive[:4])
    assert diff.max() < 1e-3, f"baseline vs adaptive mismatch: {diff}"


# ---------------------------------------------------------------------------
# Single shooting (J_G пусто): чистый LM, ранняя остановка на точных данных
# ---------------------------------------------------------------------------
def test_single_shooting_early_stop():
    config = dict(SYSTEMS_CONFIG["LotkaVolterra"], N_shoot=1,
                  noise_sigma=1e-15)
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=50)

    # точность упирается в допуски интегратора (RTOL=1e-5), не в шум данных
    assert rel_err(theta_opt, config["true_params"]) < 5e-3
    # сходимость к полу интегратора -> остановка сильно раньше лимита
    assert len(hist["mu"]) - 1 < 50


# ---------------------------------------------------------------------------
# Работает и с коллокационным интегратором (тот же контракт solve)
# ---------------------------------------------------------------------------
def test_works_with_collocation():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas, cls=CollocationShooting, n_sub=1)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=40)

    err = rel_err(theta_opt, config["true_params"])
    assert err < 0.05, f"Estimation error too high: {err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
