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
# Шаг: совпадение с плотным решением той же седловой системы
# ---------------------------------------------------------------------------
def test_step_matches_dense_saddle_solve():
    """gn_step решает ровно ту систему, которая написана в его docstring.

    Эталон здесь ВНЕШНИЙ: седловая матрица собирается плотно и решается
    numpy.linalg.solve, а не второй копией нашего же разреженного кода.
    Проверяется и матрица, и правая часть:

        [[H + lambda_reg I + lam diag(H), J_G^T], [J_G, -mu I]] [d; nu] = [g; R_G]
    """
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])
    J, R, J_G, R_G = prob.solve(theta_full)

    mu, lam, lam_reg = 1.0, 1e-3, 1e-6
    ne = NormalEquations.from_jacobian(J, R, J_G, R_G)
    delta, pred = gn_step(ne, mu, lam, lam_reg)

    H = ne.H.toarray()
    n, m = H.shape[0], J_G.shape[0]
    D = lam_reg * np.eye(n) + lam * np.diag(np.maximum(np.diag(H), 1e-10))
    K = np.block([[H + D, J_G.toarray().T],
                  [J_G.toarray(), -mu * np.eye(m)]])
    delta_ref = np.linalg.solve(K, np.concatenate([ne.g, R_G]))[:n]

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
# Гейт Пауэлла (rss_stall_tol): Аттрактор сходится и при МАЛОМ числе шутов
# ---------------------------------------------------------------------------
def test_attractor_converges_with_few_shoots():
    """Attractor, N_shoot=5, theta0=0 — закрепляет гейт rss_stall_tol.

    Без гейта mu утаптывается на ранних итерациях (rss ещё падает на порядки,
    а стыковка колеблется), ограничения начинают доминировать, и решение
    запирается на консистентной траектории вдали от измерений
    (rel_err ~ 1.2, rss ~ 9e3 при r_cont ~ 1e-10). С гейтом тот же случай
    сходится: rel_err ~ 3e-4, стыковка затягивается до ~1e-10.
    """
    config = dict(SYSTEMS_CONFIG["Attractor"], N_shoot=5)
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=80,
                                                track_covariance=False)

    err = rel_err(theta_opt, config["true_params"])
    assert err < 1e-2, f"запирание на консистентной траектории: rel_err={err:.3e}"
    # стыковка при этом дотянута штрафом, а не брошена
    assert hist["r_cont"][-1] < 1e-6


# ---------------------------------------------------------------------------
# Цикл действительно МИНИМИЗИРУЕТ: эталон - стоимость в истинной точке
# ---------------------------------------------------------------------------
def test_beats_cost_at_true_parameters():
    """Найденная точка не хуже истинных параметров по той же стоимости.

    Раньше здесь сравнивались два наших цикла между собой — сверка была
    взаимной и уехала бы вместе с общей ошибкой. Эталон тут внешний:
    параметры, которыми данные были сгенерированы. Оптимум зашумлённой
    задачи не обязан совпадать с истиной, но стоимость в нём обязана быть
    не выше — иначе цикл не дошёл до минимума.
    """
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)

    prob = make_problem(config, t_meas, meas)
    theta_opt, _ = run_optimization_adaptive(
        prob, prob.make_full_theta(config["theta_init"]), n_iter=40)

    cost_opt = NormalEquations.from_jacobian(*prob.solve(theta_opt)).cost()
    theta_true = prob.make_full_theta(config["true_params"])
    cost_true = NormalEquations.from_jacobian(*prob.solve(theta_true)).cost()

    assert cost_opt <= cost_true * (1 + 1e-6), \
        f"цикл не дошёл до минимума: cost {cost_opt:.6e} > истинная {cost_true:.6e}"
    assert rel_err(theta_opt, config["true_params"]) < 0.2


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
