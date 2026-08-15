import types
from pathlib import Path
import sys

import numpy as np
import pytest
import casadi as ca

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import ODESystem, SyntheticDataGenerator, SystemJacobian
from commom_utils.collocation import RadauTables, CollocationSystemJacobian
from gauss_newton.gauss_newton_math import MultipleShooting, run_optimization
from gauss_newton.collocation_shooting import CollocationShooting

SYSTEMS_CONFIG = {
    "LotkaVolterra": {
        "class": LotkaVoltera,
        "true_params": np.array([1.2, 0.4, 0.3, 0.1]),
        "initial_state": np.array([6.0, 5.0]),
        "time_interval": (0.0, 4.0),
        "n_measurements": 50,
        "noise_sigma": 1e-15,
        "N_shoot": 1,
        "mu": 0.0,
        "theta_init": np.array([1.0, 0.5, 0.2, 0.05]),
        "n_sub": 1,
    },
    "Attractor": {
        "class": Attractor,
        "true_params": np.array([10.0, 28.0, 8.0 / 3.0]),
        "initial_state": np.array([1.0, 1.0, 1.0]),
        "time_interval": (0.0, 5.0),
        "n_measurements": 100,
        "noise_sigma": 0.01,
        "N_shoot": 20,
        "mu": 100.0,
        "theta_init": np.array([0.0, 0.0, 0.0]),
        "n_sub": 2,
    },
}


def make_config(mu):
    cfg = types.SimpleNamespace()
    cfg.mu = mu
    cfg.n_iter = 20
    cfg.lambda_ = 0.001
    cfg.lambda_reg = 0.0
    cfg.mu_dec = 0.7
    cfg.mu_min = 1e-6
    return cfg


def generate_data(config):
    system = config["class"]()
    gen = SyntheticDataGenerator(system, sigma=config["noise_sigma"],
                                 perturb_initial=True, perturbation_scale=0.0,
                                 use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=config["initial_state"],
                                     theta=config["true_params"],
                                     time_intervals=[config["time_interval"]],
                                     n_measurements=config["n_measurements"])
    return system, t_b[0], meas_b[0]


# ---------------------------------------------------------------------------
# Таблицы Радо IIA
# ---------------------------------------------------------------------------
def test_radau_tables():
    rt = RadauTables(3)
    # Табличная матрица Бутчера Радо IIA (K=3, порядок 5)
    a_ref = np.array([
        [0.19681547722366, -0.06553542585020,  0.02377097434822],
        [0.39442431473909,  0.29207341166523, -0.04154875212600],
        [0.37640306270047,  0.51248582618842,  0.11111111111111],
    ])
    assert np.allclose(rt.butcher_a, a_ref, atol=1e-10)
    # A = 1 (x) I: каждая стадия стартует из левого края
    assert np.allclose(-rt.butcher_a @ rt.d0, np.ones(3), atol=1e-12)
    # Правый узел совпадает с концом элемента
    assert rt.tau[-1] == 1.0

    # K=1 — неявный Эйлер
    rt1 = RadauTables(1)
    assert np.allclose(rt1.butcher_a, [[1.0]])


# ---------------------------------------------------------------------------
# Интегратор против эталона (вариационные уравнения, жёсткий допуск)
# ---------------------------------------------------------------------------
def test_integrator_matches_reference():
    system = LotkaVoltera()
    theta = np.array([1.2, 0.4, 0.3, 0.1])
    c0 = np.array([6.0, 5.0])
    t_eval = np.linspace(0.0, 4.0, 50)

    ref = SystemJacobian(system)
    ref.ATOL = ref.RTOL = 1e-12
    sol_ref = ref.get_jacobian_solution(c0, theta, t_eval)

    colloc = CollocationSystemJacobian(system, K=3, n_sub=2)
    sol_c = colloc.get_jacobian_solution(c0, theta, t_eval)

    assert sol_c.shape == sol_ref.shape
    scale = np.abs(sol_ref).max(axis=1, keepdims=True) + 1e-12
    rel = np.abs(sol_c - sol_ref) / scale
    assert rel.max() < 1e-8, f"max rel diff {rel.max():.2e}"

    # get_solution согласован с get_jacobian_solution
    nx = system.nx
    state_only = colloc.get_solution(c0, theta, t_eval)
    assert np.allclose(state_only, sol_c[:nx], atol=1e-14)


# ---------------------------------------------------------------------------
# IND: чувствительности — точные производные дискретной схемы
# ---------------------------------------------------------------------------
def test_ind_property():
    system = LotkaVoltera()
    theta = np.array([1.2, 0.4, 0.3, 0.1])
    c0 = np.array([6.0, 5.0])
    t_eval = np.linspace(0.0, 4.0, 30)
    nx, nth = system.nx, system.np

    colloc = CollocationSystemJacobian(system, K=3, n_sub=1)
    sol = colloc.get_jacobian_solution(c0, theta, t_eval)
    S_th = sol[nx:nx + nx * nth, -1].reshape(nx, nth)
    S_c = sol[nx + nx * nth:, -1].reshape(nx, nx)

    eps = 1e-6
    for i in range(nth):
        d = eps * np.eye(nth)[i]
        fd = (colloc.get_solution(c0, theta + d, t_eval)[:, -1]
              - colloc.get_solution(c0, theta - d, t_eval)[:, -1]) / (2 * eps)
        assert np.abs(fd - S_th[:, i]).max() < 1e-6
    for i in range(nx):
        d = eps * np.eye(nx)[i]
        fd = (colloc.get_solution(c0 + d, theta, t_eval)[:, -1]
              - colloc.get_solution(c0 - d, theta, t_eval)[:, -1]) / (2 * eps)
        assert np.abs(fd - S_c[:, i]).max() < 1e-6


# ---------------------------------------------------------------------------
# Компилированный марш (rootfinder + mapaccum) против Python-эталона
# ---------------------------------------------------------------------------
def test_compiled_matches_python():
    system = LotkaVoltera()
    theta = np.array([1.2, 0.4, 0.3, 0.1])
    c0 = np.array([6.0, 5.0])
    t_eval = np.linspace(0.0, 4.0, 40)

    py = CollocationSystemJacobian(system, K=3, n_sub=2, use_compiled=False)
    co = CollocationSystemJacobian(system, K=3, n_sub=2, use_compiled=True)

    sol_py = py.get_jacobian_solution(c0, theta, t_eval)
    sol_co = co.get_jacobian_solution(c0, theta, t_eval)
    # Критерии остановки Ньютона различаются (dz vs невязка) — допуск 1e-8
    assert np.abs(sol_co - sol_py).max() < 1e-8

    st_py = py.get_solution(c0, theta, t_eval)
    st_co = co.get_solution(c0, theta, t_eval)
    assert np.abs(st_co - st_py).max() < 1e-8


# ---------------------------------------------------------------------------
# Жёсткая линейная система: dx/dt = -theta (x - sin t) + cos t
# Точное решение: x(t) = sin t + (c0) e^{-theta t}; L-устойчивость
# позволяет шаг h=0.1 при theta=1000 (явному методу нужен h ~ 1/theta)
# ---------------------------------------------------------------------------
class StiffLinear(ODESystem):
    def __init__(self):
        super().__init__(1, 1, 2)

    def get_derivative(self, state, theta, u):
        return ca.vertcat(-theta[0] * (state[0] - u[0]) + u[1])

    def get_input_signals(self, t):
        return [np.sin(t), np.cos(t)]


def test_stiff_simulation():
    system = StiffLinear()
    theta = np.array([1000.0])
    c0 = np.array([1.0])
    t_eval = np.linspace(0.0, 2.0, 21)   # h = 0.1, h*theta = 100

    colloc = CollocationSystemJacobian(system, K=3, n_sub=1)
    x = colloc.get_solution(c0, theta, t_eval)[0]
    exact = np.sin(t_eval) + np.exp(-theta[0] * t_eval)

    # Пограничный слой: на первом узле ошибка ~|R(-h theta)| ~ 2.5e-2,
    # затем гаснет мультипликативно до уровня интерполяции гладкой части (~1e-8).
    # Явная схема при h*theta = 100 расходится; здесь решение остаётся ограниченным.
    assert np.abs(x - exact).max() < 0.05
    assert np.abs(x[5:] - exact[5:]).max() < 1e-6


# ---------------------------------------------------------------------------
# Сквозная идентификация через существующий run_optimization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_name", list(SYSTEMS_CONFIG.keys()))
def test_identification(system_name):
    config = SYSTEMS_CONFIG[system_name]
    system, t_meas, meas = generate_data(config)

    problem = CollocationShooting(system=config["class"](),
                                  N_shoot=config["N_shoot"],
                                  gamma=np.ones(system.n_obs),
                                  c0_cost=1.0, n_sub=config["n_sub"])
    problem.add_batch(meas, t_meas)
    theta_full = problem.make_full_theta(config["theta_init"])

    _, _, _, theta_full_opt, _, _ = run_optimization(
        problem=problem, config=make_config(config["mu"]),
        theta_full=theta_full, system=system, verbose=False)

    true_params = config["true_params"]
    theta_est = theta_full_opt[:len(true_params)]
    rel_error = np.abs((theta_est - true_params) / true_params)
    print(f"\n{system_name} (collocation): est={theta_est}, rel err={rel_error}")
    assert np.all(rel_error < 0.05), f"Estimation error too high: {rel_error}"


# ---------------------------------------------------------------------------
# Согласие с MultipleShooting (быстрая система)
# ---------------------------------------------------------------------------
def test_agrees_with_multiple_shooting():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    system, t_meas, meas = generate_data(config)
    estimates = {}
    for label, cls, kwargs in [("ms", MultipleShooting, dict(use_jax=False)),
                               ("colloc", CollocationShooting,
                                dict(n_sub=config["n_sub"]))]:
        prob = cls(system=config["class"](), N_shoot=config["N_shoot"],
                   gamma=np.ones(system.n_obs), c0_cost=1.0, **kwargs)
        prob.add_batch(meas, t_meas)
        theta_full = prob.make_full_theta(config["theta_init"])
        out = run_optimization(problem=prob, config=make_config(config["mu"]),
                               theta_full=theta_full, system=system,
                               verbose=False)
        estimates[label] = out[3][:len(config["true_params"])]

    diff = np.abs(estimates["ms"] - estimates["colloc"])
    assert diff.max() < 1e-3, f"MS vs collocation mismatch: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
