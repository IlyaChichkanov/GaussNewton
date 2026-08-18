# -*- coding: utf-8 -*-
"""Проверка собранного якобиана против конечных разностей.

Зачем отдельный тест. Остальные сверки в pytests/ — взаимные: накопительный
путь сравнивается с плотным (accumulated_test), коллокации — с solve_ivp
(collocation_test), adaptive — с плотным решением (adaptive_test). Все они
делят общее ядро MultipleShooting.shoot_rows, поэтому систематическая ошибка
в нём прошла бы через любую из этих сверок незамеченной.

Здесь эталон независимый: центральные разности вектора невязок
r(theta_full) = [R; R_G], посчитанного тем же solve(). Проверяется знаковое
соглашение всего кода:

    [J; J_G] = -d[R; R_G]/d theta_full

(J — якобиан ПРЕДСКАЗАНИЙ, а невязка R = W(y - h), отсюда минус; ровно на
этом соглашении построен gn_step, где rhs = [J^T R; R_G]).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import Integrator, LotkaVoltera, MassSpringDamper
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.problem import MultipleShooting
from gauss_newton.collocation_shooting import CollocationShooting
from gauss_newton.normal_equations import normal_equations_of


def _make_data(system, c0, theta_true, t_end=2.0, n_meas=21, seed=0):
    np.random.seed(seed)
    gen = SyntheticDataGenerator(system, sigma=0.01, perturb_initial=False,
                                 use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=c0, theta=theta_true,
                                     time_intervals=[(0.0, t_end)],
                                     n_measurements=n_meas)
    return t_b[0], meas_b[0]


def _residuals(problem, theta_full):
    """Вектор невязок [R; R_G] — то, что дифференцируем численно."""
    _, R, _, R_G = problem.solve(theta_full)
    return np.concatenate([R, R_G])


def _fd_jacobian(problem, theta_full, rel_step=1e-7):
    """Центральные разности d[R; R_G]/d theta_full."""
    n = len(theta_full)
    columns = []
    for i in range(n):
        h = rel_step * max(1.0, abs(theta_full[i]))
        plus, minus = theta_full.copy(), theta_full.copy()
        plus[i] += h
        minus[i] -= h
        columns.append((_residuals(problem, plus) - _residuals(problem, minus))
                       / (2.0 * h))
    return np.column_stack(columns)


def _assembled_jacobian(problem, theta_full):
    J, _, J_G, _ = problem.solve(theta_full)
    return np.vstack([J.toarray(), J_G.toarray()])


# ---------------------------------------------------------------------------
# Вариационные уравнения (solve_ivp)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_shoot", [1, 3])
def test_jacobian_matches_finite_differences(n_shoot):
    """J и J_G совпадают с -d[R; R_G]/dp для multiple shooting."""
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]))

    problem = MultipleShooting(system, N_shoot=n_shoot, gamma=np.ones(2),
                               c0_cost=1.0, use_jax=False)
    # Допуски интегратора жёстче обычных: иначе ошибка solve_ivp (1e-5 по
    # умолчанию) доминирует над разностной производной и сверять нечего
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
    problem.add_batch(meas, t_meas)

    theta_full = problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))
    J_code = _assembled_jacobian(problem, theta_full)
    J_num = _fd_jacobian(problem, theta_full)

    assert J_code.shape == J_num.shape
    scale = max(np.abs(J_num).max(), 1e-12)
    err = np.abs(J_code + J_num).max() / scale
    assert err < 1e-6, f"J не совпадает с конечными разностями: rel err {err:.2e}"


def test_jacobian_matches_finite_differences_with_input():
    """Система со входом и НЕтождественным наблюдением (h = x_0, n_obs < nx).

    Здесь не срабатывает быстрый путь identity_observation, поэтому
    проверяется ветка через observation_batch (dh/dx, dh/dtheta).

    Вход намеренно гладкий: у MassSpringDamper сигнал разрывен при t=1, и
    adaptive solve_ivp перешагивает излом с неконтролируемой ошибкой — см.
    test_discontinuous_input_degrades_sensitivities.
    """
    system = Integrator()
    t_meas, meas = _make_data(system, np.array([0.0, 0.0]),
                              np.array([1.0]), t_end=3.0, n_meas=16)

    problem = MultipleShooting(system, N_shoot=2, gamma=np.ones(system.n_obs),
                               c0_cost=1.0, use_jax=False)
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
    problem.add_batch(meas, t_meas)

    theta_full = problem.make_full_theta(np.array([0.85]),
                                         c0_init_method='zeros')
    J_code = _assembled_jacobian(problem, theta_full)
    # Шаг разностей крупнее, чем в тесте выше: состояние Integrator растёт
    # как t^2, шум округления solve_ivp на этих величинах ~1e-11, и при
    # rel_step=1e-7 разностная производная тонет в нём (ошибка падает
    # строго как 1/h — проверено). 1e-4 — компромисс шум/срез.
    J_num = _fd_jacobian(problem, theta_full, rel_step=1e-4)

    scale = max(np.abs(J_num).max(), 1e-12)
    err = np.abs(J_code + J_num).max() / scale
    assert err < 1e-5, f"J не совпадает с конечными разностями: rel err {err:.2e}"


def _fd_error(problem, theta_full):
    J_code = _assembled_jacobian(problem, theta_full)
    J_num = _fd_jacobian(problem, theta_full)
    return np.abs(J_code + J_num).max() / max(np.abs(J_num).max(), 1e-12)


def test_discontinuous_input_degrades_sensitivities():
    """Документирует ограничение: разрывный по времени входной сигнал.

    MassSpringDamper держит вход нулевым при t < 1 и включает его скачком.
    Явный адаптивный solve_ivp про излом не знает: он перешагивает точку
    разрыва, и на интервале, её содержащем, контроль ошибки не работает —
    чувствительности теряют несколько порядков точности. Пока разрыв не
    пересечён, якобиан совпадает с разностями до ~1e-8.

    Тест фиксирует факт, а не «правильное» поведение: если интегратор
    научат ставить узел в точку разрыва (t_eval/max_step или нарезка шутов
    по излому), тест упадёт — и это будет поводом его обновить.
    """
    system = MassSpringDamper()
    theta0 = np.array([2.7, 1.2])

    def build(t_end):
        t_meas, meas = _make_data(system, np.array([1.0, 0.0]),
                                  np.array([3.0, 1.0]), t_end=t_end, n_meas=16)
        problem = MultipleShooting(system, N_shoot=2, gamma=np.ones(2),
                                   c0_cost=1.0, use_jax=False)
        problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
        problem.add_batch(meas, t_meas)
        return problem, problem.make_full_theta(theta0)

    err_before = _fd_error(*build(0.9))    # целиком до разрыва
    err_across = _fd_error(*build(3.0))    # интервал содержит t = 1

    assert err_before < 1e-7, (
        f"на гладком участке якобиан обязан сходиться: {err_before:.2e}")
    assert err_across > 100 * err_before, (
        "ожидалась потеря точности на разрыве входа; если её больше нет — "
        f"интегратор починили, обновите тест (before {err_before:.2e}, "
        f"across {err_across:.2e})")


# ---------------------------------------------------------------------------
# Коллокации: IND даёт ТОЧНЫЕ производные дискретной схемы, поэтому
# согласие с разностями той же схемы должно быть заметно лучше
# ---------------------------------------------------------------------------
def test_collocation_jacobian_matches_finite_differences():
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]))

    problem = CollocationShooting(system, N_shoot=3, gamma=np.ones(2),
                                  c0_cost=1.0, K=3, n_sub=2)
    problem.add_batch(meas, t_meas)

    theta_full = problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))
    J_code = _assembled_jacobian(problem, theta_full)
    J_num = _fd_jacobian(problem, theta_full)

    scale = max(np.abs(J_num).max(), 1e-12)
    err = np.abs(J_code + J_num).max() / scale
    assert err < 1e-7, f"IND-якобиан расходится с разностями схемы: {err:.2e}"


# ---------------------------------------------------------------------------
# Градиент нормальных уравнений: g = J^T R должен быть -1/2 grad ||R||^2
# ---------------------------------------------------------------------------
def test_gradient_matches_finite_differences():
    """g из NormalEquations согласован с численным градиентом ||R||^2."""
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]))

    problem = MultipleShooting(system, N_shoot=2, gamma=np.ones(2),
                               c0_cost=1.0, use_jax=False)
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
    problem.add_batch(meas, t_meas)
    theta_full = problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))

    g_code = normal_equations_of(problem, theta_full).g

    def rss(q):
        _, R, _, _ = problem.solve(q)
        return float(R @ R)

    g_num = np.empty_like(theta_full)
    for i in range(len(theta_full)):
        h = 1e-6 * max(1.0, abs(theta_full[i]))
        plus, minus = theta_full.copy(), theta_full.copy()
        plus[i] += h
        minus[i] -= h
        g_num[i] = (rss(plus) - rss(minus)) / (2.0 * h)

    # g = J^T R, а d||R||^2/dp = 2 R^T dR/dp = -2 J^T R  =>  g = -g_num / 2
    scale = max(np.abs(g_num).max(), 1e-12)
    err = np.abs(g_code + g_num / 2.0).max() / scale
    assert err < 1e-6, f"g не согласован с градиентом ||R||^2: rel err {err:.2e}"
