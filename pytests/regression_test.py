# -*- coding: utf-8 -*-
"""Численный эталон математического ядра.

Зачем. Переименования безопасны, а перестановка осей в einsum, замена
moveaxis на dataclass или другой порядок вычисления f_x — нет: они меняют
ЧИСЛА, а не падают. Ни один из остальных тестов такое не ловит надёжно —
взаимные сверки поедут вместе с обеими сторонами, а сквозная идентификация
сойдётся к тем же параметрам даже при перепутанных столбцах якобиана
начальных состояний.

Здесь для четырёх фиксированных задач заморожены сырые матрицы шага ГН:
J, R, J_G, R_G (плотная сборка), H, g (накопление), delta и pred (шаг).
Рефакторинг обязан быть численно нулевым: изменение любого из этих массивов
означает, что поведение изменилось, а не только имена.

Четыре случая покрывают все ветки, которых касается рефакторинг:

    lv_scipy        вариационные уравнения через solve_ivp (numpy-ветка)
    lv_jax          те же уравнения через jax odeint (_variational_rhs_jax)
    lv_colloc       коллокации Радо IIA + IND (свой марш, свои рекурсии)
    integrator_obs  НЕтождественное наблюдение h = x_0 и вход u(t)
                    (ветка observation_batch: dh/dx, dh/dtheta)

Эталон пересоздаётся `GN_REGEN_REFERENCE=1 uv run pytest
pytests/regression_test.py`. Делать это можно ТОЛЬКО когда изменение чисел
осознанное и объяснимое — иначе тест теряет смысл.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import Integrator, LotkaVoltera
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.normal_equations import (MultipleShootingAccum,
                                           CollocationShootingAccum)
from gauss_newton.adaptive import gn_step

REFERENCE = Path(__file__).parent / "data" / "reference.npz"
REGEN = os.environ.get("GN_REGEN_REFERENCE", "0") not in ("0", "", "false", "False")

# Допуск сверки — свой на каждый интегратор, и вот почему.
#
# Коллокации идут по ФИКСИРОВАННОЙ сетке элементов: последовательность
# операций одна и та же везде, расхождение — только последние биты.
#
# solve_ivp и jax odeint выбирают шаг АДАПТИВНО, а решение о размере шага
# принимается по сравнению оценки ошибки с допуском. Чуть другая арифметика
# (другой CPU, другая сборка BLAS) — и последовательность шагов расходится,
# а с ней и результат. Замерено: на раннере GitHub Actions случай
# integrator_obs разошёлся с локальным эталоном на 1.0e-8 при допуске 1e-10.
# Это НЕ ошибка якобиана: 1e-8 — точность самого интегратора, а не схемы.
#
# 1e-6 всё ещё на порядки строже того, что должен ловить этот тест:
# перепутанная ось или неверный индекс einsum меняют элементы на O(1),
# а не в восьмом знаке.
TOL_EXACT = dict(rtol=1e-10, atol=1e-12)     # фиксированный шаг
TOL_ADAPTIVE = dict(rtol=1e-6, atol=1e-10)   # адаптивный шаг

# Фиксированная точка линеаризации для шага ГН
MU, LAM = 1e-3, 1e-3


def _make_data(system, c0, theta_true, t_end, n_meas, seed=0):
    np.random.seed(seed)
    gen = SyntheticDataGenerator(system, sigma=0.01, perturb_initial=False,
                                 use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=c0, theta=theta_true,
                                     time_intervals=[(0.0, t_end)],
                                     n_measurements=n_meas)
    return t_b[0], meas_b[0]


# ---------------------------------------------------------------------------
# Построители задач. Каждый возвращает (problem, theta_full).
# Тонкие допуски интеграторов — чтобы эталон определялся математикой,
# а не случайным выбором шага адаптивным solve_ivp.
# ---------------------------------------------------------------------------
def _lotka(use_jax):
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]),
                              t_end=2.0, n_meas=21)
    problem = MultipleShootingAccum(system, N_shoot=3, gamma=np.ones(2),
                                    c0_cost=1.0, use_jax=use_jax)
    problem.system.ATOL = problem.system.RTOL = 1e-12
    problem.add_batch(meas, t_meas)
    return problem, problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))


def _case_lv_scipy():
    return _lotka(use_jax=False)


def _case_lv_jax():
    return _lotka(use_jax=True)


def _case_lv_colloc():
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]),
                              t_end=2.0, n_meas=21)
    problem = CollocationShootingAccum(system, N_shoot=3, gamma=np.ones(2),
                                       c0_cost=1.0, K=3, n_sub=2)
    problem.add_batch(meas, t_meas)
    return problem, problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))


def _case_integrator_obs():
    system = Integrator()
    t_meas, meas = _make_data(system, np.array([0.0, 0.0]), np.array([1.0]),
                              t_end=3.0, n_meas=16)
    problem = MultipleShootingAccum(system, N_shoot=2,
                                    gamma=np.ones(system.n_obs),
                                    c0_cost=1.0, use_jax=False)
    problem.system.ATOL = problem.system.RTOL = 1e-12
    problem.add_batch(meas, t_meas)
    return problem, problem.make_full_theta(np.array([0.85]),
                                            c0_init_method='zeros')


# case -> (построитель, допуск)
CASES = {
    "lv_scipy": (_case_lv_scipy, TOL_ADAPTIVE),
    "lv_jax": (_case_lv_jax, TOL_ADAPTIVE),
    "lv_colloc": (_case_lv_colloc, TOL_EXACT),
    "integrator_obs": (_case_integrator_obs, TOL_ADAPTIVE),
}


def _snapshot(build):
    """Сырьё шага ГН для одной задачи: плотная сборка + накопление + шаг."""
    problem, theta_full = build()

    J, R, J_G, R_G = problem.solve(theta_full)
    ne = problem.normal_equations(theta_full)
    delta, pred = gn_step(ne, MU, LAM)

    return {
        "theta_full": theta_full,
        "J": J.toarray(),
        "R": R,
        "J_G": J_G.toarray(),
        "R_G": R_G,
        "H": ne.H.toarray(),
        "g": ne.g,
        "delta": delta,
        "pred": np.array(pred),
    }


def _regenerate():
    out = {}
    for case, (build, _) in CASES.items():
        for key, value in _snapshot(build).items():
            out[f"{case}__{key}"] = value
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REFERENCE, **out)
    return out


@pytest.fixture(scope="session")
def reference():
    if REGEN or not REFERENCE.exists():
        return _regenerate()
    return dict(np.load(REFERENCE))


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_reference(case, reference):
    if REGEN:
        pytest.skip("эталон только что перегенерирован — сверять не с чем")

    build, tol = CASES[case]
    got = _snapshot(build)
    for key, value in got.items():
        expected = reference[f"{case}__{key}"]
        assert value.shape == expected.shape, \
            f"{case}.{key}: форма {value.shape}, эталон {expected.shape}"
        np.testing.assert_allclose(
            value, expected, **tol,
            err_msg=f"{case}.{key} разошёлся с эталоном — поведение изменилось, "
                    f"а не только имена")
