# -*- coding: utf-8 -*-
"""Ковариация theta при точных ограничениях непрерывности.

Непрерывность — ограничение, а не измерение, поэтому ковариация берётся из
ККТ-матрицы [[H, J_G^T], [J_G, 0]], а не из H + J_G^T J_G. Прежняя формула
завышала интервалы тем сильнее, чем больше шутов, и зависела от
произвольного масштаба J_G.

Эталоны здесь ВНЕШНИЕ, два независимых:
1. single shooting — тот же оценщик, но ограничений нет вовсе, поэтому
   ковариация считается обычной формулой МНК; multiple shooting обязан
   давать то же самое;
2. Монте-Карло — предсказанное СКО против фактического разброса оценок по
   реализациям шума и покрытие 95%-х интервалов.
"""
from pathlib import Path
import sys

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.problem import MultipleShooting
from gauss_newton.normal_equations import NormalEquations, correlation_matrix
from gauss_newton.adaptive import run_optimization_adaptive

TRUE = np.array([1.2, 0.4, 0.3, 0.1])
THETA0 = np.array([1.0, 0.5, 0.2, 0.05])
X0 = np.array([6.0, 5.0])
N_THETA = 4


def generate_data(seed):
    np.random.seed(seed)
    gen = SyntheticDataGenerator(LotkaVoltera(), sigma=0.01, perturb_initial=True,
                                 perturbation_scale=0.0, use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=X0, theta=TRUE,
                                     time_intervals=[(0.0, 4.0)],
                                     n_measurements=50)
    return t_b[0], meas_b[0]


def fit(t_meas, meas, N_shoot):
    prob = MultipleShooting(system=LotkaVoltera(), N_shoot=N_shoot,
                            gamma=np.ones(2), c0_cost=1.0, use_jax=True)
    prob.add_batch(meas, t_meas)
    theta_opt, _ = run_optimization_adaptive(
        prob, prob.make_full_theta(THETA0), n_iter=60, track_covariance=False)
    return theta_opt, NormalEquations.from_jacobian(*prob.solve(theta_opt))


def se_of(ne):
    return np.sqrt(np.diag(ne.covariance_theta(N_THETA)[0]))


# ---------------------------------------------------------------------------
# Эталон 1: single shooting (ограничений нет) — MS обязан совпасть
# ---------------------------------------------------------------------------
def test_matches_single_shooting():
    t_meas, meas = generate_data(0)
    _, ne_ms = fit(t_meas, meas, N_shoot=5)
    _, ne_ss = fit(t_meas, meas, N_shoot=1)

    assert ne_ms.n_cont > 0 and ne_ss.n_cont == 0
    rel = np.abs(se_of(ne_ms) / se_of(ne_ss) - 1.0)
    assert rel.max() < 0.05, f"MS и single shooting разошлись на {rel.max():.1%}"


# ---------------------------------------------------------------------------
# Ограничение не имеет масштаба: J_G можно умножить на что угодно
# ---------------------------------------------------------------------------
def test_invariant_to_constraint_scaling():
    t_meas, meas = generate_data(0)
    _, ne = fit(t_meas, meas, N_shoot=5)

    scaled = NormalEquations(H=ne.H, g=ne.g, J_G=1e3 * ne.J_G, R_G=1e3 * ne.R_G,
                             rss=ne.rss, n_rows=ne.n_rows)
    ratio = se_of(scaled) / se_of(ne)
    assert np.abs(ratio - 1.0).max() < 1e-6, f"зависимость от масштаба: {ratio}"


# ---------------------------------------------------------------------------
# Эталон 2: Монте-Карло — разброс оценок и покрытие интервалов
# ---------------------------------------------------------------------------
def test_coverage_monte_carlo():
    n_runs = 12
    theta_hat, se_pred = [], []
    for seed in range(n_runs):
        t_meas, meas = generate_data(seed)
        theta_opt, ne = fit(t_meas, meas, N_shoot=5)
        theta_hat.append(theta_opt[:N_THETA])
        se_pred.append(se_of(ne))
    theta_hat, se_pred = np.array(theta_hat), np.array(se_pred)

    empirical = theta_hat.std(axis=0, ddof=1)
    ratio = se_pred.mean(axis=0) / empirical
    # выборка мала, поэтому полоса широкая; прежняя формула давала 1.6-2.4
    assert np.all(ratio > 0.6) and np.all(ratio < 1.5), \
        f"предсказанное СКО не согласуется с разбросом: {ratio}"

    covered = (np.abs(theta_hat - TRUE) <= 1.96 * se_pred).mean(axis=0)
    assert np.all(covered >= 0.75), f"покрытие 95% интервалов слишком мало: {covered}"


# ---------------------------------------------------------------------------
# sigma^2 — только по измерениям; dof — по числу СВОБОДНЫХ неизвестных
# ---------------------------------------------------------------------------
def test_sigma2_and_dof():
    t_meas, meas = generate_data(0)
    _, ne = fit(t_meas, meas, N_shoot=5)
    _, sigma2, dof = ne.covariance_theta(N_THETA)

    n_state = 2
    assert dof == ne.n_rows - N_THETA - n_state
    assert np.isclose(sigma2, ne.rss / dof, rtol=1e-12)


# ---------------------------------------------------------------------------
# Диагностика идентифицируемости из той же ковариации
# ---------------------------------------------------------------------------
def test_correlation_diagnostics():
    t_meas, meas = generate_data(0)
    _, ne = fit(t_meas, meas, N_shoot=5)
    corr, cond = ne.correlation_theta(N_THETA)

    assert corr.shape == (N_THETA, N_THETA)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-10)
    assert np.allclose(corr, corr.T, atol=1e-10)
    assert np.abs(corr).max() <= 1.0 + 1e-10
    assert cond >= 1.0
    # у Лотки-Вольтерры alpha и beta почти неразличимы порознь
    assert abs(corr[0, 1]) > 0.9, f"corr(alpha, beta) = {corr[0, 1]:.3f}"


def test_correlation_matrix_helper():
    cov = np.array([[4.0, 1.0], [1.0, 1.0]])
    corr, cond = correlation_matrix(cov)
    assert np.allclose(corr, [[1.0, 0.5], [0.5, 1.0]])
    assert np.isclose(cond, 1.5 / 0.5)
