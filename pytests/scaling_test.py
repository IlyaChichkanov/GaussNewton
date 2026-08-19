# -*- coding: utf-8 -*-
"""Инвариантность к единицам измерения состояний (cont_scale).

Блок -mu*I седловой системы одинаково взвешивает невязки стыковки всех
состояний, поэтому без масштабирования метод зависит от единиц: одна и та же
задача, где вторая координата выражена в 1000 раз более крупных единицах,
сходится к той же точности, но стыковку затягивает на порядки хуже.

Эталон здесь внешний и точный: ЗАДАЧА ТА ЖЕ, переписана в других единицах,
поэтому оценка theta обязана совпасть, а качество стыковки при включённом
cont_scale — восстановиться.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
from casadi import vertcat

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.ode_system import ODESystem, SyntheticDataGenerator
from commom_utils.systems import LotkaVoltera
from gauss_newton.problem import MultipleShooting
from gauss_newton.adaptive import run_optimization_adaptive

S = 1e3                      # вторая координата в S раз крупнее
TRUE = np.array([1.2, 0.4, 0.3, 0.1])
THETA0 = np.array([1.0, 0.5, 0.2, 0.05])


class LotkaVolteraScaled(ODESystem):
    """Лотка-Вольтерра, где вторая координата измеряется в единицах в S раз
    крупнее: y_s = S*y. Та же система, другие единицы."""

    def __init__(self):
        super().__init__(2, 4, 0)

    def get_derivative(self, state, theta, u):
        x, ys = state[0], state[1]
        alpha, beta, gamma, delta = theta[0], theta[1], theta[2], theta[3]
        return vertcat(alpha * x - (beta / S) * x * ys,
                       delta * x * ys - gamma * ys)

    def observation(self, state, theta, u):
        return vertcat(state[0], state[1])


def data():
    np.random.seed(0)
    gen = SyntheticDataGenerator(LotkaVoltera(), sigma=0.01, perturb_initial=True,
                                 perturbation_scale=0.0, use_jax=True)
    t_b, meas_b, _, _ = gen.generate(c0=np.array([6.0, 5.0]), theta=TRUE,
                                     time_intervals=[(0.0, 4.0)],
                                     n_measurements=50)
    return t_b[0], meas_b[0]


def fit(cls, t_meas, meas, gamma, cont_scale=None):
    prob = MultipleShooting(system=cls(), N_shoot=5, gamma=gamma, c0_cost=1.0,
                            use_jax=True, cont_scale=cont_scale)
    prob.add_batch(meas, t_meas)
    theta_opt, hist = run_optimization_adaptive(
        prob, prob.make_full_theta(THETA0), n_iter=60, track_covariance=False)
    rel_err = float(np.max(np.abs((theta_opt[:4] - TRUE) / TRUE)))
    return rel_err, hist['r_cont'][-1]


@pytest.mark.parametrize("cont_scale", ["explicit", "auto"])
def test_continuity_scaling_restores_reference_quality(cont_scale):
    t_meas, meas = data()
    meas_scaled = meas.copy()
    meas_scaled[:, 1] *= S                       # те же данные в других единицах
    gamma_scaled = np.array([1.0, 1.0 / S])      # gamma = 1/sigma, тоже в единицах

    err_ref, cont_ref = fit(LotkaVoltera, t_meas, meas, np.ones(2))
    err_bad, cont_bad = fit(LotkaVolteraScaled, t_meas, meas_scaled, gamma_scaled)
    scale = np.array([1.0, S]) if cont_scale == "explicit" else "auto"
    err_fix, cont_fix = fit(LotkaVolteraScaled, t_meas, meas_scaled, gamma_scaled,
                            cont_scale=scale)

    # без масштабирования стыковка деградирует на порядки
    assert cont_bad > 1e3 * cont_ref
    # с масштабированием возвращается к уровню эталона
    assert cont_fix < 1e-3 * cont_bad
    # оценка параметров при этом не хуже эталонной
    assert err_fix < 2 * max(err_ref, 1e-3)


def test_cont_scale_default_is_identity():
    """cont_scale=None -> веса ровно 1: числа прежние (regression_test)."""
    t_meas, meas = data()
    prob = MultipleShooting(system=LotkaVoltera(), N_shoot=5, gamma=np.ones(2),
                            use_jax=True)
    prob.add_batch(meas, t_meas)
    assert np.array_equal(prob._cont_weights(), np.ones(2))


def test_cont_scale_validation():
    t_meas, meas = data()

    def make(cont_scale):
        prob = MultipleShooting(system=LotkaVoltera(), N_shoot=5,
                                gamma=np.ones(2), use_jax=True,
                                cont_scale=cont_scale)
        prob.add_batch(meas, t_meas)
        return prob

    with pytest.raises(ValueError, match="масштабы"):
        make(np.array([1.0, -1.0]))._cont_weights()
    with pytest.raises(ValueError, match="ожидался массив"):
        make(np.array([1.0, 1.0, 1.0]))._cont_weights()
    with pytest.raises(ValueError, match="ожидалось"):
        make('rms')._cont_weights()
