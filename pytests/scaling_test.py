"""Invariance to the units of the state (cont_scale).

The reference is external and exact: THE SAME PROBLEM written in different
units, so the estimate of theta must agree and the quality of the junction
must be restored once cont_scale is on. See docs/math.md.
"""

import numpy as np
import pytest
from casadi import vertcat
from commom_utils.ode_system import ODESystem, SyntheticDataGenerator
from commom_utils.systems import LotkaVoltera
from gauss_newton.problem import MultipleShooting
from gauss_newton.adaptive import run_optimization_adaptive

S = 1e3                      # the second coordinate is S times larger
TRUE = np.array([1.2, 0.4, 0.3, 0.1])
THETA0 = np.array([1.0, 0.5, 0.2, 0.05])


class LotkaVolteraScaled(ODESystem):
    """Lotka-Volterra with the second coordinate in units S times larger:
    y_s = S*y. The same system, different units."""

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
    meas_scaled[:, 1] *= S                       # the same data in other units
    gamma_scaled = np.array([1.0, 1.0 / S])      # gamma = 1/sigma, also in units

    err_ref, cont_ref = fit(LotkaVoltera, t_meas, meas, np.ones(2))
    err_bad, cont_bad = fit(LotkaVolteraScaled, t_meas, meas_scaled, gamma_scaled)
    scale = np.array([1.0, S]) if cont_scale == "explicit" else "auto"
    err_fix, cont_fix = fit(LotkaVolteraScaled, t_meas, meas_scaled, gamma_scaled,
                            cont_scale=scale)

    # without scaling the junction degrades by orders of magnitude
    assert cont_bad > 1e3 * cont_ref
    # with scaling it returns to the reference level
    assert cont_fix < 1e-3 * cont_bad
    # and the parameter estimate is no worse than the reference
    assert err_fix < 2 * max(err_ref, 1e-3)


def test_cont_scale_default_is_identity():
    """cont_scale=None -> weights of exactly 1, so the numbers are unchanged."""
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

    with pytest.raises(ValueError, match="scales must be > 0"):
        make(np.array([1.0, -1.0]))._cont_weights()
    with pytest.raises(ValueError, match="expected an array"):
        make(np.array([1.0, 1.0, 1.0]))._cont_weights()
    with pytest.raises(ValueError, match="expected None, 'auto' or an array"):
        make('rms')._cont_weights()
