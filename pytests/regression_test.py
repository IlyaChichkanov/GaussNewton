"""Numerical reference for the mathematical core.

Renames are safe; a permuted einsum axis, a different order of evaluation, or
a rewritten unpacking are not - they change the NUMBERS without failing. No
other test catches that reliably: the mutual checks move together with both
sides, and an end-to-end identification converges to the same parameters even
with the initial-state Jacobian columns swapped.

Frozen here for four fixed problems: the raw matrices of a Gauss-Newton step -
J, R, J_G, R_G (dense assembly), H, g (accumulation), delta and pred (step).
A refactor must be numerically a no-op.

The four cases cover every branch a refactor can touch:

    lv_scipy        variational equations through solve_ivp (numpy branch)
    lv_jax          the same equations through jax odeint (_variational_rhs_jax)
    lv_colloc       Radau IIA collocation + IND (its own march and recursions)
    integrator_obs  NON-identity observation h = x_0 with an input u(t)
                    (the observation_batch branch: dh/dx, dh/dtheta)

Regenerate with `GN_REGEN_REFERENCE=1 uv run pytest
pytests/regression_test.py`, and only when the change in numbers is deliberate
and explainable - otherwise the test loses its point.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from commom_utils.systems import Integrator, LotkaVoltera
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.normal_equations import (MultipleShootingAccum,
                                           CollocationShootingAccum)
from gauss_newton.adaptive import gn_step

REFERENCE = Path(__file__).parent / "data" / "reference.npz"
REGEN = os.environ.get("GN_REGEN_REFERENCE", "0") not in ("0", "", "false", "False")

# The tolerance differs per integrator, and here is why.
#
# Collocation runs on a FIXED element grid: the sequence of operations is the
# same everywhere and only the last bits can differ.
#
# solve_ivp and jax odeint choose the step ADAPTIVELY, by comparing an error
# estimate against a tolerance. Slightly different arithmetic (another CPU,
# another BLAS build) gives a different sequence of steps and therefore a
# different result. Measured: on a GitHub Actions runner integrator_obs
# diverged from the local reference by 1.0e-8 at a tolerance of 1e-10. That is
# not a Jacobian error - 1e-8 is the accuracy of the integrator itself.
#
# 1e-6 is still orders of magnitude stricter than what this test must catch: a
# swapped axis or a wrong einsum index changes elements by O(1), not in the
# eighth digit.
TOL_EXACT = dict(rtol=1e-10, atol=1e-12)     # fixed step
TOL_ADAPTIVE = dict(rtol=1e-6, atol=1e-10)   # adaptive step

# Fixed linearization point for the Gauss-Newton step
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
# Problem builders, each returning (problem, theta_full). The integrator
# tolerances are tight so that the reference is determined by the mathematics
# rather than by the step sizes solve_ivp happens to choose.
# ---------------------------------------------------------------------------
def _lotka(use_jax):
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]),
                              t_end=2.0, n_meas=21)
    problem = MultipleShootingAccum(system, N_shoot=3, gamma=np.ones(2),
                                    c0_cost=1.0, use_jax=use_jax)
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
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
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
    problem.add_batch(meas, t_meas)
    return problem, problem.make_full_theta(np.array([0.85]),
                                            c0_init_method='zeros')


# case -> (builder, tolerance)
CASES = {
    "lv_scipy": (_case_lv_scipy, TOL_ADAPTIVE),
    "lv_jax": (_case_lv_jax, TOL_ADAPTIVE),
    "lv_colloc": (_case_lv_colloc, TOL_EXACT),
    "integrator_obs": (_case_integrator_obs, TOL_ADAPTIVE),
}


def _snapshot(build):
    """Raw material of a Gauss-Newton step: dense assembly, accumulation, step."""
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
        pytest.skip("the reference was just regenerated, nothing to compare with")

    build, tol = CASES[case]
    got = _snapshot(build)
    for key, value in got.items():
        expected = reference[f"{case}__{key}"]
        assert value.shape == expected.shape, \
            f"{case}.{key}: shape {value.shape}, reference {expected.shape}"
        np.testing.assert_allclose(
            value, expected, **tol,
            err_msg=f"{case}.{key} differs from the reference: the behaviour "
                    f"changed, not just the names")
