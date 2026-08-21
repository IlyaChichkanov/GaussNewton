"""The assembled Jacobian against finite differences.

Why a separate test: every other check in pytests/ is mutual - the accumulated
path against the dense one, collocation against solve_ivp, adaptive against a
dense solve. They all share MultipleShooting.shoot_rows, so a systematic error
there would pass through any of them unnoticed.

The reference here is independent: central differences of the residual vector
r(theta_full) = [R; R_G] computed by the same solve(). This pins down the sign
convention of the whole code:

    [J; J_G] = -d[R; R_G]/d theta_full

(J is the Jacobian of the PREDICTIONS while the residual is R = W(y - h),
hence the minus; gn_step is built on exactly this convention, with
rhs = [J^T R; R_G]).
"""

import numpy as np
import pytest
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
    """The residual vector [R; R_G] - what is differentiated numerically."""
    _, R, _, R_G = problem.solve(theta_full)
    return np.concatenate([R, R_G])


def _fd_jacobian(problem, theta_full, rel_step=1e-7):
    """Central differences of d[R; R_G]/d theta_full."""
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
# Variational equations (solve_ivp)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_shoot", [1, 3])
def test_jacobian_matches_finite_differences(n_shoot):
    """J and J_G match -d[R; R_G]/dp for multiple shooting."""
    system = LotkaVoltera()
    t_meas, meas = _make_data(system, np.array([6.0, 5.0]),
                              np.array([1.2, 0.4, 0.3, 0.1]))

    problem = MultipleShooting(system, N_shoot=n_shoot, gamma=np.ones(2),
                               c0_cost=1.0, use_jax=False)
    # Tighter integrator tolerances than usual: otherwise the solve_ivp error
    # (1e-5 by default) dominates the difference quotient
    problem.integrator.ATOL = problem.integrator.RTOL = 1e-12
    problem.add_batch(meas, t_meas)

    theta_full = problem.make_full_theta(np.array([1.1, 0.45, 0.32, 0.09]))
    J_code = _assembled_jacobian(problem, theta_full)
    J_num = _fd_jacobian(problem, theta_full)

    assert J_code.shape == J_num.shape
    scale = max(np.abs(J_num).max(), 1e-12)
    err = np.abs(J_code + J_num).max() / scale
    assert err < 1e-6, f"J disagrees with finite differences: rel err {err:.2e}"


def test_jacobian_matches_finite_differences_with_input():
    """A system with an input and a NON-identity observation (h = x_0).

    The identity_observation fast path does not apply here, so this exercises
    the observation_batch branch (dh/dx, dh/dtheta).

    The input is deliberately smooth: the MassSpringDamper signal is
    discontinuous at t=1 and the adaptive solve_ivp steps over the kink with
    an uncontrolled error - see
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
    # A larger difference step than in the test above: the Integrator state
    # grows as t^2, the solve_ivp round-off noise at those magnitudes is
    # ~1e-11, and at rel_step=1e-7 the difference quotient drowns in it (the
    # error falls exactly as 1/h - checked). 1e-4 balances noise against
    # truncation.
    J_num = _fd_jacobian(problem, theta_full, rel_step=1e-4)

    scale = max(np.abs(J_num).max(), 1e-12)
    err = np.abs(J_code + J_num).max() / scale
    assert err < 1e-5, f"J disagrees with finite differences: rel err {err:.2e}"


def _fd_error(problem, theta_full):
    J_code = _assembled_jacobian(problem, theta_full)
    J_num = _fd_jacobian(problem, theta_full)
    return np.abs(J_code + J_num).max() / max(np.abs(J_num).max(), 1e-12)


def test_discontinuous_input_degrades_sensitivities():
    """Documents a limitation: a time-discontinuous input signal.

    MassSpringDamper holds its input at zero for t < 1 and then switches it on
    abruptly. The explicit adaptive solve_ivp knows nothing about the kink: it
    steps over it, error control stops working on the interval containing it,
    and the sensitivities lose several digits. Before the discontinuity is
    crossed the Jacobian agrees with the differences to ~1e-8.

    The test records the fact, not the desirable behaviour: if the integrator
    is taught to put a node at the discontinuity, this test will fail - and
    that will be the moment to update it.
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

    err_before = _fd_error(*build(0.9))    # entirely before the discontinuity
    err_across = _fd_error(*build(3.0))    # the interval contains t = 1

    assert err_before < 1e-7, (
        f"on a smooth stretch the Jacobian must converge: {err_before:.2e}")
    assert err_across > 100 * err_before, (
        "expected a loss of accuracy at the input discontinuity; if it is "
        f"gone the integrator was fixed, so update this test (before "
        f"{err_before:.2e}, "
        f"across {err_across:.2e})")


# ---------------------------------------------------------------------------
# Collocation: IND gives EXACT derivatives of the discrete scheme, so the
# agreement with differences of that same scheme should be markedly better
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
    assert err < 1e-7, f"the IND Jacobian disagrees with the scheme: {err:.2e}"


# ---------------------------------------------------------------------------
# Gradient of the normal equations: g = J^T R must be -1/2 grad ||R||^2
# ---------------------------------------------------------------------------
def test_gradient_matches_finite_differences():
    """g from NormalEquations agrees with the numeric gradient of ||R||^2."""
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

    # g = J^T R and d||R||^2/dp = 2 R^T dR/dp = -2 J^T R  =>  g = -g_num / 2
    scale = max(np.abs(g_num).max(), 1e-12)
    err = np.abs(g_code + g_num / 2.0).max() / scale
    assert err < 1e-6, f"g disagrees with the gradient of ||R||^2: rel err {err:.2e}"
