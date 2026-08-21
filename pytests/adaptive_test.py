import numpy as np
import pytest
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
        # a deliberately poor starting point: a fixed unlucky mu0 (0.01, say)
        # diverges from here, while the adaptive schedule must get through
        # without any manual mu
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
# The step against a dense solve of the same saddle system
# ---------------------------------------------------------------------------
def test_step_matches_dense_saddle_solve():
    """gn_step solves exactly the system its docstring claims.

    The reference is EXTERNAL: the saddle matrix is assembled densely and
    solved by numpy.linalg.solve, not by a second copy of our sparse code.
    Both the matrix and the right-hand side are checked:

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


def test_step_with_multipliers_matches_dense_saddle_solve():
    """gn_step(..., lam_dual): the AL shift of the right-hand side and nu.

    The same external reference: the dense saddle matrix with the constraint
    block shifted by -mu*lam_dual; nu is the dual part of the solution. Plus
    the invariant that lam_dual = 0 reproduces the plain step bit for bit.
    """
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])
    J, R, J_G, R_G = prob.solve(theta_full)

    mu, lam, lam_reg = 1.0, 1e-3, 1e-6
    ne = NormalEquations.from_jacobian(J, R, J_G, R_G)
    rng = np.random.default_rng(1)
    lam_dual = rng.standard_normal(ne.n_cont)

    delta, pred, nu = gn_step(ne, mu, lam, lam_reg, lam_dual=lam_dual)

    H = ne.H.toarray()
    n, m = H.shape[0], J_G.shape[0]
    D = lam_reg * np.eye(n) + lam * np.diag(np.maximum(np.diag(H), 1e-10))
    K = np.block([[H + D, J_G.toarray().T],
                  [J_G.toarray(), -mu * np.eye(m)]])
    sol_ref = np.linalg.solve(K, np.concatenate([ne.g, R_G - mu * lam_dual]))

    scale = np.abs(sol_ref).max()
    assert np.abs(delta - sol_ref[:n]).max() < 1e-9 * scale
    assert np.abs(nu - sol_ref[n:]).max() < 1e-9 * scale
    # nu = lam_dual + (J_G delta - R_G)/mu is the first-order update
    nu_formula = lam_dual + (J_G @ delta - R_G) / mu
    assert np.abs(nu - nu_formula).max() < 1e-8 * max(np.abs(nu).max(), 1.0)

    delta0, pred0 = gn_step(ne, mu, lam, lam_reg)
    delta_z, pred_z, _ = gn_step(ne, mu, lam, lam_reg,
                                 lam_dual=np.zeros(ne.n_cont))
    assert np.array_equal(delta0, delta_z)
    assert pred0 == pred_z


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
            assert pred > 0, f"pred <= 0 at mu={mu}, lam={lam}"
    # merit agrees with pred by construction: Phi_mu >= 0
    assert ne.merit(1.0) >= 0


# ---------------------------------------------------------------------------
# End-to-end identification without a manual mu (Attractor starts at theta = 0)
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
    # mu starts from the curvature and never grows on rejections
    assert hist["mu"][0] > 0
    assert hist["mu"][-1] <= hist["mu"][0] + 1e-15


# ---------------------------------------------------------------------------
# Powell's gate (rss_stall_tol): the attractor converges with FEW shots too
# ---------------------------------------------------------------------------
def test_attractor_converges_with_few_shoots():
    """Attractor, N_shoot=5, theta0=0 - pins down the rss_stall_tol gate.

    Without the gate mu collapses on the early iterations and the solution
    locks onto a consistent trajectory far from the measurements
    (rel_err ~ 1.2, rss ~ 9e3 at r_cont ~ 1e-10). With the gate the same case
    converges: rel_err ~ 3e-4 and the junction tightens to ~1e-10.
    """
    config = dict(SYSTEMS_CONFIG["Attractor"], N_shoot=5)
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=80,
                                                track_covariance=False)

    err = rel_err(theta_opt, config["true_params"])
    assert err < 1e-2, f"locked onto a consistent trajectory: rel_err={err:.3e}"
    # and the junction is pulled in by the penalty rather than abandoned
    assert hist["r_cont"][-1] < 1e-6


# ---------------------------------------------------------------------------
# The loop really MINIMIZES: the reference is the cost at the true parameters
# ---------------------------------------------------------------------------
def test_beats_cost_at_true_parameters():
    """The point found is no worse than the true parameters, by the same cost.

    The reference is external: the parameters the data was generated with. The
    optimum of a noisy problem need not coincide with the truth, but its cost
    must not be higher - otherwise the loop did not reach a minimum.
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
        f"the loop did not reach a minimum: cost {cost_opt:.6e} > true {cost_true:.6e}"
    assert rel_err(theta_opt, config["true_params"]) < 0.2


# ---------------------------------------------------------------------------
# Single shooting (J_G empty): plain LM, early stop on exact data
# ---------------------------------------------------------------------------
def test_single_shooting_early_stop():
    config = dict(SYSTEMS_CONFIG["LotkaVolterra"], N_shoot=1,
                  noise_sigma=1e-15)
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=50)

    # accuracy is limited by the integrator tolerances (RTOL=1e-5), not by noise
    assert rel_err(theta_opt, config["true_params"]) < 5e-3
    # convergence to the integrator floor stops the loop well before the limit
    assert len(hist["mu"]) - 1 < 50


# ---------------------------------------------------------------------------
# Works with the collocation integrator too (same solve contract)
# ---------------------------------------------------------------------------
def test_works_with_collocation():
    config = SYSTEMS_CONFIG["LotkaVolterra"]
    _, t_meas, meas = generate_data(config)
    prob = make_problem(config, t_meas, meas, cls=CollocationShooting, n_sub=1)
    theta_full = prob.make_full_theta(config["theta_init"])

    theta_opt, hist = run_optimization_adaptive(prob, theta_full, n_iter=40)

    err = rel_err(theta_opt, config["true_params"])
    assert err < 0.05, f"Estimation error too high: {err}"
