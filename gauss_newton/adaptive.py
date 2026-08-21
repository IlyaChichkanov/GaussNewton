"""Optimization layer: the Gauss-Newton step and the adaptive loop.

Knows only NormalEquations (H, g, J_G, R_G); how they were obtained is
irrelevant here. Theory and experiments: docs/math.md and
adaptive_regularization.ipynb.
"""
import numpy as np
from scipy.sparse import diags, hstack, vstack, eye as speye
from scipy.sparse.linalg import spsolve

from gauss_newton.normal_equations import (confidence_intervals,
                                           correlation_matrix,
                                           normal_equations_of)


def gn_step(ne, mu, lam, lambda_reg=0.0, lam_dual=None):
    """One step of the mu-regularized saddle system, plus pred.

        [[H + D, J_G^T], [J_G, -mu I]] [delta; nu] = [g; R_G],
        D = lambda_reg*I + lam*diag(H)

    pred = delta^T (g_eff + D delta) >= 0 is the reduction of Phi_mu predicted
    by the model; the caller treats pred <= 0 as a failed step.

    With lam_dual given, the constraint right-hand side is shifted by
    -mu*lam_dual (the augmented-Lagrangian variant) and the return value
    becomes (delta, pred, nu), nu being the first-order multiplier update.
    Otherwise the return value is (delta, pred) and the numbers are unchanged.
    See docs/math.md.
    """
    n = ne.H.shape[0]
    # Floor on the diagonal: a column the residuals are locally insensitive to
    # would otherwise get zero damping and an arbitrarily large step
    D = lambda_reg * speye(n) + lam * diags(np.maximum(ne.H.diagonal(), 1e-10))
    nu = None
    if ne.n_cont > 0:
        K = vstack([hstack([ne.H + D, ne.J_G.T]),
                    hstack([ne.J_G, -mu * speye(ne.n_cont)])]).tocsr()
        rhs_G = ne.R_G if lam_dual is None else ne.R_G - mu * lam_dual
        sol = spsolve(K, np.concatenate([ne.g, rhs_G]))
        delta, nu = sol[:n], sol[n:]
        g_eff = ne.g + (1.0 / mu) * (ne.J_G.T @ ne.R_G)
        if lam_dual is not None:
            g_eff = g_eff - ne.J_G.T @ lam_dual
    else:
        g_eff = ne.g
        delta = spsolve((ne.H + D).tocsr(), ne.g)
    pred = float(delta @ g_eff + delta @ (D @ delta))
    if lam_dual is not None:
        return delta, pred, nu
    return delta, pred


def run_optimization_adaptive(problem, theta_full, n_iter=40,
                              lam0=1e-3, lambda_reg=0.0,
                              mu_rule='curvature', mu_dec=0.5,
                              viol_target=0.25, rss_stall_tol=0.99, kappa=0.1,
                              mu_min=1e-8, mu_max=1e8,
                              rho_accept=0.0, max_rejects=8,
                              track_covariance=True, verbose=False):
    """Gauss-Newton iterations with adaptive lambda (Nielsen) and mu (Powell).

    `problem` is anything with `solve` (MultipleShooting, CollocationShooting)
    or with `normal_equations` (the ...Accum classes). n_iter is an upper
    bound: the loop stops on a run of rejections, on stalling, or on pred ~ 0.

    Returns (theta_full, hist); the arguments and the contents of hist are
    documented in docs/api-reference.md, the schedules in docs/math.md.
    """
    theta_full = theta_full.copy()
    n_theta = problem.system.dims()[1]
    try:
        ne = normal_equations_of(problem, theta_full)
    except RuntimeError as exc:
        raise RuntimeError(
            "The integrator failed at the INITIAL point theta0, before the "
            "first Gauss-Newton step. For collocation: increase n_sub, relax "
            "newton_tol or raise newton_maxiter; otherwise pick a more "
            f"plausible theta0. Original error: {exc}") from exc

    def mu_ratio(ne_):
        return float(np.clip(ne_.cont_sq() / (kappa * max(ne_.rss, 1e-300)),
                             mu_min, mu_max))

    lam, nu_esc = lam0, 2.0
    if ne.n_cont == 0:
        mu = 1.0  # unused: merit = ||R||^2
    elif mu_rule == 'curvature':
        mu = float(np.clip(ne.mu_curvature(), mu_min, mu_max))
    else:
        mu = mu_ratio(ne)
    prev_cont, prev_rss = ne.cont_sq(), ne.rss

    hist = dict(theta=[], cost=[], mu=[], lam=[], r_meas=[], r_cont=[],
                ci_low=[], ci_high=[], corr_cond=[], accepted=[], n_solves=1)

    def record():
        hist['theta'].append(theta_full.copy())
        hist['cost'].append(ne.cost())
        hist['mu'].append(mu)
        hist['lam'].append(lam)
        hist['r_meas'].append(ne.rss / max(ne.n_rows, 1))
        hist['r_cont'].append(ne.cont_sq() / max(ne.n_cont, 1))
        if track_covariance:
            cov, _, dof = ne.covariance_theta(n_theta)
            ci_low, ci_high = confidence_intervals(theta_full[:n_theta], cov, dof)
            hist['ci_low'].append(ci_low)
            hist['ci_high'].append(ci_high)
            # Identifiability from the covariance already computed: a growing
            # corr_cond means the parameters drift into a flat valley
            hist['corr_cond'].append(correlation_matrix(cov)[1])

    record()
    rejects = stalls = 0
    for it in range(n_iter):
        delta, pred = gn_step(ne, mu, lam, lambda_reg)
        ok = np.all(np.isfinite(delta)) and pred > 0
        if ok:
            theta_trial = theta_full + delta
            try:
                ne_trial = normal_equations_of(problem, theta_trial)
                hist['n_solves'] += 1
                phi0 = ne.merit(mu)
                phi1 = ne_trial.merit(mu)
                rho = (phi0 - phi1) / pred if np.isfinite(phi1) else -np.inf
            except RuntimeError:
                # The integrator could not handle the trial point (for example
                # collocation Newton did not converge): a failed step, not a
                # fatal error
                rho = -np.inf
        else:
            rho = -np.inf

        if np.isfinite(rho) and rho > rho_accept:
            theta_full, ne = theta_trial, ne_trial
            lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-12)
            nu_esc, rejects = 2.0, 0
            if ne.n_cont > 0:
                cont, rss = ne.cont_sq(), ne.rss
                if mu_rule == 'curvature':
                    # Powell with a gate: tighten the penalty only when the
                    # junction is not improving on its own AND the measurements
                    # are already squeezed out (see docs/math.md)
                    if cont > viol_target * prev_cont and rss > rss_stall_tol * prev_rss:
                        mu = max(mu * mu_dec, mu_min)
                    prev_cont, prev_rss = cont, rss
                else:
                    mu = mu_ratio(ne)
            hist['accepted'].append(it)
            stalls = stalls + 1 if (phi0 - phi1) <= 1e-10 * max(phi0, 1e-300) else 0
            if verbose:
                print(f'Iter {it:3d} | accept rho={rho:6.3f} | '
                      f'cost {hist["cost"][-1]:.3e} -> {ne.cost():.3e} | '
                      f'mu {mu:.2e} | lam {lam:.2e}')
        else:
            lam = min(lam * nu_esc, 1e10)
            nu_esc *= 2.0
            rejects += 1
            if verbose:
                print(f'Iter {it:3d} | reject rho={rho:6.3f} | lam {lam:.2e}')

        record()
        if rejects >= max_rejects or stalls >= 2:
            break
        if ok and pred < 1e-12 * max(ne.merit(mu), 1e-300):
            break

    for key in ('theta', 'ci_low', 'ci_high'):
        hist[key] = np.array(hist[key]) if hist[key] else np.zeros((0, n_theta))
    return theta_full, hist
