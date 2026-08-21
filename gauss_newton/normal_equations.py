"""Normal equations of the Gauss-Newton step: H = J^T J and g = J^T r.

Two ways to get them - from an explicit J, or by accumulation over the
measurements without ever forming J. See docs/architecture.md and
docs/math.md.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.sparse import bmat, csr_matrix, eye as speye
from scipy.sparse.linalg import splu

from gauss_newton.problem import MultipleShooting
from gauss_newton.collocation_shooting import CollocationShooting


@dataclass
class NormalEquations:
    """Normal equations at a point, without J itself.

    H, g       : H = J^T J, g = J^T r over the measurement residuals;
    J_G, R_G   : continuity rows, kept separate - continuity is a CONSTRAINT,
                 not an observation, both for the step and for the covariance;
    rss        : ||r||^2 over the measurements; n_rows: measurement row count.
    """
    H: object
    g: np.ndarray
    J_G: object
    R_G: np.ndarray
    rss: float
    n_rows: int

    @classmethod
    def from_jacobian(cls, J, R, J_G, R_G):
        """Same quantities, but starting from an explicit J."""
        return cls(H=(J.T @ J).tocsr(), g=J.T @ R, J_G=J_G, R_G=R_G,
                   rss=float(R @ R), n_rows=J.shape[0])

    @property
    def n_cont(self):
        return self.J_G.shape[0]

    def cont_sq(self):
        """||R_G||^2."""
        return float(self.R_G @ self.R_G) if self.R_G.size else 0.0

    def cost(self):
        """||r||^2 + ||R_G||^2."""
        return self.rss + self.cont_sq()

    def merit(self, mu):
        """Phi_mu = ||r||^2 + (1/mu)||R_G||^2 - what the step minimizes."""
        return self.rss + self.cont_sq() / mu if self.R_G.size else self.rss

    def mu_curvature(self):
        """Starting mu: ||J_G||_F^2 / tr(H), using tr(J^T J) instead of J."""
        if not self.R_G.size:
            return 1.0
        return float((self.J_G.multiply(self.J_G)).sum()
                     / max(self.H.diagonal().sum(), 1e-300))

    def covariance_theta(self, n_theta, ridge=1e-8):
        """Covariance of theta under exact continuity -> (cov, sigma2, dof).

        The theta block of the inverse KKT matrix [[H, J_G^T], [J_G, 0]], which
        is the projection of H^-1 onto the tangent subspace of the constraints;
        sigma^2 = ||r||^2 / dof uses the measurement residual only. Computing
        it as (H + J_G^T J_G)^-1 instead is wrong - see docs/math.md.
        """
        n_params = self.H.shape[0]
        m = self.n_cont
        dof = max(self.n_rows + m - n_params, 1)
        sigma2 = self.rss / dof

        H_reg = self.H + ridge * speye(n_params)
        if m:
            K = bmat([[H_reg, self.J_G.T], [self.J_G, None]], format='csc')
        else:
            K = H_reg.tocsc()                      # single shooting: plain least squares
        rhs = np.zeros((n_params + m, n_theta))
        rhs[:n_theta, :] = np.eye(n_theta)
        try:
            X = splu(K).solve(rhs)
        except RuntimeError:
            X = np.linalg.pinv(K.toarray()) @ rhs
        return sigma2 * X[:n_theta, :], sigma2, dof

    def correlation_theta(self, n_theta, ridge=1e-8):
        """Correlation matrix of theta and its condition number.

        |corr| near 1 means the parameters are only distinguishable in
        combination; needs no extra solve beyond the covariance.
        """
        cov, _, _ = self.covariance_theta(n_theta, ridge=ridge)
        return correlation_matrix(cov)


class AccumulateMixin:
    """Builds H and g by accumulation, without ever forming J."""

    def normal_equations(self, theta_full):
        n_state, n_theta, _ = self.system.dims()
        H_theta = np.zeros((n_theta, n_theta))
        g_theta = np.zeros(n_theta)
        H_theta_c, H_c, g_c = [], [], []         # one block per shot
        J_G_batches, R_G_batches = [], []
        rss = 0.0
        n_rows = 0

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches,
                    self.t_eval_measurements_batches)):
            rows = self.shoot_rows(theta_full, state_measured, t_meas, batch)

            for shoot in rows:
                J_theta, J_c, r = shoot.J_theta, shoot.J_c, shoot.r
                # Sums over the points 'm' of a shot: the vectorized form of
                # H_{k+1} = H_k + J^T J and g_{k+1} = g_k + J^T r
                H_theta += np.einsum('mop,moq->pq', J_theta, J_theta)
                H_theta_c.append(np.einsum('mop,moc->pc', J_theta, J_c))
                H_c.append(np.einsum('moc,mod->cd', J_c, J_c))
                g_theta += np.einsum('mop,mo->p', J_theta, r)
                g_c.append(np.einsum('moc,mo->c', J_c, r))
                rss += float((r * r).sum())
                n_rows += r.size

            J_G, R_G = self.continuity_rows(rows)
            J_G_batches.append(J_G)
            R_G_batches.append(R_G)

        # Arrow structure: the theta row on top, one block per shot on the
        # diagonal, zeros between different shots
        T = len(H_c)
        blocks = [[csr_matrix(H_theta)] + [csr_matrix(B) for B in H_theta_c]]
        for j in range(T):
            row = [csr_matrix(H_theta_c[j].T)] + [None] * T
            row[1 + j] = csr_matrix(H_c[j])
            blocks.append(row)

        return NormalEquations(
            H=bmat(blocks, format='csr'),
            g=np.concatenate([g_theta] + g_c),
            J_G=self._concatenate_jacobian_batches(J_G_batches),
            R_G=np.concatenate(R_G_batches),
            rss=rss, n_rows=n_rows)


class MultipleShootingAccum(AccumulateMixin, MultipleShooting):
    """Multiple shooting (variational equations) with accumulated H and g."""


class CollocationShootingAccum(AccumulateMixin, CollocationShooting):
    """Multiple shooting on Radau IIA collocation with accumulated H and g."""


def normal_equations_of(problem, theta_full):
    """H and g for any problem: by accumulation if it can, else through J."""
    if hasattr(problem, 'normal_equations'):
        return problem.normal_equations(theta_full)
    return NormalEquations.from_jacobian(*problem.solve(theta_full))


def correlation_matrix(cov):
    """(corr, cond) from a covariance matrix."""
    d = np.sqrt(np.diag(cov))
    scale = np.where(d > 0, d, 1.0)
    corr = cov / np.outer(scale, scale)
    s = np.linalg.svd(corr, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return corr, cond


def confidence_intervals(theta_opt, cov, dof, alpha=0.05):
    """Two-sided Student intervals: theta_i +- t_{alpha/2, dof} * sqrt(Cov_ii)."""
    se = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - alpha / 2, df=dof)
    return theta_opt - t_crit * se, theta_opt + t_crit * se
