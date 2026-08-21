"""Collocation integrator: Radau IIA stages solved by a CasADi rootfinder,
sensitivities by internal numerical differentiation.

A drop-in replacement for VariationalIntegrator, suitable for stiff systems.
See docs/math.md for the scheme and docs/pitfalls.md for the rootfinder
settings.
"""
import os

import numpy as np
import casadi as ca

from commom_utils.ode_system import CompiledModel
from commom_utils.sensitivity import (SensitivityTrajectory,
                                      group_by_grid_length, initial_flat_row)


class RadauTables:
    """Radau IIA tables: nodes, differentiation matrix, Butcher table.

    The polynomial of degree K is built on {0, tau_1..tau_K}; tilde_D holds the
    basis derivatives at the collocation nodes, split into the left-edge column
    d0 and the invertible block D1. Butcher table a = D1^-1.
    """

    _NODES = {
        1: [1.0],
        2: [1.0 / 3.0, 1.0],
        3: [(4.0 - np.sqrt(6.0)) / 10.0, (4.0 + np.sqrt(6.0)) / 10.0, 1.0],
    }

    def __init__(self, K=3):
        if K not in self._NODES:
            raise ValueError(f"K={K} is not supported (available: {sorted(self._NODES)})")
        self.K = K
        self.tau = np.array(self._NODES[K])
        self.nodes = np.concatenate([[0.0], self.tau])

        tilde_D = self._differentiation_matrix(self.nodes, self.tau)  # K x (K+1)
        self.d0 = tilde_D[:, 0]
        self.D1 = tilde_D[:, 1:]
        self.butcher_a = np.linalg.inv(self.D1)

        # Exactness on constants: zero row sums and A = 1 (x) I
        assert np.allclose(tilde_D.sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose(-self.butcher_a @ self.d0, np.ones(K), atol=1e-12)

    @staticmethod
    def _differentiation_matrix(nodes, at):
        """D[k, j] = dl_j/dtau at the points at[k]."""
        n = len(nodes)
        D = np.zeros((len(at), n))
        for k, x in enumerate(at):
            for j in range(n):
                s = 0.0
                for m in range(n):
                    if m == j:
                        continue
                    prod = 1.0 / (nodes[j] - nodes[m])
                    for r in range(n):
                        if r != j and r != m:
                            prod *= (x - nodes[r]) / (nodes[j] - nodes[r])
                    s += prod
                D[k, j] = s
        return D


class CollocationStepFunctions:
    """Builds and caches the CasADi functions of one element step.

    Responsible for the symbolic side only: stage residual, rootfinder, the
    (step_sens, step_x) pair and their mapaccum/map wrappers. The march itself
    and the convergence policy live in CollocationIntegrator.
    """

    def __init__(self, model: CompiledModel, tables: RadauTables,
                 newton_tol, newton_maxiter,
                 rootfinder_plugin, rootfinder_options, n_threads):
        self.model = model
        self.tables = tables
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.rootfinder_plugin = rootfinder_plugin
        self.rootfinder_options = dict(rootfinder_options or {})
        self.n_threads = n_threads

        # B (x) I - the Butcher table acting on the stacked stage vector
        self._B_kron = np.kron(tables.butcher_a, np.eye(model.nx))
        self._step_fns = None
        # Separate caches on purpose: True == 1 in Python, so (n, True) and
        # (n, 1) would collide in a single dict
        self._accum_cache = {}
        self._map_cache = {}

    def _stage_residual_fn(self):
        """Step symbols and the stage residual.

        Phi(z; x_prev, theta, u, h) = z - A x_prev - h (B (x) I) F(z),
        A = 1_K (x) I. Returns (syms, phi_fn) with
        syms = (z, x_prev, theta, u_flat, h, step_param).
        """
        K = self.tables.K
        nx, n_theta, nu = self.model.nx, self.model.n_theta, self.model.nu

        z = ca.MX.sym('z', K * nx)
        x_prev = ca.MX.sym('x_prev', nx)
        theta = ca.MX.sym('theta', n_theta)
        u_flat = ca.MX.sym('u_flat', K * nu)
        h = ca.MX.sym('h')

        # F(z): the right-hand side at every stage. vertsplit cuts the stacked
        # vectors into K stage blocks, so each block is indexed on its own axes
        # instead of through k*nx + i arithmetic
        z_stages = ca.vertsplit(z, nx)
        u_stages = ca.vertsplit(u_flat, nu) if nu else [u_flat] * K
        th = [theta[i] for i in range(n_theta)]
        F = ca.vcat([self.model._f_ca(*(zk[i] for i in range(nx)),
                                      *(uk[j] for j in range(nu)), *th)
                     for zk, uk in zip(z_stages, u_stages)])

        B_kron = ca.DM(self._B_kron)
        step_param = ca.vertcat(x_prev, theta, u_flat, h)
        Phi = z - ca.repmat(x_prev, K, 1) - h * ca.mtimes(B_kron, F)
        phi_fn = ca.Function('colloc_res', [z, step_param], [Phi])
        return (z, x_prev, theta, u_flat, h, step_param), phi_fn

    def _make_rootfinder(self, phi_fn):
        """Stage Newton: convergence options and the rootfinder itself."""
        rf_opts = {
            'abstol': self.newton_tol,       # residual: max|Phi| < tol
            'abstolStep': self.newton_tol,   # step: max|dz| < tol; the criteria
                                             # act as OR, which saves large state
                                             # scales where abstol hits the
                                             # round-off floor
            'max_iter': self.newton_maxiter,
            'line_search': True,
            'error_on_fail': False,          # quiet: the march checks stage_res
                                             # itself, without C++ dumps
        }
        rf_opts.update(self.rootfinder_options)
        return ca.rootfinder('colloc_rf', self.rootfinder_plugin,
                             phi_fn, rf_opts)

    def step_pair(self):
        """One element step as a pair of CasADi functions (built once).

        step_sens: (x_prev, theta, u, h) -> (x_next, Psi, Gamma, stage_res);
        step_x is the same without sensitivities. Psi and Gamma are derivatives
        of the rootfinder output, i.e. exact derivatives of the discrete scheme.
        """
        if self._step_fns is not None:
            return self._step_fns

        (_, x_prev, theta, u_flat, h, step_param), phi_fn = \
            self._stage_residual_fn()
        stage_solver = self._make_rootfinder(phi_fn)

        K, nx = self.tables.K, self.model.nx
        z0 = ca.repmat(x_prev, K, 1)          # the solution does not depend on z0
        z_sol = stage_solver(z0, step_param)
        # Radau IIA is stiffly accurate (tau_K = 1): the last node is the right
        # edge of the element, so x_next is the last stage block
        x_next = z_sol[-nx:]
        Psi = ca.jacobian(x_next, x_prev)     # implicit function theorem
        Gamma = ca.jacobian(x_next, theta)
        # Scaled residual at the solution - the convergence marker; on a
        # divergence (inf/nan in z) it is inf/nan as well
        stage_res = ca.norm_inf(phi_fn(z_sol, step_param)) \
            / (1.0 + ca.norm_inf(z_sol))

        step_sens = ca.Function('colloc_step_sens', [x_prev, theta, u_flat, h],
                                [x_next, Psi, Gamma, stage_res])
        step_x = ca.Function('colloc_step_x', [x_prev, theta, u_flat, h],
                             [x_next, stage_res])
        self._step_fns = (step_sens, step_x)
        return self._step_fns

    def mapaccum(self, n_elems, with_sens):
        """March over the n_elems elements of one shot; only x accumulates."""
        key = (n_elems, with_sens)
        fn = self._accum_cache.get(key)
        if fn is None:
            step_sens, step_x = self.step_pair()
            base = step_sens if with_sens else step_x
            fn = base.mapaccum(n_elems)
            self._accum_cache[key] = fn
        return fn

    def mapmarch(self, n_elems, group_size):
        """map('thread') over mapaccum: a group of shots marched in parallel."""
        key = (n_elems, group_size)
        fn = self._map_cache.get(key)
        if fn is None:
            march = self.mapaccum(n_elems, with_sens=True)
            fn = march.map(group_size, 'thread',
                           min(self.n_threads, group_size))
            self._map_cache[key] = fn
        return fn


class CollocationIntegrator:
    """Radau IIA sensitivity integrator for stiff systems.

    Same output contract as VariationalIntegrator; holds the same
    CompiledModel by composition. Arguments are documented in
    docs/api-reference.md.
    """

    # A converged element must satisfy stage_res <= RES_SAFETY * newton_tol
    # (the slack covers stopping on the step rather than on the residual)
    RES_SAFETY = 10.0

    def __init__(self, model, K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25,
                 rootfinder_plugin='newton', rootfinder_options=None,
                 n_threads=None):
        self.model = (model if isinstance(model, CompiledModel)
                      else CompiledModel(model))
        if int(n_sub) < 1:
            raise ValueError(f"n_sub={n_sub}: must be >= 1")
        if float(newton_tol) <= 0:
            raise ValueError(f"newton_tol={newton_tol}: must be > 0")
        if int(newton_maxiter) < 1:
            raise ValueError(f"newton_maxiter={newton_maxiter}: must be >= 1")
        self.colloc = RadauTables(K)
        self.n_sub = int(n_sub)
        self.newton_tol = float(newton_tol)
        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 1)
        self.n_threads = int(n_threads)
        self.newton_maxiter = int(newton_maxiter)

        self.steps = CollocationStepFunctions(
            self.model, self.colloc, self.newton_tol, self.newton_maxiter,
            rootfinder_plugin, rootfinder_options, self.n_threads)
        self._node_inp_cache = {}

    # ------------------------------------------------------------------
    # Helpers shared by the single and batched march
    # ------------------------------------------------------------------
    def _accumulate_sens(self, c0, x_all, Psi_all, Gamma_all, n_pts):
        """Element recursions -> the flat layout of the integrator contract.

        S_c <- Psi_e S_c (S_c[0] = I), S_th <- Psi_e S_th + Gamma_e (S_th[0]=0).
        x_all (nx, n_elems), Psi_all (n_elems, nx, nx),
        Gamma_all (n_elems, nx, n_theta). The points of t_eval are the ends of
        every n_sub-th element, so a point is recorded once per n_sub elements.
        The order of the multiplications is frozen by regression_test at 1e-10.
        """
        nx, n_theta = self.model.nx, self.model.n_theta
        S_th, S_c = np.zeros((nx, n_theta)), np.eye(nx)
        xs, S_ths, S_cs = [np.asarray(c0, float)], [S_th], [S_c]
        for e in range(x_all.shape[1]):
            S_c = Psi_all[e] @ S_c
            S_th = Psi_all[e] @ S_th + Gamma_all[e]
            if (e + 1) % self.n_sub == 0:
                xs.append(np.asarray(x_all[:, e], float))
                S_ths.append(S_th)
                S_cs.append(S_c)
        # len(xs) == n_pts by construction; __post_init__ checks the shapes
        traj = SensitivityTrajectory(np.array(xs), np.array(S_ths),
                                     np.array(S_cs))
        return traj.pack()

    def _node_inputs(self, t_eval):
        """Inputs u at the nodes of every sub-element: (N-1, n_sub, K, nu)."""
        # An array is unhashable, so the key is its bytes; safe because t_eval
        # always arrives here already normalized to a float array
        key = (t_eval.tobytes(), self.n_sub)
        cached = self._node_inp_cache.get(key)
        if cached is not None:
            return cached

        K, nu = self.colloc.K, self.model.nu
        n_int = len(t_eval) - 1
        h_int = (t_eval[1:] - t_eval[:-1]) / self.n_sub          # (n_int,)
        # t_nodes[i, s, k] = t_eval[i] + (s + tau_k) * h_int[i]
        offs = (np.arange(self.n_sub)[:, None] + self.colloc.tau[None, :])
        t_nodes = t_eval[:-1, None, None] + offs[None, :, :] * h_int[:, None, None]
        flat_t = t_nodes.ravel()

        # For nu == 0 this returns (len, 0), so the reshape needs no branch
        inp = self.model._inp_on_times(flat_t).reshape(n_int, self.n_sub, K, nu)

        self._node_inp_cache[key] = inp
        return inp

    def _march_inputs(self, theta, t_eval):
        """mapaccum inputs, one column per element: theta, u, h."""
        K, nu = self.colloc.K, self.model.nu
        n_elems = (len(t_eval) - 1) * self.n_sub
        inp_nodes = self._node_inputs(t_eval)                  # (n_int, n_sub, K, nu)
        u_cols = inp_nodes.reshape(n_elems, K * nu).T          # (K*nu, n_elems)
        h_cols = np.repeat((t_eval[1:] - t_eval[:-1]) / self.n_sub,
                           self.n_sub).reshape(1, n_elems)
        theta_cols = np.tile(theta.reshape(-1, 1), (1, n_elems))
        return theta_cols, u_cols, h_cols

    def _unstack_mapaccum(self, x_stack, Psi_stack, Gamma_stack, n_elems):
        """Unpack the mapaccum outputs of one shot.

        mapaccum concatenates matrix outputs horizontally: Psi_stack is
        [Psi_1 | Psi_2 | ...] of size (nx, n_elems*nx), and Gamma likewise.
        Returns x_all (nx, n_elems), Psi_all (n_elems, nx, nx),
        Gamma_all (n_elems, nx, n_theta).
        """
        nx, n_theta = self.model.nx, self.model.n_theta
        x_all = np.asarray(x_stack)
        Psi_all = np.asarray(Psi_stack).reshape(nx, n_elems, nx).transpose(1, 0, 2)
        Gamma_all = np.asarray(Gamma_stack).reshape(nx, n_elems, n_theta) \
            .transpose(1, 0, 2)
        return x_all, Psi_all, Gamma_all

    def _check_converged(self, stage_res):
        """Check stage Newton over all elements (see docs/pitfalls.md)."""
        stage_res = np.asarray(stage_res, float).ravel()
        if stage_res.size == 0:
            return
        # nan -> inf so that both divergence and NaN take the failure branch
        clean = np.nan_to_num(stage_res, nan=np.inf)
        worst = float(np.max(clean))
        limit = self.RES_SAFETY * self.newton_tol
        if worst > limit:
            n_bad = int(np.sum(clean > limit))
            raise RuntimeError(
                f"Collocation: stage Newton did not converge in {n_bad} of "
                f"{stage_res.size} elements (worst scaled residual "
                f"{worst:.2e} > {limit:.1e}; max_iter={self.newton_maxiter}). "
                f"Increase n_sub (the element step h = dt/n_sub gets smaller), "
                f"relax newton_tol or raise newton_maxiter. At a trial point of "
                f"the optimization this is a normal step rejection and the loop "
                f"rolls it back.")

    # ------------------------------------------------------------------
    # Marching over the grid: one shot and a batch of shots
    # ------------------------------------------------------------------
    def _march_compiled(self, c0, theta, t_eval, with_sens):
        nx = self.model.nx
        n_pts = len(t_eval)
        n_elems = (n_pts - 1) * self.n_sub
        theta_cols, u_cols, h_cols = self._march_inputs(theta, t_eval)

        march = self.steps.mapaccum(n_elems, with_sens)

        if not with_sens:
            x_stack, stage_res = march(c0, theta_cols, u_cols, h_cols)
            self._check_converged(stage_res)
            out = np.empty((nx, n_pts))
            out[:, 0] = c0
            # mapaccum returns the state after EVERY element; the points of
            # t_eval are the ends of every n_sub-th one
            out[:, 1:] = np.asarray(x_stack)[:, self.n_sub - 1::self.n_sub]
            return out

        x_stack, Psi_stack, Gamma_stack, stage_res = march(
            c0, theta_cols, u_cols, h_cols)
        self._check_converged(stage_res)
        x_all, Psi_all, Gamma_all = self._unstack_mapaccum(
            x_stack, Psi_stack, Gamma_stack, n_elems)
        return self._accumulate_sens(c0, x_all, Psi_all, Gamma_all, n_pts)

    @staticmethod
    def _shoot_block(stack, j, n_elems, width):
        """Block of shot j in a map output (shots are stacked horizontally).

        One formula for all three stacks: the per-element column width is 1 for
        x, nx for Psi and n_theta for Gamma.
        """
        return stack[:, j * n_elems * width:(j + 1) * n_elems * width]

    def _march_group(self, idxs, c0_list, theta, t_grids):
        """A group of shots of equal grid length in one threaded map call."""
        nx, n_theta = self.model.nx, self.model.n_theta
        n_pts = len(t_grids[idxs[0]])
        n_elems = (n_pts - 1) * self.n_sub

        x0_mat = np.column_stack([c0_list[i] for i in idxs])
        theta_cols, u_cols, h_cols = zip(
            *(self._march_inputs(theta, t_grids[i]) for i in idxs))
        marchmap = self.steps.mapmarch(n_elems, len(idxs))
        x_stack, Psi_stack, Gamma_stack, stage_res = marchmap(
            x0_mat, np.hstack(theta_cols), np.hstack(u_cols),
            np.hstack(h_cols))
        self._check_converged(stage_res)

        x_stack = np.asarray(x_stack)
        Psi_stack = np.asarray(Psi_stack)
        Gamma_stack = np.asarray(Gamma_stack)
        outs = []
        for j, i in enumerate(idxs):
            x_all, Psi_all, Gamma_all = self._unstack_mapaccum(
                self._shoot_block(x_stack, j, n_elems, 1),
                self._shoot_block(Psi_stack, j, n_elems, nx),
                self._shoot_block(Gamma_stack, j, n_elems, n_theta),
                n_elems)
            outs.append(self._accumulate_sens(c0_list[i], x_all, Psi_all,
                                              Gamma_all, n_pts))
        return outs

    def _march_batch_compiled(self, c0_list, theta, t_grids):
        """All shots of a batch: grouped by grid length, then marched."""
        results = [None] * len(t_grids)
        for idxs in group_by_grid_length(t_grids):
            if len(idxs) == 1 or self.n_threads == 1:
                for i in idxs:
                    results[i] = self._march_compiled(c0_list[i], theta,
                                                      t_grids[i],
                                                      with_sens=True)
            else:
                for i, out in zip(idxs, self._march_group(idxs, c0_list,
                                                          theta, t_grids)):
                    results[i] = out
        return results

    def _march(self, c0, theta, t_eval, with_sens):
        """Normalize the inputs, handle degenerate grids, run the march."""
        c0 = np.asarray(c0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)
        theta = np.asarray(theta, dtype=float)[:self.model.n_theta]
        if len(t_eval) < 2:                    # no elements at all
            if not with_sens:
                return np.tile(c0.reshape(-1, 1), (1, len(t_eval)))
            row0 = initial_flat_row(c0, self.model.n_theta)
            out = np.zeros((row0.size, len(t_eval)))
            if len(t_eval) == 1:
                out[:, 0] = row0
            return out
        return self._march_compiled(c0, theta, t_eval, with_sens)

    # ------------------------------------------------------------------
    # Public entry points (the VariationalIntegrator contract)
    # ------------------------------------------------------------------
    def get_jacobian_solution(self, c0, theta, t_eval):
        """State and sensitivities in the flat layout (see sensitivity.py)."""
        return self._march(c0, theta, t_eval, with_sens=True)

    def get_solution(self, c0, theta, t_eval):
        """State only, (nx, N)."""
        return self._march(c0, theta, t_eval, with_sens=False)

    # The jax-named entry points route to the same implementation: use_jax is
    # meaningless for this integrator
    def get_jacobian_solution_jax_batch(self, c0_list, theta, t_grids):
        # Inputs are normalized once here; the internal march methods assume
        # float arrays and a theta of length n_theta
        theta = np.asarray(theta, dtype=float)[:self.model.n_theta]
        c0_list = [np.asarray(c0, dtype=float) for c0 in c0_list]
        t_grids = [np.asarray(t, dtype=float) for t in t_grids]
        return self._march_batch_compiled(c0_list, theta, t_grids)

    def get_solution_jax(self, c0, theta, t_eval):
        return self._march(c0, theta, t_eval, with_sens=False)
