"""Multiple-shooting problem assembly: shots, unknowns layout, residual rows.

See docs/architecture.md for how this layer relates to the model, the
integrator and the normal equations.
"""
import time
from dataclasses import dataclass

import numpy as np
from commom_utils.ode_system import (CompiledModel, ODESystem,
                                     VariationalIntegrator)
from commom_utils.sensitivity import SensitivityTrajectory
from scipy.sparse import (bmat, block_diag, csr_matrix, diags, vstack, hstack)


@dataclass
class ShootRows:
    """Blocks of one shot, feeding both the sparse J and the accumulated H/g.

    Rows of the RESIDUAL Jacobian:
        J_theta : (m, n_obs, n_theta)  dr/dtheta = W (h_x S_theta + h_theta)
        J_c     : (m, n_obs, n_state)  dr/dc_j   = W  h_x S_c
        r       : (m, n_obs)           weighted residuals W (y_i - h_i)

    State SENSITIVITIES at the end of the shot, from which the continuity
    rows are built (not Jacobian rows):
        S_theta_end : (n_state, n_theta)   dx(tau_{j+1})/dtheta
        S_c_end     : (n_state, n_state)   dx(tau_{j+1})/dc_j

    x_end is the final state of the shot, c0 its initial state.
    """
    J_theta: np.ndarray
    J_c: np.ndarray
    r: np.ndarray
    S_theta_end: np.ndarray
    S_c_end: np.ndarray
    x_end: np.ndarray
    c0: np.ndarray


class UnknownsLayout:
    """Where things live in theta_full = [theta; c_1 .. c_T]."""

    def __init__(self, n_theta, n_state):
        self.n_theta = n_theta
        self.n_state = n_state
        self._c_offsets = []      # start of each batch's c-block
        self._n_shoots = []
        self.n_unknowns = n_theta

    def add_batch(self, n_shoot):
        """Register a batch by its ACTUAL shot count (may differ from N_shoot)."""
        self._c_offsets.append(self.n_unknowns)
        self._n_shoots.append(n_shoot)
        self.n_unknowns += n_shoot * self.n_state

    @property
    def theta(self):
        return slice(0, self.n_theta)

    def c(self, batch, shoot):
        """Slice of the initial state of shot `shoot` in batch `batch`."""
        start = self._c_offsets[batch] + shoot * self.n_state
        return slice(start, start + self.n_state)

    def n_shoots(self, batch):
        return self._n_shoots[batch]


class TimeIntervalManager:
    """Splits a measurement grid into shots.

    The actual shot count (`self.N_shoot`) can differ from the requested one:
    nodes are placed at a constant stride `len(t) // N_shoot`, so an inexact
    division makes the last interval longer. The last point of a shot grid is
    a junction point only and does not enter the measurement residual.
    """

    def __init__(self, N_shoot, t_eval_measurements):
        self.t_eval_measurements = t_eval_measurements
        N_measurement = len(t_eval_measurements)
        if N_measurement < 2:
            raise ValueError(
                f"at least 2 measurement points are needed, got {N_measurement}")
        if int(N_shoot) < 1:
            raise ValueError(f"N_shoot must be >= 1, got {N_shoot}")

        self.measurement_indexes = np.arange(N_measurement)
        step = N_measurement // int(N_shoot)
        if step < 1:
            # step == 0 used to surface as "slice step cannot be zero" below
            raise ValueError(
                f"N_shoot={N_shoot} exceeds the number of measurements "
                f"({N_measurement}); the maximum for this grid is {N_measurement}")

        shoot_indexes = self.measurement_indexes[0:-1:step]
        self.shoot_indexes = np.append(shoot_indexes, self.measurement_indexes[-1])
        self.N_shoot = len(self.shoot_indexes) - 1

        # A shot's grid is its own measurements plus the next point (the junction)
        self._intervals = []
        for j in range(self.N_shoot):
            meas_idx = np.arange(self.shoot_indexes[j], self.shoot_indexes[j + 1])
            t_interval = np.append(self.t_eval_measurements[meas_idx],
                                   self.t_eval_measurements[meas_idx[-1] + 1])
            self._intervals.append((t_interval, meas_idx))

    def get_time_interval(self, shoot):
        """(shot grid including the junction point, indices of its measurements)."""
        return self._intervals[shoot]


class MultipleShooting:
    """Gauss-Newton problem over [theta; shot initial states].

    Arguments are documented in docs/api-reference.md.
    """

    def __init__(self, system: ODESystem, N_shoot: int, gamma: np.ndarray = None,
                 c0_cost: float = 1, use_jax: bool = False, verbose: bool = False,
                 cont_scale=None):
        # The model and the integrator are kept apart: CollocationShooting
        # replaces only self.integrator, the model stays shared
        self.system = CompiledModel(system)
        self.integrator = VariationalIntegrator(self.system)
        self.N_shoot = N_shoot
        self.gamma = gamma
        self.c0_cost = c0_cost
        self.use_jax = use_jax
        self.verbose = verbose
        self.cont_scale = cont_scale
        self._cont_w = None            # cached 1/scale per state

        self.state_measured_batches = []
        self.t_eval_measurements_batches = []
        self.interval_managers = []
        self.layout = UnknownsLayout(self.system.n_theta, self.system.nx)

    def add_batch(self, state_measured, t_eval_measurements):
        tm = TimeIntervalManager(self.N_shoot, t_eval_measurements)
        self.state_measured_batches.append(state_measured)
        self.t_eval_measurements_batches.append(t_eval_measurements)
        self.interval_managers.append(tm)
        self.layout.add_batch(tm.N_shoot)
        self._cont_w = None            # the scale is computed over all batches

    def _cont_weights(self):
        """Continuity row weights 1/scale, computed once and then fixed.

        cont_scale: None (no scaling), an (nx,) array of state scales, or
        'auto' (RMS of the measurements, requires h(x) = x). The weights must
        not change between iterations - see docs/math.md.
        """
        if self._cont_w is not None:
            return self._cont_w

        nx = self.system.nx
        req = self.cont_scale
        if req is None:
            scale = np.ones(nx)
        elif isinstance(req, str):
            if req != 'auto':
                raise ValueError(
                    f"cont_scale: expected None, 'auto' or an array ({nx},), "
                    f"got {req!r}")
            if not self.system.identity_observation:
                raise ValueError(
                    "cont_scale='auto' only works for the identity observation "
                    "h(x) = x; otherwise the measurements do not give state "
                    "scales. Pass an array of scales explicitly.")
            if not self.state_measured_batches:
                raise ValueError("cont_scale='auto': add_batch must come first")
            meas = np.vstack([np.asarray(m, float)
                              for m in self.state_measured_batches])
            scale = np.sqrt((meas ** 2).mean(axis=0))
            scale = np.where(scale > 0, scale, 1.0)      # constant zero -> 1
        else:
            scale = np.asarray(req, dtype=float)
            if scale.shape != (nx,):
                raise ValueError(
                    f"cont_scale: expected an array ({nx},), got {scale.shape}")
            if np.any(scale <= 0):
                raise ValueError("cont_scale: scales must be > 0")

        self._cont_w = 1.0 / scale
        return self._cont_w

    def make_full_theta(self, theta0, c0_guess=None, c0_init_method='inverse_h',
                        n_iter=1):
        """theta_full = [theta0; c_1..c_T], c_j from the shot's first measurement.

        c0_init_method: 'inverse_h' (invert the observation by Newton),
        'measurement_pad' (copy the measurement, for h(x) = x) or 'zeros'.
        """
        n_state = self.system.nx

        def c0_zeros(y, t):
            return np.zeros(n_state)

        def c0_measurement_pad(y, t):
            c0_ = np.zeros(n_state)
            c0_[:n_state] = y[:n_state]
            return c0_

        def c0_inverse_h(y, t):
            x_guess = (np.zeros(n_state)
                       if c0_guess is None or np.any(np.isnan(c0_guess))
                       else c0_guess)
            return self.system.inverse_h(y, t, theta0,
                                         x_guess=x_guess, n_iter=n_iter)

        methods = {'zeros': c0_zeros, 'measurement_pad': c0_measurement_pad,
                   'inverse_h': c0_inverse_h}
        if c0_init_method not in methods:
            raise ValueError(f"Unknown c0_init_method: {c0_init_method}")
        init_c0 = methods[c0_init_method]

        parts = [np.copy(theta0)]
        for state_measured, t_meas, tm in zip(self.state_measured_batches,
                                              self.t_eval_measurements_batches,
                                              self.interval_managers):
            for shoot in range(tm.N_shoot):
                idx = tm.shoot_indexes[shoot]
                parts.append(init_c0(state_measured[idx], t_meas[idx]))
        return np.concatenate(parts)

    def _concatenate_jacobian_batches(self, jacobians):
        """Stack per-batch Jacobians: shared theta columns, block-diagonal c."""
        if not jacobians:
            raise ValueError("At least one batch Jacobian is required")

        _, n_theta, _ = self.system.dims()
        theta_blocks = [jacobian[:, :n_theta] for jacobian in jacobians]
        c0_blocks = [jacobian[:, n_theta:] for jacobian in jacobians]

        theta_block = vstack(theta_blocks, format='csr')
        c0_block = block_diag(c0_blocks, format='csr')
        return hstack([theta_block, c0_block], format='csr')

    def solve(self, theta_full):
        """Reference assembly with an explicit J -> (J, R, J_G, R_G)."""
        solve_start = time.perf_counter()

        J_batches = []
        J_G_batches = []
        R_batches = []
        R_G_batches = []

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches, self.t_eval_measurements_batches)):
            batch_start = time.perf_counter()
            J_batch, J_G_batch, R_batch, R_G_batch = self._solve_batch(
                theta_full, state_measured, t_meas, batch
            )
            if self.verbose:
                print(f'  Batch {batch}: {time.perf_counter() - batch_start:.3f}s')

            J_batches.append(J_batch)
            J_G_batches.append(J_G_batch)
            R_batches.append(R_batch)
            R_G_batches.append(R_G_batch)

        J_total = self._concatenate_jacobian_batches(J_batches)
        J_G_total = self._concatenate_jacobian_batches(J_G_batches)
        R_total = np.concatenate(R_batches)
        R_G_total = np.concatenate(R_G_batches)

        if self.verbose:
            print(f'Solve total: {time.perf_counter() - solve_start:.3f}s | '
                  f'J: {J_total.shape}, J_G: {J_G_total.shape}')
        return J_total, R_total, J_G_total, R_G_total

    def shoot_rows(self, theta_full, state_measured, t_meas, batch_idx):
        """Integrate the shots of a batch and weight them -> list of ShootRows.

        The shared core of both assembly paths: _solve_batch builds a sparse J
        out of these blocks, while gauss_newton/normal_equations.py folds them
        straight into H and g.
        """
        n_state, n_theta, n_obs = self.system.dims()
        if self.gamma is not None and len(self.gamma) != n_obs:
            raise ValueError(f"gamma length must be {n_obs}, got {len(self.gamma)}")

        tm = self.interval_managers[batch_idx]
        theta = theta_full[self.layout.theta]
        intervals = [tm.get_time_interval(sh) for sh in range(tm.N_shoot)]
        c0_list = [theta_full[self.layout.c(batch_idx, sh)]
                   for sh in range(tm.N_shoot)]

        if self.use_jax:
            # All shots in one batched call (vmap or threads inside)
            flats = self.integrator.get_jacobian_solution_jax_batch(
                c0_list, theta, [ti for ti, _ in intervals])
        else:
            flats = [self.integrator.get_jacobian_solution(c0, theta, ti)
                     for c0, (ti, _) in zip(c0_list, intervals)]

        # gamma is sqrt(W): the residual is multiplied by it and the cost
        # squares it, so gamma = 1 means sigma = 1, not "weight 1"
        gamma = self.gamma if self.gamma is not None else np.ones(n_obs)

        rows = []
        for c0, flat, (t_interval, meas_idx) in zip(c0_list, flats, intervals):
            traj = SensitivityTrajectory.unpack(flat, n_state, n_theta)
            # Points 0..m-1 are the measurements; the last grid point is the
            # junction and does not enter the measurement residual
            m = len(meas_idx)
            meas = traj.head(m)

            if self.system.identity_observation:
                # h(x) = x: dh/dx = I, dh/dtheta = 0
                h_pred, J_theta_all, J_c_all = meas.x, meas.S_theta, meas.S_c
            else:
                h_pred, dh_dx, dh_dtheta = self.system.observation_batch(
                    meas.x.T, t_interval[:m], theta)
                J_theta_all = np.einsum('mij,mjk->mik', dh_dx, meas.S_theta) + dh_dtheta
                J_c_all = np.einsum('mij,mjk->mik', dh_dx, meas.S_c)

            W = np.tile(gamma, (m, 1))
            W[0] *= self.c0_cost       # extra weight on the first point

            rows.append(ShootRows(
                J_theta=W[:, :, None] * J_theta_all,
                J_c=W[:, :, None] * J_c_all,
                r=W * (state_measured[meas_idx] - h_pred),
                S_theta_end=traj.S_theta[-1],
                S_c_end=traj.S_c[-1],
                x_end=traj.x[-1],
                c0=c0,
            ))
        return rows

    def continuity_rows(self, rows):
        """Continuity rows (J_G, R_G) from the final blocks of the shots.

        G_j = x_j(t_{j+1}; c_j, theta) - c_{j+1}: the theta part is S_theta_end
        of shot j, the c part is block-bidiagonal (S_c_end in column j, -I in
        column j+1). Rows are divided by the state scale (see _cont_weights);
        with cont_scale=None the weights are 1 and the numbers are unchanged.
        """
        n_state, n_theta, _ = self.system.dims()
        n_shoot = len(rows)
        n_cont = (n_shoot - 1) * n_state
        if n_cont == 0:
            return csr_matrix((0, n_theta + n_shoot * n_state)), np.zeros(0)

        w = self._cont_weights()                   # (n_state,), 1/scale
        R_G = np.concatenate([w * -(rows[j].x_end - rows[j + 1].c0)
                              for j in range(n_shoot - 1)])
        minus_eye = -diags(w)
        blocks = []
        for j in range(n_shoot - 1):
            row = [csr_matrix(w[:, None] * rows[j].S_theta_end)] + [None] * n_shoot
            row[1 + j] = csr_matrix(w[:, None] * rows[j].S_c_end)
            row[2 + j] = minus_eye
            blocks.append(row)
        return bmat(blocks, format='csr'), R_G

    def _solve_batch(self, theta_full, state_measured, t_meas, batch_idx):
        _, _, n_obs = self.system.dims()
        rows = self.shoot_rows(theta_full, state_measured, t_meas, batch_idx)

        # Dense Jacobian blocks: the theta part is shared by all rows, the c
        # part is block-diagonal over shots, so J is one hstack
        J = hstack([
            csr_matrix(np.vstack([r.J_theta.reshape(-1, r.J_theta.shape[2])
                                  for r in rows])),
            block_diag([r.J_c.reshape(-1, r.J_c.shape[2]) for r in rows],
                       format='csr'),
        ], format='csr')
        R = np.concatenate([r.r.ravel() for r in rows])

        J_G, R_G = self.continuity_rows(rows)
        return J, J_G, R, R_G
