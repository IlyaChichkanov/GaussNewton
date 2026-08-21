"""Symbolic ODE model, its compiled form, and the variational-equation
integrator. See docs/architecture.md and docs/api-reference.md.
"""
import numpy as np
import jax
from jax.experimental.ode import odeint
from jax import numpy as jnp
from scipy.integrate import solve_ivp
from casadi import SX, vertcat, Function, jacobian
import casadi as ca
from abc import abstractmethod
from typing import NamedTuple
from jaxadi import convert

from commom_utils.sensitivity import (group_by_grid_length, initial_flat_row,
                                      split_row)


class Dims(NamedTuple):
    """Problem sizes; unpacks as `nx, n_theta, n_obs = system.dims()`."""
    nx: int
    n_theta: int
    n_obs: int


class ODESystem:
    """Symbolic model written by the user: derivative, observation, inputs."""

    def __init__(self, nx, n_theta, nu):
        self.nx = nx
        self.n_theta = n_theta
        self.nu = nu
        self.state = ca.SX.sym("x", nx)
        self.theta = ca.SX.sym("theta", n_theta)
        self.u = ca.SX.sym("u", nu)
        self.n_obs = self.observation(self.state, self.theta, self.u).shape[0]

    @abstractmethod
    def get_derivative(self, state: SX, theta: SX, u: SX) -> SX:
        pass

    def get_system(self):
        f = self.get_derivative(self.state, self.theta, self.u)
        return self.state, self.theta, self.u, f

    def observation(self, state: SX, theta: SX, u: SX):
        return state

    def get_input_signals(self, t):
        """Input signals at time t; called inside the ODE right-hand side,
        so it must stay traceable: jnp only, no math.*, no `if t < ...`."""
        return []


class CompiledModel:
    """Compiled model: f, h and their Jacobians, numpy and jax back ends.

    The model only — integrators hold it by composition. See
    docs/api-reference.md.
    """

    def __init__(self, model: ODESystem):
        self.model = model

        state_var, theta_var, inp_signal_var, f = self.model.get_system()
        h_observ = self.model.observation(state_var, theta_var, inp_signal_var)

        state_list = state_var.elements()
        inp_list = inp_signal_var.elements()
        theta_list = theta_var.elements()

        self.nu = len(inp_list)
        self.nx = len(state_list)
        self.n_theta = len(theta_list)
        self.n_obs = len(h_observ.elements())

        # Every compiled function takes the same argument list
        args = [*state_list, *inp_list, *theta_list]
        state_vec, theta_vec = vertcat(*state_list), vertcat(*theta_list)

        def _fn(name, expr):
            return Function(name, args, [expr])

        # Private handles are named after their public wrappers:
        # _f_ca <-> f, _dh_dx_ca <-> dh_dx, ...
        self._f_ca = _fn('f', f)
        self._f_jax_ca = convert(self._f_ca, compile=True)
        self._h_ca = _fn('h', h_observ)
        self._dh_dx_ca = _fn('dh_dx', jacobian(h_observ, state_vec))
        self._dh_dtheta_ca = _fn('dh_dtheta', jacobian(h_observ, theta_vec))
        self._df_dtheta_ca = _fn('df_dtheta', jacobian(f, theta_vec))
        self._df_dtheta_jax_ca = convert(self._df_dtheta_ca, compile=True)
        self._df_dx_ca = _fn('df_dx', jacobian(f, state_vec))
        self._df_dx_jax_ca = convert(self._df_dx_ca, compile=True)

        self.identity_observation = self._is_identity_observation(h_observ,
                                                                  state_vec)

        self._obs_map_cache = {}     # (name, n_points) -> mapped CasADi function
        self._jax_vmap_full = None

    @staticmethod
    def _is_identity_observation(h_observ, state_vec):
        """h(x) = x? Then dh/dx = I and dh/dtheta = 0 need not be evaluated.

        20 is the structural comparison depth of ca.is_equal, not a tolerance;
        is_equal can throw on incomparable shapes, which counts as "not
        identity".
        """
        try:
            return (h_observ.shape == state_vec.shape
                    and bool(ca.is_equal(h_observ, state_vec, 20)))
        except Exception:
            return False

    def _get_inp_signals(self, t):
        """Input signals at t; a failure here is raised, never substituted.

        Swallowing it would turn a typo in get_input_signals into silently
        wrong sensitivities instead of a crash.
        """
        try:
            return self.model.get_input_signals(t)
        except Exception as exc:
            raise RuntimeError(
                f"{type(self.model).__name__}.get_input_signals(t) failed at "
                f"t={t}. It is called INSIDE the ODE right-hand side, "
                f"including with a traced time under jax odeint and with an "
                f"array of times in the collocation path: use jnp only, no "
                f"math.* and no Python `if t < ...` (use jnp.where).") from exc

    def dims(self):
        """(nx, n_theta, n_obs)."""
        return Dims(self.nx, self.n_theta, self.n_obs)

    def _inp_on_times(self, t_flat):
        """Input signals on a vector of times -> (len(t_flat), nu).

        Fast path: one call with the whole array (interpolators usually accept
        it); otherwise fall back to a loop over the points.
        """
        t_flat = np.asarray(t_flat)
        n = t_flat.size
        if self.nu == 0:
            return np.zeros((n, 0))
        try:
            sigs = self.model.get_input_signals(t_flat)
            return np.array([np.asarray(s, dtype=float).reshape(n)
                             for s in sigs]).T
        except Exception:
            return np.array(
                [np.asarray(self._get_inp_signals(t), dtype=float)
                 .reshape(self.nu) for t in t_flat]).reshape(n, self.nu)

    # ----------------------------------------------------------------------
    # Pointwise evaluation (numpy)
    # ----------------------------------------------------------------------
    def h(self, state, t, theta):
        """Observation at time t."""
        inp = self._get_inp_signals(t)
        return np.array(self._h_ca(*state, *inp, *theta)).flatten()

    def inverse_h(self, y, t, theta, x_guess=None, n_iter=1):
        """Solve h(x, theta) = y for x with n_iter Gauss-Newton steps."""
        if x_guess is None:
            x_guess = np.zeros(self.nx)
        x = x_guess.copy()
        for _ in range(n_iter):
            dh_dx = self.dh_dx(x, t, theta)
            h_val = self.h(x, t, theta)
            residual = y - h_val
            if np.any(np.isnan(dh_dx)) or np.any(np.isnan(residual)):
                raise ValueError("NaN encountered in inverse_h iteration")
            try:
                delta_x = np.linalg.lstsq(dh_dx, residual, rcond=None)[0]
            except np.linalg.LinAlgError:
                delta_x = np.linalg.pinv(dh_dx, rcond=1e-6) @ residual
            x = x + delta_x
        return x

    def f(self, state, t, theta):
        """Right-hand side at time t."""
        inp = self._get_inp_signals(t)
        return np.array(self._f_ca(*state, *inp, *theta)).flatten()

    def dh_dx(self, state, t, theta):
        """Observation Jacobian with respect to the state."""
        inp = self._get_inp_signals(t)
        return np.array(self._dh_dx_ca(*state, *inp, *theta))

    def dh_dtheta(self, state, t, theta):
        """Observation Jacobian with respect to the parameters."""
        inp = self._get_inp_signals(t)
        return np.array(self._dh_dtheta_ca(*state, *inp, *theta)).squeeze()

    def df_dtheta(self, state, t, theta):
        """Right-hand side Jacobian with respect to the parameters."""
        inp = self._get_inp_signals(t)
        return np.array(self._df_dtheta_ca(*state, *inp, *theta))

    def df_dx(self, state, t, theta):
        """Right-hand side Jacobian with respect to the state."""
        inp = self._get_inp_signals(t)
        return np.array(self._df_dx_ca(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # Batched observations (CasADi Function.map)
    # ----------------------------------------------------------------------
    def _obs_mapped(self, name, n_points):
        """Cached map-version of an observation function over n_points."""
        key = (name, n_points)
        if key not in self._obs_map_cache:
            base = {'h': self._h_ca,
                    'dh_dx': self._dh_dx_ca,
                    'dh_dtheta': self._dh_dtheta_ca}[name]
            self._obs_map_cache[key] = base.map(n_points)
        return self._obs_map_cache[key]

    def observation_batch(self, states, t_array, theta):
        """h, dh/dx and dh/dtheta for a whole grid: 3 CasADi calls, not 3N.

        states (nx, N), t_array (N,), theta (n_theta,) ->
        h (N, n_obs), dh_dx (N, n_obs, nx), dh_dtheta (N, n_obs, n_theta).
        """
        n_points = states.shape[1]
        inp = self._inp_on_times(t_array)

        # Each scalar input of a mapped function is a row (1, N); theta broadcasts
        args = [states[i, :].reshape(1, n_points) for i in range(self.nx)]
        args += [inp[:, j].reshape(1, n_points) for j in range(self.nu)]
        args += [float(theta[k]) for k in range(self.n_theta)]

        def unstack(mat, width):
            # map concatenates matrix outputs along columns:
            # (n_obs, width*N) -> (N, n_obs, width)
            return np.array(mat).reshape(self.n_obs, n_points,
                                         width).transpose(1, 0, 2)

        h = np.array(self._obs_mapped('h', n_points)(*args))  # (n_obs, N)
        dh_dx = unstack(self._obs_mapped('dh_dx', n_points)(*args), self.nx)
        dh_dtheta = unstack(self._obs_mapped('dh_dtheta', n_points)(*args),
                            self.n_theta)
        return h.T, dh_dx, dh_dtheta

    # ----------------------------------------------------------------------
    # JAX facade (jaxadi-compiled functions)
    # ----------------------------------------------------------------------
    def f_jax(self, y, t, *theta):
        """Right-hand side, jax back end."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._f_jax_ca(*y, *inp, *theta)[0].flatten())

    def df_dtheta_jax(self, state, t, theta):
        """df/dtheta, jax back end, (nx, n_theta)."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._df_dtheta_jax_ca(*state, *inp, *theta))[0]

    def df_dx_jax(self, state, t, theta):
        """df/dx, jax back end, (nx, nx).

        jaxadi returns a list of outputs: without the [0] the shape would be
        (1, nx, nx) and a matrix product would silently gain an axis.
        """
        inp = self._get_inp_signals(t)
        return jnp.array(self._df_dx_jax_ca(*state, *inp, *theta))[0]


class VariationalIntegrator:
    """Sensitivities by integrating the variational equations.

        S_theta' = f_x S_theta + f_theta,   S_theta(t_0) = 0
        S_c'     = f_x S_c,                 S_c(t_0)     = I

    Holds a CompiledModel by composition. Both back ends are explicit
    (solve_ivp RK45 / jax dopri), so this path is not for stiff systems —
    use CollocationIntegrator instead. See docs/math.md.
    """

    def __init__(self, model, method: str = 'RK45'):
        # Accepts a raw ODESystem too, so notebooks and tests can be terse
        self.model = (model if isinstance(model, CompiledModel)
                      else CompiledModel(model))
        self.method = method
        self.ATOL = 1e-5
        self.RTOL = 1e-5
        self._jax_vmap_full = None

    def get_solution(self, c0, theta, t_eval):
        """State only, (nx, m)."""
        m = self.model

        def system(t, y):
            return m.f(y, t, theta[:m.n_theta])

        sol = solve_ivp(system, (t_eval[0], t_eval[-1]), c0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol.y

    def get_jacobian_solution(self, c0, theta, t_eval):
        """State and sensitivities in the flat layout (see sensitivity.py)."""
        p = self.model.n_theta
        y0 = initial_flat_row(c0, p)

        def full_ode(t, y):
            return self._variational_rhs(y, t, theta[:p])

        sol = solve_ivp(full_ode, (t_eval[0], t_eval[-1]), y0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Sensitivity integration failed: {sol.message}")
        return sol.y

    def get_solution_jax(self, c0, theta, t_eval):
        """State only, jax back end."""
        sol = odeint(self.model.f_jax,
                     jnp.array(c0),
                     jnp.array(t_eval),
                     *theta[:self.model.n_theta],
                     rtol=self.RTOL, atol=self.ATOL)
        return np.array(sol).T

    def _vmapped_full_integrator(self):
        """Cached jit+vmap wrapper; jax recompiles on a shape change."""
        if self._jax_vmap_full is None:
            n, p = self.model.nx, self.model.n_theta
            # Tail of the initial row (S_theta = 0, S_c = I) is a constant
            sens0 = jnp.array(initial_flat_row(np.zeros(n), p)[n:])

            def integrate_one(c0, t_grid, theta):
                y0 = jnp.concatenate([c0, sens0])
                return odeint(self._variational_rhs_jax, y0, t_grid, *theta,
                              rtol=self.RTOL, atol=self.ATOL)

            self._jax_vmap_full = jax.jit(
                jax.vmap(integrate_one, in_axes=(0, 0, None)))
        return self._jax_vmap_full

    def get_jacobian_solution_jax_batch(self, c0_list, theta, t_grids):
        """All shots at once: one vmap call per group of equal grid length.

        c0_list (S, nx), t_grids: list of time arrays. Returns S flat matrices,
        one per shot, as get_jacobian_solution would.
        """
        theta_j = jnp.array(np.asarray(theta[:self.model.n_theta], dtype=float))

        results = [None] * len(t_grids)
        integrate = self._vmapped_full_integrator()
        for idxs in group_by_grid_length(t_grids):
            ts_stack = jnp.array(np.stack([np.asarray(t_grids[i]) for i in idxs]))
            c0_stack = jnp.array(np.stack([np.asarray(c0_list[i]) for i in idxs]))
            sols = np.array(integrate(c0_stack, ts_stack, theta_j))  # (k, L, dim)
            for j, i in enumerate(idxs):
                results[i] = sols[j].T
        return results

    # ----------------------------------------------------------------------
    # Variational equations: the two right-hand sides, same scheme
    # ----------------------------------------------------------------------
    def _variational_rhs(self, y, t, theta):
        """Extended right-hand side (numpy).

        The compiled functions are called directly with a shared argument
        tuple, so the user's get_input_signals runs once per call rather than
        once per wrapper.
        """
        m = self.model
        n, p = m.nx, m.n_theta
        x, S_theta, S_c = split_row(y, n, p)

        args = (*x, *m._get_inp_signals(t), *theta)
        dx = np.array(m._f_ca(*args)).ravel()
        f_x = np.array(m._df_dx_ca(*args))
        f_theta = np.array(m._df_dtheta_ca(*args))

        dS = f_x @ np.concatenate([S_theta, S_c], axis=1)
        # The flat layout is all of S_theta then all of S_c (each in C order),
        # so the product has to be split again rather than ravelled whole
        return np.concatenate([dx, (dS[:, :p] + f_theta).ravel(),
                               dS[:, p:].ravel()])

    def _variational_rhs_jax(self, y, t, *theta):
        """Extended right-hand side (jax) — the same scheme as above."""
        m = self.model
        n, p = m.nx, m.n_theta
        x, S_theta, S_c = split_row(y, n, p)

        args = (*x, *m._get_inp_signals(t), *theta)
        dx = jnp.array(m._f_jax_ca(*args)[0].flatten())
        f_x = jnp.array(m._df_dx_jax_ca(*args))[0]
        f_theta = jnp.array(m._df_dtheta_jax_ca(*args))[0]

        dS = f_x @ jnp.concatenate([S_theta, S_c], axis=1)
        return jnp.concatenate([dx, (dS[:, :p] + f_theta).flatten(),
                                dS[:, p:].flatten()])


class SystemIntegrator(CompiledModel):
    """Integration with the input u held by the caller (MPC simulation).

    Everywhere else the inputs come from the model via get_input_signals(t);
    here u is an argument and stays constant over the step.
    """

    def __init__(self, model: ODESystem, method: str = 'RK45'):
        super().__init__(model)
        self.method = method
        # df/du is the only function the parent does not already compile
        state_var, theta_var, inp_var, f = model.get_system()
        self._f_u_ca = Function(
            'J_u', [*state_var.elements(), *inp_var.elements(),
                    *theta_var.elements()],
            [jacobian(f, vertcat(*inp_var.elements()))])

    def f_of_u(self, state, u, theta):
        """Right-hand side for an explicitly given input u."""
        return np.array(self._f_ca(*state, *u, *theta)).ravel()

    def f_of_u_jax(self, state, t, u, theta):
        # t is unused (u is held) but required: odeint calls f(y, t, *args)
        return jnp.array(self._f_jax_ca(*state, *u, *theta)[0]).flatten()

    def _check(self, c0, u, theta):
        if not (len(c0) == self.nx and len(u) == self.nu
                and len(theta) == self.n_theta):
            raise ValueError(
                f"expected x({self.nx}), u({self.nu}), theta({self.n_theta}); "
                f"got x({len(c0)}), u({len(u)}), theta({len(theta)})")

    def integrate(self, c0, u, theta, t_span):
        """Trajectory over t_span at constant u."""
        self._check(c0, u, theta)
        sol = solve_ivp(lambda t, y: self.f_of_u(y, u, theta), t_span, c0,
                        method=self.method)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return sol.y.T

    def step(self, c0, u, theta, dt):
        """One step of length dt at constant u."""
        return self.integrate(c0, u, theta, (0.0, dt))[-1]

    def step_jax(self, c0, u, theta, dt):
        self._check(c0, u, theta)
        # Tolerances deliberately left at the odeint defaults (1.4e-8): MPC
        # simulation has always been computed with them. Relaxing them here
        # would be a decision of its own, not a refactoring side effect.
        sol = odeint(self.f_of_u_jax, jnp.array(c0), jnp.array([0.0, dt]),
                     u, theta)
        return np.array(sol[-1])

    def get_lin_system_dynamics(self, state, u, theta):
        """Linearization (A, B, D) = (df/dx, df/du, df/dtheta) at a point."""
        self._check(state, u, theta)
        args = (*state, *u, *theta)
        return (np.array(self._df_dx_ca(*args)),
                np.array(self._f_u_ca(*args)),
                np.array(self._df_dtheta_ca(*args)))


class SyntheticDataGenerator:
    """Synthetic measurements for a system.

    Noise of standard deviation `sigma` is added to the STATES before h(x) is
    applied. With `perturb_initial`, each batch starts from
    c0 * (1 + perturbation_scale * (rand - 0.5)).
    """

    def __init__(self, system_ode: ODESystem, sigma=0.01, perturb_initial=False,
                 perturbation_scale=0.1, use_jax=True):
        self.system = CompiledModel(system_ode)
        self.integrator = VariationalIntegrator(self.system)
        self.sigma = sigma
        self.perturb_initial = perturb_initial
        self.perturbation_scale = perturbation_scale
        self.use_jax = use_jax

        self.state_len, self.theta_len, self.meas_len = self.system.dims()

    def generate_batch(self, c0, theta, t_start, t_end, n_measurements, seed=None):
        """One batch on [t_start, t_end].

        Returns (t_eval (n,), measurements (n, meas_len),
        noisy_states (n, state_len), inputs (n, nu)).
        """
        if seed is not None:
            np.random.seed(seed)

        if self.perturb_initial:
            c0_true = c0 * (1 + self.perturbation_scale
                            * (np.random.random(self.state_len) - 0.5))
        else:
            c0_true = c0

        t_eval = np.linspace(t_start, t_end, n_measurements)

        # Both integrator entry points return (state_len, n_measurements)
        integrate = (self.integrator.get_solution_jax if self.use_jax
                     else self.integrator.get_solution)
        solution = integrate(c0_true, theta, t_eval)

        noise = self.sigma * np.random.normal(size=(self.state_len, n_measurements))
        noisy_states = (solution + noise).T  # (n_measurements, state_len)

        measurements = np.zeros((n_measurements, self.meas_len))
        inp_signal = np.zeros((n_measurements, self.system.nu))
        for i, state in enumerate(noisy_states):
            measurements[i] = self.system.h(state, t_eval[i], theta)
            inp_signal[i] = self.system.model.get_input_signals(t_eval[i])
        return t_eval, measurements, noisy_states, inp_signal

    def generate(self, c0, theta, time_intervals, n_measurements, seeds=None):
        """One batch per time interval.

        Returns four lists — times, measurements, states, inputs — with one
        entry per interval.
        """
        if seeds is not None and len(seeds) != len(time_intervals):
            raise ValueError("seeds must have one entry per time interval")

        t_batches = []
        measured_batches = []
        state_batches = []
        inp_signals = []
        for idx, (t_start, t_end) in enumerate(time_intervals):
            seed = seeds[idx] if seeds is not None else None
            t_eval, meas, states, inp_signal = self.generate_batch(
                c0, theta, t_start, t_end, n_measurements, seed=seed
            )
            t_batches.append(t_eval)
            measured_batches.append(meas)
            state_batches.append(states)
            inp_signals.append(inp_signal)

        return t_batches, measured_batches, state_batches, inp_signals


class MHESyntheticDataGenerator:
    """Synthetic sliding-window data for MHE; noise is added to the OUTPUT."""

    def __init__(self, system_ode: ODESystem, sigma=1e-3):
        self.system = CompiledModel(system_ode)
        self.integrator = VariationalIntegrator(self.system)
        self.sigma = sigma
        self.state_dim, self.param_dim, self.meas_dim = self.system.dims()
        self.control_dim = system_ode.nu

    def _generate_trajectory(self, c0, theta, t, sigma=None):
        """Trajectory on t -> (t, u, full_states, measured_states)."""
        # sigma is a scalar (one per channel) or a vector (meas_dim,)
        sigma = np.atleast_1d(np.asarray(
            0.0 if sigma is None else sigma, dtype=float))
        if sigma.size == 1:
            sigma = np.full(self.meas_dim, sigma[0])

        u = np.zeros((len(t), self.control_dim))
        for i, ti in enumerate(t):
            u[i] = self.system.model.get_input_signals(ti)

        full_states = self.integrator.get_solution(c0, theta, t).T

        measured = np.zeros((len(t), self.meas_dim))
        for i, state in enumerate(full_states):
            measured[i] = self.system.h(state, t[i], theta)

        mean = np.zeros(self.meas_dim)
        cov = np.diag(sigma**2)
        noise = np.random.multivariate_normal(mean, cov, len(t))
        measured += noise
        return t, u, full_states, measured

    def generate_sliding_windows_exact(self, c0, theta, t0, tf, num_windows,
                                       n_measurement, overlap_points=1):
        """Overlapping windows of tf seconds each, n_measurement points wide.

        Consecutive windows share `overlap_points` points. Returns four lists
        (times, inputs, measurements, full states), one entry per window.
        """
        assert len(c0) == self.state_dim, "c0 length must match the state size"
        assert len(theta) == self.param_dim, "theta length must match the model"

        dt = tf / (n_measurement - 1)          # step inside a window
        step = n_measurement - overlap_points  # step between window starts

        total_points = 1 + (num_windows - 1) * step + (n_measurement - 1)
        t_long = np.linspace(t0, t0 + (num_windows - 1) * step * dt + tf,
                             total_points)

        t_long, u_long, full_long, meas_long = self._generate_trajectory(
            c0, theta, t_long, self.sigma
        )

        t_windows = []
        u_windows = []
        meas_windows = []
        full_windows = []

        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + n_measurement
            t_windows.append(t_long[start_idx:end_idx])
            u_windows.append(u_long[start_idx:end_idx])
            meas_windows.append(meas_long[start_idx:end_idx])
            full_windows.append(full_long[start_idx:end_idx])

        return t_windows, u_windows, meas_windows, full_windows


def check_system_ok(system_ode: ODESystem):
    """Compile a model and check that it declares as many inputs as it returns."""
    system = CompiledModel(system_ode)
    n_inp = len(system_ode.get_input_signals(0))
    if system.nu != n_inp:
        raise ValueError(
            f"{type(system_ode).__name__}: declares nu={system.nu}, but "
            f"get_input_signals(t) returns {n_inp} signals")
    return True
