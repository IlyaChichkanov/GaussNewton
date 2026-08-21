# API reference

Only the public surface; private helpers are described in
[architecture.md](architecture.md) where they matter. Shapes use `nx`
(state size), `n_theta`, `n_obs`, `m` (grid points), `T` (shots).

## `commom_utils/ode_system.py`

### `ODESystem(nx, n_theta, nu)`

Base class for a symbolic model. Override:

| Method | Returns |
|---|---|
| `get_derivative(state, theta, u)` | CasADi expression of `dx/dt`, length `nx` |
| `observation(state, theta, u)` | what is measured; default is `state` |
| `get_input_signals(t)` | list of `nu` input values at time `t` |

`n_obs` is inferred from `observation`. `get_input_signals` is called *inside*
the right-hand side, including under a jax trace and with an array of times in
the collocation path: use `jnp` only, no `math.*` and no Python `if t < ...`
(use `jnp.where`).

### `CompiledModel(model: ODESystem)`

The compiled model — CasADi and jaxadi functions, no integration.

| Member | Meaning |
|---|---|
| `f(state, t, theta)` | right-hand side, `(nx,)` |
| `h(state, t, theta)` | observation, `(n_obs,)` |
| `df_dx`, `df_dtheta` | `(nx, nx)`, `(nx, n_theta)` |
| `dh_dx`, `dh_dtheta` | `(n_obs, nx)`, `(n_obs, n_theta)` |
| `f_jax`, `df_dx_jax`, `df_dtheta_jax` | the same under jax |
| `dims()` | `Dims(nx, n_theta, n_obs)`, a `NamedTuple` |
| `identity_observation` | `True` when `h(x) = x`, which lets callers skip the observation Jacobians |
| `observation_batch(states, t_array, theta)` | `(h, dh_dx, dh_dtheta)` on a whole grid; `states` is `(nx, m)` |
| `inverse_h(y, t, theta, x_guess=None, n_iter=1)` | Newton inversion of the observation, used to guess shot initial states |

### `VariationalIntegrator(model, method='RK45')`

Sensitivities by integrating the variational equations. `model` may be a
`CompiledModel` or a raw `ODESystem`. Tolerances are attributes: `ATOL`,
`RTOL` (both `1e-5` by default).

| Method | Returns |
|---|---|
| `get_solution(c0, theta, t_eval)` | state only, `(nx, m)` |
| `get_jacobian_solution(c0, theta, t_eval)` | the flat contract below, `(nx + nx·n_theta + nx·nx, m)` |
| `get_solution_jax(c0, theta, t_eval)` | state only, jax back end |
| `get_jacobian_solution_jax_batch(c0_list, theta, t_grids)` | one flat matrix per shot; shots are grouped by grid length and run under `vmap`+`jit` |

**The flat contract.** `get_jacobian_solution` returns rows
`[x; S_theta.flatten(); S_c.flatten()]` in C order, one column per point of
`t_eval`. Unpack it with `SensitivityTrajectory.unpack(flat, nx, n_theta)` —
never by hand.

### `SystemIntegrator(CompiledModel)`

Integration with the input `u` supplied by the caller and held over the step
(MPC simulation), rather than taken from the model.
`integrate(c0, u, theta, t_span)`, `step(c0, u, theta, dt)`, `step_jax(...)`,
and `get_lin_system_dynamics(state, u, theta)` → `(A, B, D)`.

### Data generators

- `SyntheticDataGenerator(system_ode, sigma=0.01, perturb_initial=False,
  perturbation_scale=0.1, use_jax=True)` —
  `generate(c0, theta, time_intervals, n_measurements, seeds=None)` returns
  `(t_batches, meas_batches, ...)`, one entry per interval. Noise is added to
  the **states** before `h(x)` is applied.
- `MHESyntheticDataGenerator(system_ode, sigma=1e-3)` —
  `generate_sliding_windows_exact(...)` for the MHE tests. Noise is added to
  the output.
- `check_system_ok(system_ode)` — asserts that a model declares as many inputs
  as `get_input_signals` returns.

## `commom_utils/sensitivity.py`

### `SensitivityTrajectory`

Dataclass with `x (m, nx)`, `S_theta (m, nx, n_theta)`, `S_c (m, nx, nx)`;
shape consistency is checked on construction.

| Member | Meaning |
|---|---|
| `unpack(flat, nx, n_theta)` | classmethod: read the flat integrator output |
| `pack()` | back to the flat layout — the inverse of `unpack` |
| `head(m)` | first `m` points, i.e. a shot without its junction point |
| `n_points` | number of grid points |

### Module functions

- `initial_flat_row(c0, n_theta)` — the initial extended state
  `[c0; S_theta = 0; S_c = I]` in the flat layout.
- `split_row(y, nx, n_theta)` → `(x, S_theta, S_c)` for one point. Slices and
  reshapes only, so it works under a jax trace.
- `group_by_grid_length(t_grids)` — indices of shots grouped by grid length,
  which both batch integrators require (a `vmap` precondition for jax, a shared
  `mapaccum` function for collocation).

## `commom_utils/collocation.py`

### `RadauTables(K=3)`

Radau IIA nodes for K ∈ {1, 2, 3}: `tau`, `nodes` (with τ₀ = 0), `d0`, `D1`,
and `butcher_a = inv(D1)`. K = 3 gives order 5 and L-stability.

### `CollocationIntegrator(model, K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25, rootfinder_plugin='newton', rootfinder_options=None, n_threads=None)`

A drop-in replacement for `VariationalIntegrator` — same output contract,
same methods (`get_jacobian_solution`, `get_solution`,
`get_jacobian_solution_jax_batch`, `get_solution_jax`; the "jax" batch entry
point is implemented with threads and uses no JAX).

| Argument | Meaning |
|---|---|
| `K` | number of Radau IIA stages, order 2K−1 |
| `n_sub` | elements per interval between neighbouring points of `t_eval` — this is the accuracy knob, and there is no error estimate |
| `newton_tol` | stage Newton tolerance: it is **both** `abstol` (on the residual) and `abstolStep` (on the step); the criteria act as OR |
| `newton_maxiter` | hard iteration limit |
| `rootfinder_plugin`, `rootfinder_options` | switch to `kinsol`/`fast_newton` and pass their own options |
| `n_threads` | threads for `map('thread')` across shots |

Newton failure does not raise from C++ (`error_on_fail=False`): every element
also returns a scaled residual `stage_res`, and the march itself checks
`max(stage_res) <= 10·newton_tol` and raises a one-line `RuntimeError`. During
optimization that is a normal step rejection, which the loop catches.

### `CollocationStepFunctions`

Builds and caches the CasADi functions of one step (stage residual,
rootfinder, the `step_sens`/`step_x` pair, `mapaccum` and `map`). Constructed by
`CollocationIntegrator`; not used directly.

## `gauss_newton/problem.py`

### `MultipleShooting(system, N_shoot, gamma=None, c0_cost=1, use_jax=False, verbose=False, cont_scale=None)`

| Argument | Meaning |
|---|---|
| `system` | an `ODESystem`; it is compiled into `self.system` |
| `N_shoot` | requested number of shots; the actual count can differ (see `TimeIntervalManager`) |
| `gamma` | `(n_obs,)`, **√W** — the residual is multiplied by it |
| `c0_cost` | extra weight on the first point of each shot |
| `use_jax` | route the shots through the batch integrator entry point |
| `cont_scale` | `None`, an `(nx,)` array of state scales, or `'auto'` (RMS of the measurements, requires `h(x) = x`) — see [math.md](math.md) |

| Method | Returns |
|---|---|
| `add_batch(state_measured, t_eval_measurements)` | registers a data batch; batches share θ |
| `make_full_theta(theta0, c0_guess=None, c0_init_method='inverse_h', n_iter=1)` | `theta_full = [θ; c_1..c_T]`, with `c_j` from the first measurement of the shot; methods: `'inverse_h'`, `'measurement_pad'`, `'zeros'` |
| `shoot_rows(theta_full, state_measured, t_meas, batch_idx)` | list of `ShootRows` — the shared core of both assembly paths |
| `continuity_rows(rows)` | `(J_G, R_G)` |
| `solve(theta_full)` | `(J, R, J_G, R_G)` — the reference assembly with an explicit sparse `J` |

### `ShootRows`

The blocks of one shot.

| Field | Shape | Meaning |
|---|---|---|
| `J_theta` | `(m, n_obs, n_theta)` | `dr/dθ = W(h_x S_θ + h_θ)` |
| `J_c` | `(m, n_obs, nx)` | `dr/dc_j = W h_x S_c` |
| `r` | `(m, n_obs)` | weighted residuals `W(y − h)` |
| `S_theta_end` | `(nx, n_theta)` | state sensitivity at the end of the shot |
| `S_c_end` | `(nx, nx)` | state sensitivity at the end of the shot |
| `x_end`, `c0` | `(nx,)` | final and initial state of the shot |

`J_*` are rows of the *residual* Jacobian; `S_*_end` are *state*
sensitivities, from which the continuity rows are built. Do not conflate them.

### `UnknownsLayout` and `TimeIntervalManager`

`UnknownsLayout(n_theta, nx)` answers where things live in `theta_full`:
`layout.theta`, `layout.c(batch, shoot)`, `layout.n_shoots(batch)`. It is built
once, in `add_batch`.

`TimeIntervalManager(N_shoot, t_eval_measurements)` splits the grid: nodes are
placed at a constant stride `len(t) // N_shoot`, so when the division is not
exact the last interval is longer and the actual shot count `self.N_shoot`
differs from the request. `get_time_interval(shot)` returns
`(grid including the junction point, indices of the shot's measurements)`; the
last point of a shot grid is a junction only and does not enter the measurement
residual.

### `CollocationShooting(...)` — `gauss_newton/collocation_shooting.py`

Same constructor plus the collocation knobs (`K`, `n_sub`, `newton_tol`,
`newton_maxiter`, `rootfinder_plugin`, `rootfinder_options`, `n_threads`). It
replaces only `self.integrator`.

## `gauss_newton/normal_equations.py`

### `NormalEquations(H, g, J_G, R_G, rss, n_rows)`

| Member | Meaning |
|---|---|
| `from_jacobian(J, R, J_G, R_G)` | classmethod: `H = JᵀJ`, `g = JᵀR` |
| `n_cont` | number of continuity rows |
| `cont_sq()` | `‖R_G‖²` |
| `cost()` | `‖R‖² + ‖R_G‖²` |
| `merit(mu)` | `Φ_μ = ‖R‖² + (1/μ)‖R_G‖²` — what the step minimizes |
| `mu_curvature()` | starting μ: `‖J_G‖²_F / tr(H)` |
| `covariance_theta(n_theta, ridge=1e-8)` | `(cov, sigma2, dof)` from the KKT matrix |
| `correlation_theta(n_theta, ridge=1e-8)` | `(corr, cond)` |

### Accumulation

- `AccumulateMixin.normal_equations(theta_full)` → `NormalEquations`, building
  `H` and `g` by accumulation without ever forming `J`.
- `MultipleShootingAccum`, `CollocationShootingAccum` — the mixin applied to the
  two problem classes. **These are the classes to use** for identification runs.
- `normal_equations_of(problem, theta_full)` — accumulate if the problem can,
  otherwise go through `solve()`.

### Statistics

- `correlation_matrix(cov)` → `(corr, cond)`.
- `confidence_intervals(theta_opt, cov, dof, alpha=0.05)` → `(low, high)`.

## `gauss_newton/adaptive.py`

### `gn_step(ne, mu, lam, lambda_reg=0.0, lam_dual=None)`

One step from the μ-regularized saddle system, plus `pred` for the gain ratio.
Returns `(delta, pred)`, or `(delta, pred, nu)` when `lam_dual` is given (the
augmented-Lagrangian variant, used by the notebook, not by the loop). See
[math.md](math.md).

### `run_optimization_adaptive(problem, theta_full, ...)`

The only optimization loop. `problem` is anything with `solve` or
`normal_equations`. Returns `(theta_full, hist)`.

| Argument | Default | Meaning |
|---|---|---|
| `n_iter` | 40 | upper bound, not a budget — the loop stops by itself |
| `lam0` | 1e-3 | starting Marquardt damper; then driven by the gain ratio |
| `mu_rule` | `'curvature'` | `'curvature'` starts at `‖J_G‖²_F/tr(H)` and tightens by Powell's rule; `'ratio'` uses `‖R_G‖²/(κ‖R‖²)` and does worse on real data |
| `mu_dec`, `viol_target` | 0.5, 0.25 | Powell's tightening factor and target |
| `rss_stall_tol` | 0.99 | the gate: tighten μ only once the measurement residual has stalled |
| `rho_accept` | 0.0 | acceptance threshold on the gain ratio |
| `track_covariance` | `True` | compute confidence intervals per iteration (one `splu` each) |

`hist` holds per-iteration lists (rejected iterations repeat the previous
values): `theta`, `cost`, `mu`, `lam`, `r_meas`, `r_cont`, `ci_low`, `ci_high`,
`corr_cond`, plus `accepted` (indices) and `n_solves`. It is the input to
`plot_solution`.

## `gauss_newton/utils.py`

`plot_solution(fig=None, problem=None, theta_hist=None, ...)` builds the plotly
figure: phase trajectories (2D/3D), time series, parameter convergence with
confidence intervals, and residuals. The `plot_*` flags select panels;
`param_names`, `state_names`, `theta_true`, `ci_low_hist`, `ci_high_hist`
decorate them.

## `commom_utils/systems.py`, `commom_utils/system_config.py`

Concrete `ODESystem` subclasses (`LotkaVoltera`, `Attractor`, `Pendulum`,
`Integrator`, the vehicle models, …) and `create_system(cfg)` /
`create_mhe_params(...)`, which build a system and its MHE parameters from a
config dictionary.
