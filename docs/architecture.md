# Architecture

Four layers, each of which knows only the one below it:

```
optimization      gauss_newton/adaptive.py            gn_step, run_optimization_adaptive
                        │  NormalEquations (H, g, J_G, R_G)
normal equations  gauss_newton/normal_equations.py    NormalEquations, AccumulateMixin
                        │  ShootRows per shot
problem assembly  gauss_newton/problem.py             MultipleShooting (+ CollocationShooting)
                        │  self.integrator            │  self.system
sensitivities     ode_system.VariationalIntegrator    model  commom_utils/ode_system.py
                  collocation.CollocationIntegrator          CompiledModel over ODESystem
```

The optimization layer never sees an integrator, and the problem layer never
sees a raw `ODESystem` — `MultipleShooting.__init__` compiles it into a
`CompiledModel` once.

## Model layer — `commom_utils/ode_system.py`

- **`ODESystem`** — the symbolic model a user writes: `get_derivative`,
  `observation`, `get_input_signals`, and the sizes `nx`, `n_theta`, `nu`,
  `n_obs`. `get_input_signals` is called *inside* the ODE right-hand side, so
  it must be traceable: `jnp` only, no `math.*`, no Python `if t < ...`
  (use `jnp.where`).
- **`CompiledModel`** — the model and nothing else: CasADi/jaxadi functions for
  `f`, `h` and their Jacobians `df_dx`, `df_dtheta`, `dh_dx`, `dh_dtheta`, plus
  `observation_batch`, `inverse_h`, `dims()` and the `identity_observation`
  flag. It does not integrate anything.
- **`VariationalIntegrator`** — sensitivities by integrating the variational
  equations alongside the state. Holds a `CompiledModel` by composition
  (the constructor also accepts a raw `ODESystem`) and its own `ATOL`, `RTOL`,
  `method`. Both back ends are explicit (scipy `RK45`, jax `dopri`), so it is
  not applicable to stiff systems.
- **`SystemIntegrator(CompiledModel)`** — a different job: integration with an
  input `u` *held by the caller* (MPC simulation) instead of taken from the
  model. Used by `mpc/`, not by the identification path.
- **`SyntheticDataGenerator`, `MHESyntheticDataGenerator`** — test data.

## Sensitivity container — `commom_utils/sensitivity.py`

Integrators are forced to return a flat array: both `solve_ivp` and jax
`odeint` integrate vectors, so the extended state is laid out as

```
[x (nx); S_theta.flatten() (nx*n_theta); S_c.flatten() (nx*nx)]
```

one column per grid point. `SensitivityTrajectory` is the boundary: the flat
layout exists only between an integrator and `unpack()`; everything downstream
uses named arrays with checked shapes. The block order is known to this module
alone — `initial_flat_row` and `split_row` are the only other places that
touch it.

## Collocation — `commom_utils/collocation.py`

- **`RadauTables(K)`** — Radau IIA nodes for K ∈ {1, 2, 3}, differentiation
  matrix D̃ = [d0 | D1], Butcher table `butcher_a = inv(D1)`.
- **`CollocationStepFunctions`** — builds and caches the CasADi functions of
  one step: stage residual, `rootfinder`, the `step_sens`/`step_x` pair, and
  their `mapaccum`/`map('thread')` wrappers.
- **`CollocationIntegrator`** — the march driver and the convergence policy. A
  drop-in replacement for `VariationalIntegrator` with the same output
  contract, suitable for stiff systems (Radau IIA is L-stable).

## Problem assembly — `gauss_newton/problem.py`

`MultipleShooting` keeps the model and the integrator apart:
`self.system` (a `CompiledModel`: observations, dimensions, `inverse_h`) and
`self.integrator` (shot sensitivities). `CollocationShooting`
(`gauss_newton/collocation_shooting.py`) subclasses it and swaps **only**
`self.integrator`.

- **`UnknownsLayout`** — where `theta_full = [θ; c_1..c_T]` lives: `layout.theta`,
  `layout.c(batch, shoot)`. Built once in `add_batch`.
- **`TimeIntervalManager`** — splits a measurement grid into shots; the actual
  shot count can differ from the requested `N_shoot`.
- **`ShootRows`** — the blocks of one shot. `J_theta`/`J_c` are rows of the
  *residual* Jacobian; `S_theta_end`/`S_c_end` are state sensitivities at the
  end of the shot, from which the continuity rows are built. Two different
  things, two different names.
- **`shoot_rows`** — the shared core of both assembly paths: integrate the
  shots, apply observations and weights, return `ShootRows`.
- **`solve(theta_full)` → `(J, R, J_G, R_G)`** — the reference assembly with an
  explicit sparse `J`. It is not used inside the optimization loop; it exists
  so that `pytests/jacobian_fd_test.py` can compare `J` against finite
  differences.

## Normal equations — `gauss_newton/normal_equations.py`

How to get `H` and `g`. Two sources, one class:

- `NormalEquations.from_jacobian(J, R, J_G, R_G)` — `H = JᵀJ`;
- `AccumulateMixin.normal_equations(theta_full)` — accumulates
  `H = Σ Jᵢᵀ Jᵢ` with einsums over the points of each shot, exploiting the
  arrow structure (shared θθ block, per-shot θc and cc blocks). The big `J` is
  never built. Concrete classes: `MultipleShootingAccum`,
  `CollocationShootingAccum`; `normal_equations_of(problem, theta_full)` picks
  whichever path the problem supports.

`NormalEquations` also carries the derived quantities the loop needs:
`merit(mu)`, `mu_curvature()`, `covariance_theta(n_theta)`,
`correlation_theta(n_theta)`.

## Optimization — `gauss_newton/adaptive.py`

What to do with `H` and `g`. `gn_step(ne, mu, lam)` solves one saddle system
and returns `(delta, pred)`; `run_optimization_adaptive` is the only
optimization loop in the project and drives λ and μ automatically. See
[math.md](math.md).

## Plotting — `gauss_newton/utils.py`

`plot_solution` builds the plotly figure from a problem and the `hist` dict
returned by `run_optimization_adaptive`: phase trajectories (2D/3D), time
series, parameter convergence with confidence intervals, measurement and
continuity residuals.

## Contracts that must not break

- `get_jacobian_solution(c0, theta, t_eval)` — both integrators — returns a
  matrix whose rows are `[x; S_theta.flatten(); S_c.flatten()]` (C-order) and
  whose columns are the points of `t_eval`. Do not unpack it by hand: use
  `SensitivityTrajectory.unpack(flat, nx, n_theta)`.
- The problem layer calls the integrator **only** through `self.integrator`
  and the model **only** through `self.system`.
- `solve(theta_full)` → `(J, R, J_G, R_G)`;
  `normal_equations(theta_full)` → `NormalEquations`. The optimization layer
  knows these two and nothing else.
- One quantity, one name — see [notation.md](notation.md). A second name for
  something that already has one is a sign that something went wrong.
- `pytests/regression_test.py` freezes the numbers of a GN step. A refactor
  must pass it untouched; regenerate (`GN_REGEN_REFERENCE=1`) only when the
  change of numbers is deliberate and explainable.
- Public names are load-bearing: notebooks under `experiments/` and `mpc/`
  import them. (Including the misspelled package name `commom_utils`.)

## Repository layout

```
commom_utils/     model, sensitivities, collocation, example systems
gauss_newton/     problem assembly, normal equations, optimization, plotting
mhe/  mpc/        moving horizon estimation and MPC on acados
experiments/      run notebooks: sintetic_data/, real_data_cars/, datasets/ (outside git)
pytests/          test suite
tools/            nbstrip.py (git filter) and setup_repo.sh
docs/             this documentation
*.ipynb           theory notebooks
```

Notebooks under `experiments/` open with a bootstrap cell that walks up from
the current directory to `pyproject.toml`, putting the repository root on
`sys.path` and defining `DATASETS` (overridable with `GN_DATASETS`) and
`CODEGEN = REPO/tmp_generated`.
