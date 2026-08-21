# Pitfalls

Every item here was found in practice and cost time. Read this before changing
anything numerical.

## Method and accuracy

- **Do not tighten μ while `rss` is still falling fast.** During globalization
  the junction residual oscillates, so Powell's rule without a gate fires on
  every accepted step, μ collapses within 3–5 iterations and the solution locks
  onto a consistent trajectory far from the measurements (Lorenz attractor,
  `N_shoot=5`: rel. error 1.18 at `r_cont ~ 1e-10`, for any `n_iter`). The
  `rss_stall_tol` gate in `run_optimization_adaptive` fixes this without
  changing the `N_shoot` 10/20 cases.
- **The method is not invariant to the units of the state.** The `−μI` block
  weights the junction residuals of all states equally. The same problem with
  one coordinate 1000× larger gives rel. error 2.8e-1 instead of 9.5e-3, and
  with a hand-tuned gamma a junction of 1.1e-5 instead of 1.7e-12. Use
  `cont_scale`; the weights must be **fixed once** — changing them between
  iterations changes the merit function, and then the gain ratio compares
  different things.
- **Do not compute the covariance as `(H + J_GᵀJ_G)⁻¹`.** That treats
  continuity as an observation with an arbitrary weight. The tell-tale signs
  are a standard error that depends on the scale of `J_G` and disagreement with
  single shooting (which has no constraints). External checks: agreement with
  single shooting and Monte-Carlo interval coverage
  (`pytests/covariance_test.py`).
- **Lagrange multiplier updates (AL) with a Gauss–Newton Hessian do not
  advance feasibility** on strongly nonlinear `G`: the model does not see ∇²G
  (on Lorenz the actual merit curvature is ~2e4 while `pred` is ~1e-8), so
  λ-steps are rejected by the gain ratio. Experiments and conclusion:
  `adaptive_regularization.ipynb` §9. Do not attempt to enable AL in the loop
  without second derivatives.
- **Conditioning of the normal equations is not a problem** — measured:
  `cond(J) ≈ 1.8e2` → `cond(H) ≈ 3.3e4`, about 2 digits lost out of 16. A
  sparse QR or LSMR is not needed for these problems.

## Integrators

- **A time-discontinuous input destroys the sensitivities of the explicit
  adaptive `solve_ivp`.** It steps over the kink, error control does not work
  there, and the Jacobian disagrees with finite differences by 3+ orders of
  magnitude (`jacobian_fd_test::test_discontinuous_input_degrades_sensitivities`).
  Work around it by placing a shot boundary at the discontinuity, or use
  collocation, whose element grid is fixed.
- **`jax.experimental.odeint` defaults to tolerances of 1.4e-8** and that is the
  main slowdown; always pass `rtol=self.RTOL, atol=self.ATOL`. The one
  exception is `SystemIntegrator.step_jax` (MPC simulation), where the defaults
  are kept deliberately.
- **The output of an adaptive integrator is not reproducible across machines
  beyond ~1e-8.** Step size is chosen by comparing an error estimate against a
  tolerance, and different arithmetic (another CPU, another BLAS build) yields a
  different sequence of steps. Observed on a GitHub Actions runner:
  `integrator_obs` diverged from the local reference by 1.0e-8 at a tolerance of
  1e-10. Never freeze a `solve_ivp`/`odeint` result tighter than 1e-6;
  collocation has a fixed grid and no such problem.
- **Collocation runs at a fixed step**: accuracy is set by `n_sub` and there is
  no error estimate — check convergence by running `n_sub` and `2*n_sub`.
- **A Radau basis built on the collocation points only** (degree K−1, without
  τ₀ = 0) is **degenerate** — constants are in the kernel of D. The correct
  formulation is degree K with τ₀ = 0.
- **Finite-difference checks of the Jacobian: a step of 1e-7 is not always
  right.** An error that falls exactly as 1/h is integrator round-off noise, not
  a Jacobian bug (the `Integrator` system's state grows as t², and there a step
  of ~1e-4 is needed). Running the same problem through collocation
  distinguishes the two: IND gives exact derivatives of the scheme and agrees to
  ~1e-10.

## CasADi and JAX

- **The ODE right-hand side is called tens of thousands of times per solve**, so
  anything that can be computed once per call must be. Input signals used to be
  computed three times (via `f`, `df_dx`, `df_dtheta`); removing that gave 1.7×
  on the numpy path (3982 → 2294 ms on `Integrator`, 16 points, 2 shots).
- **Python dicts: `True == 1`**, so cache keys `(N, True)` and `(N, 1)` collide.
  Use string prefixes in the keys or separate dictionaries (`_accum_cache` and
  `_map_cache` in the collocation module).
- **`ca.rootfinder` supports neither codegen nor JIT**, and `map('openmp')` is
  slower than `map('thread')` (CasADi releases the GIL, so threads suffice).
- **`abstol` of the Newton rootfinder is an absolute tolerance on the residual
  Φ**, which has the scale of the state: at |x| ~ 1e6 a threshold of 1e-10 is
  unreachable (round-off floor ~1e-9) and Newton "fails to converge" only on
  large data. The cure is `abstolStep` (a tolerance on the step; combined with
  `abstol` it acts as OR, and the step is checked *before* it is applied).
- **`error_on_fail=True` on a rootfinder prints multi-line C++ dumps of its
  inputs** even when the exception is caught in Python, and `map('thread')`
  makes it worse. The quiet path is `error_on_fail=False` plus an explicit
  `stage_res` output (the scaled residual at the solution) checked after the
  march. Note that a non-converged element with `error_on_fail=False` returns
  its last iterate **without** NaNs — without the `stage_res` check those are
  silently wrong numbers.
- **jaxadi functions return a *list* of outputs**: without `[0]` the result is
  `(1, nx, nx)` and a matrix product against `(1, ·, ·)` silently gains an axis.
  All wrappers take `[0]` today, but new code written against the `_*_jax_ca`
  handles can easily forget again.
- **First build of the `mapaccum` functions takes ~0.6 s** (once per grid
  length); JAX JIT takes ~5 s.

## Repository hygiene

- `experiments/datasets/*` is in `.gitignore` **with the star**: git does not
  descend into an excluded directory, so with the form `datasets/` the
  `!datasets/README.md` rule cannot re-open it. The rule `experiments/*.csv` is
  not recursive — moving data deeper makes it stop matching. Check with
  `git check-ignore -v`, or 215 MB end up in the history.
- `tools/setup_repo.sh` registers the `nbstrip` git filter. `.gitattributes`
  declares `filter=nbstrip`, but the filter itself is a local git setting and is
  not stored in the repository; without running the script, heavy notebook
  output comes back into the history.
- `experiments/real_data_cars/mhe_test_rosbag.ipynb` **deliberately shadows** our
  `mhe` package with the one from SDA (`sys.path.insert(0, CODEGEN_ROOT)` in its
  first cell). The shared bootstrap cell must not be added there — it would
  override that order — and its kernel must be fresh, not shared with
  `sintetic_data/mhe_test.ipynb`. A solver cache in SDA may have been built by a
  different acados version, in which case loading fails with
  `KeyError: 'code_gen_opts'`; the cure is `MHE_NB_FORCE_REBUILD=1`.

## Known limitations

- `SyntheticDataGenerator` adds noise **to the states** before computing `h(x)`
  (process noise, not measurement noise); `MHESyntheticDataGenerator` adds it to
  the output.
- Bounds on θ exist only in the MHE part; the Gauss–Newton path has no parameter
  bounds.
