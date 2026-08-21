# Testing

```bash
uv run pytest pytests/                       # everything (91 passed, 2 skipped)
uv run pytest pytests/jacobian_fd_test.py    # Jacobian against finite differences
GN_TEST_PLOT=1 uv run pytest pytests/collocation_accum_test.py   # with figures
```

Plotly figures are not opened by default; `GN_TEST_PLOT=1` enables them.
`pytests/mhe_test.py` and `pytests/mpc_test.py` are skipped when acados is not
installed (`pytest.importorskip`).

Tests are run through pytest, never as scripts: `pyproject.toml` puts the
repository root on the path (`[tool.pytest.ini_options] pythonpath = ["."]`),
so the test files carry no `sys.path` preamble of their own.

## Checks against an external reference

These are the tests that can catch a mistake shared by the whole library,
because they compare against something computed outside it.

| Test | External reference |
|---|---|
| `pytests/jacobian_fd_test.py` | finite differences of the residuals — also covers the gradient, the collocation path and, as a negative result, the loss of accuracy on a discontinuous input |
| `pytests/covariance_test.py` | single shooting (no constraints at all) and Monte-Carlo: spread of the estimates and coverage of the 95% intervals |
| `pytests/scaling_test.py` | the same problem written in different units — an exact reference for `cont_scale` |
| `adaptive_test.py::test_step_matches_dense_saddle_solve` | dense `numpy.linalg.solve` of the same saddle system (plain and with `lam_dual`) |
| `collocation_test.py::test_integrator_matches_reference` | the variational-equation integrator |
| `collocation_test.py::test_radau_tables` | analytic Radau IIA values |

## The rest

| Test | Guards |
|---|---|
| `pytests/accumulated_test.py` | the accumulated `H`/`g` path against the dense one: matrices, step, covariance, and a full identification run |
| `pytests/adaptive_test.py` | `pred > 0` across regimes, identification, `N_shoot=5` on the attractor (the `rss_stall_tol` gate), early stop on single shooting, the collocation path |
| `pytests/collocation_test.py` | Radau tables, the IND property, a stiff simulation, identification, agreement with multiple shooting |
| `pytests/collocation_accum_test.py` | identification through `CollocationShootingAccum` |
| `pytests/gauss_newton_test.py` | the basic identification run |
| `pytests/sensitivity_test.py` | round trip of the flat integrator layout — the one thing that would otherwise silently permute axes |
| `pytests/systems_smoke_test.py` | every system in `commom_utils/systems.py` has consistent dimensions and compiles |
| `pytests/regression_test.py` | see below |

## The frozen reference

`pytests/regression_test.py` holds frozen `J`, `R`, `J_G`, `R_G`, `H`, `g` and `delta`
for four problems. It catches changes in the **numbers** that do not make
anything fail — a permuted axis, a different order of operations. A refactor is
expected to pass it untouched.

Tolerances differ by case on purpose: collocation runs on a fixed grid and is
compared at 1e-10, while the adaptive `solve_ivp`/`odeint` path is compared only
at 1e-6, because the output of an adaptive integrator is not reproducible across
machines beyond ~1e-8 (see [pitfalls.md](pitfalls.md)).

Regenerate with `GN_REGEN_REFERENCE=1` **only** when the change in numbers is
deliberate and can be explained. After regeneration the test skips itself for
that run ("nothing to compare against").
