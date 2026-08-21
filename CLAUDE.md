# GaussNewton — working notes for Claude

Parameter identification for ODE systems: Gauss–Newton + multiple shooting,
with two sensitivity integrators (variational equations and Radau IIA
collocation).

**Read the documentation before changing anything** — it is the single source
of truth and must be updated with the code:

| Page | What is in it |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, class relationships, the contracts that must not break |
| [docs/api-reference.md](docs/api-reference.md) | Every public class and function, with shapes |
| [docs/math.md](docs/math.md) | The method, the μ/λ schedules, the covariance |
| [docs/notation.md](docs/notation.md) | Theory ↔ code symbol table |
| [docs/pitfalls.md](docs/pitfalls.md) | Non-obvious failure modes found the hard way |
| [docs/testing.md](docs/testing.md) | What each test guards |
| [docs/performance.md](docs/performance.md) | Measured timings |
| [docs/style.md](docs/style.md) | Comment and docstring rules |

## Rules for changes here

- **Style**: English everywhere in code — comments, docstrings, exception and
  assert texts. Docstrings stay short; explanations belong in `docs/`, formulas
  in the notebooks. Full rules: [docs/style.md](docs/style.md).
- **The user reads formulas in the notebooks, not in chat.** Derivations go to
  `theory_gauss_newton.ipynb`, `collocation.ipynb` or
  `adaptive_regularization.ipynb`.
- **Do not rename public names.** Notebooks under `experiments/` and `mpc/`
  import them, including the misspelled package `commom_utils`.
- **Do not change existing code without being asked.** New capabilities go into
  new classes or files (that is how `CollocationShooting` was added).
- **`pytests/regression_test.py` is the numerical reference.** A refactor must
  pass it untouched; regenerate (`GN_REGEN_REFERENCE=1`) only when the change of
  numbers is deliberate and explainable. Tolerances differ by case: collocation
  is compared at 1e-10, the adaptive `solve_ivp`/`odeint` path only at 1e-6
  (see [docs/pitfalls.md](docs/pitfalls.md)).
- **One quantity, one name** — [docs/notation.md](docs/notation.md). A second
  name for something that already has one means a layer boundary was crossed.

## Commands

```bash
uv run pytest pytests/            # 91 passed, 2 skipped (mhe/mpc need acados)
uvx ruff@0.15.8 check .           # the CI lint job is advisory
bash tools/setup_repo.sh          # register the nbstrip git filter (once per clone)
```

`GN_TEST_PLOT=1` makes the tests open their plotly figures.
