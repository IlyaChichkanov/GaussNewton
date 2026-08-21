# GaussNewton documentation

Parameter identification for ODE systems: Gauss–Newton with multiple shooting,
two sensitivity integrators (variational equations and Radau IIA collocation),
adaptive regularization, covariance and confidence intervals.

| Page | What is in it |
|---|---|
| [architecture.md](architecture.md) | Layers, classes and how they call each other |
| [api-reference.md](api-reference.md) | Every public class and function: purpose, arguments, shapes |
| [math.md](math.md) | The method: residuals, saddle system, μ and λ schedules, sensitivities, covariance |
| [notation.md](notation.md) | Theory ↔ code symbol table |
| [pitfalls.md](pitfalls.md) | Non-obvious failure modes, all found the hard way |
| [testing.md](testing.md) | What each test guards and which external reference it uses |
| [performance.md](performance.md) | Measured timings |
| [style.md](style.md) | Comment and docstring rules for this repository |

Reading order for a newcomer: [architecture.md](architecture.md) →
[math.md](math.md) → [api-reference.md](api-reference.md). Before changing
anything numerical, read [pitfalls.md](pitfalls.md) and
[testing.md](testing.md).

## Theory notebooks

Full derivations live in notebooks at the repository root (in Russian) and are
the authority for the formulas; this documentation states results and points
back to them.

| Notebook | Subject |
|---|---|
| `theory_gauss_newton.ipynb` | Gauss–Newton, multiple shooting, covariance, confidence intervals |
| `collocation.ipynb` | Orthogonal collocation on finite elements, Radau IIA, the Ψ/Γ recursions, verification |
| `adaptive_regularization.ipynb` | What μ means, Nielsen's λ, Powell's μ with the stall gate, the augmented-Lagrangian experiment (§9) |
| `theory_mhe.ipynb` | Moving horizon estimation |
