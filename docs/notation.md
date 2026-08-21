# Notation: theory ↔ code

One quantity, one name. The table exists because the notebooks are written in
mathematical notation and the code in programming notation, and the two used to
disagree: `H` meant the design matrix in `theory_gauss_newton.ipynb` and the
normal matrix `JᵀJ` in the code. The notebooks now follow the code.

## Model and data

| Theory | Code | Meaning |
|---|---|---|
| `x ∈ ℝ^{n_x}` | `x`, `nx` | state |
| `θ ∈ ℝ^{n_θ}` | `theta`, `n_theta` | estimated parameters |
| `u(t)` | `u`, `nu` | input signals, `ODESystem.get_input_signals` |
| `f(t, x, θ)` | `CompiledModel.f` | right-hand side |
| `h(x, θ)` | `CompiledModel.h` | observation function, `n_obs` components |
| `f_x`, `f_θ` | `df_dx`, `df_dtheta` | Jacobians of the right-hand side |
| `h_x`, `h_θ` | `dh_dx`, `dh_dtheta` | Jacobians of the observation |
| `y_i` | `state_measured[i]` | measurement at `t_i` |
| `W_i = Σ_i^{-1}` | `gamma²` | weights; **`gamma` is √W** — the residual is multiplied by it, the cost squares it |

## Unknowns and sensitivities

| Theory | Code | Meaning |
|---|---|---|
| `p = [θ; c_1 … c_T]` | `theta_full`, `UnknownsLayout` | vector of unknowns |
| `c_j` | shot `c0`, `layout.c(batch, shoot)` | initial state of shot `j` |
| `J_θ(t) = ∂x/∂θ` | `S_theta`, `(m, nx, n_theta)` | sensitivity to the parameters |
| `J_{c_0}(t) = ∂x/∂c_0` | `S_c`, `(m, nx, nx)` | sensitivity to the initial state |
| — | `SensitivityTrajectory` | `x`, `S_theta`, `S_c` on a grid |
| `Ψ`, `Γ` | `Psi`, `Gamma` | element transition and parameter contribution (collocation) |

The document "MS and Orthogonal Collocations" writes the initial state as `s`;
in multiple shooting that is the `c_j` of a particular shot, and the code has
no second name for it: `ShootRows.J_c`, `H_theta_c`, `H_c`, `g_c`.

## The Gauss–Newton system

| Theory (notebooks **before** the cleanup) | Code and notebooks **now** | Meaning |
|---|---|---|
| `H(p)` | `J` | Jacobian of the measurement residuals |
| `G(p)` | `J_G` | Jacobian of the continuity residuals |
| `r`, `h_cont` | `R`, `R_G` | the residuals themselves |
| `HᵀWH` | `H = JᵀJ` | the **normal** matrix (weights already inside `J`) |
| `HᵀWr` | `g = JᵀR` | gradient |
| `ρ` | `1/mu` | weight of the quadratic penalty |
| `diag(λ)` | `lambda_reg·I + lam·diag(H)` | step damping |
| `λ` (multipliers) | `nu` in `gn_step` | Lagrange multipliers |

The step (`gauss_newton/adaptive.py::gn_step`):

```
[[H + lambda_reg·I + lam·diag(H),  J_Gᵀ],   [delta]   [g  ]
 [J_G,                            -mu·I]] · [nu   ] = [R_G]
```

Sign convention, checked by `pytests/jacobian_fd_test.py`:

```
[J; J_G] = −∂[R; R_G]/∂theta_full
```

`J` is the Jacobian of the **predictions** while the residual is `R = W(y − h)`,
hence the minus.

## Where things live

| Layer | File | Responsible for |
|---|---|---|
| model | `commom_utils/ode_system.py` | `ODESystem`, `CompiledModel`, `VariationalIntegrator`, `SystemIntegrator` |
| sensitivities | `commom_utils/sensitivity.py` | `SensitivityTrajectory`, `group_by_grid_length` |
| collocation | `commom_utils/collocation.py` | `RadauTables`, `CollocationStepFunctions`, `CollocationIntegrator` |
| problem assembly | `gauss_newton/problem.py` | `ShootRows`, `UnknownsLayout`, `MultipleShooting` |
| normal equations | `gauss_newton/normal_equations.py` | `NormalEquations`, accumulation of `H`/`g`, covariance, CI |
| optimization | `gauss_newton/adaptive.py` | `gn_step`, `run_optimization_adaptive` |
