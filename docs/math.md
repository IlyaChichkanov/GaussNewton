# The method

Full derivations are in the theory notebooks (Russian); this page states the
results that the code implements and names the functions that implement them.

## Problem

Unknowns are `theta_full = [θ; c_1..c_T]` — the parameters plus the initial
state of every shot.

- **Measurement residuals.** `r_i = W (y_i − h(x(t_i)))`, and `J` is their
  Jacobian with respect to `[θ; c]`. In the code `gamma` is √W: the residual is
  multiplied by it and the cost squares it, so `gamma = 1` means σ = 1.
- **Continuity.** `G_j = x_j(t_{j+1}; c_j, θ) − c_{j+1}`, giving the rows
  `(J_G, R_G)`. Continuity is a *constraint*, not an observation — both the
  step and the covariance keep it in a separate block.
- **Sign convention**, checked by `pytests/jacobian_fd_test.py`:
  `[J; J_G] = −∂[R; R_G]/∂theta_full`. `J` is the Jacobian of the
  *predictions* while the residual is `R = W(y − h)`, hence the minus.

## The step

`gauss_newton/adaptive.py::gn_step` — the only implementation — solves

```
[[H + λ_reg·I + λ·diag(H),  J_Gᵀ],   [δ ]   [g  ]
 [J_G,                     −μ·I ]] · [ν ] = [R_G]
```

with `H = JᵀJ` and `g = JᵀR`. Eliminating ν shows this is exactly the
Gauss–Newton step for

```
Φ_μ = ‖R‖² + (1/μ)‖R_G‖²
```

so μ is the relaxation of the continuity constraints. The saddle form is used
rather than the eliminated form `H + (1/μ)J_GᵀJ_G` because μ enters linearly:
as μ → 0 the eliminated matrix loses conditioning as O(1/μ), the saddle matrix
does not. Exact junction is therefore reached by driving μ → 0.

`pred = δᵀ(g_eff + Dδ) ≥ 0` is the reduction of Φ_μ predicted by the model and
feeds the gain ratio; `pred ≤ 0` can only come from a numerical failure and the
loop treats it as a rejected step.

This is the quadratic penalty method in saddle form — equivalently, stabilized
SQP with the multipliers held at zero. ν is solved for and discarded; it is not
accumulated between iterations.

### Why not augmented Lagrangian

An AL step is the same saddle system with the constraint right-hand side
shifted to `R_G − μλ`; then `ν = λ + (J_Gδ − R_G)/μ` is precisely the
first-order multiplier update, `g_eff` gains `−J_Gᵀλ`, and the `pred` formula
is unchanged. `gn_step(..., lam_dual=λ)` implements this and returns
`(δ, pred, ν)`.

It is **not** used by the loop. With a Gauss–Newton Hessian (no ∇²G term)
multiplier updates do not advance feasibility on strongly nonlinear
constraints — on the Lorenz attractor the actual merit curvature is ~2e4 while
`pred` is ~1e-8, and λ-steps are rejected by the gain ratio. Three variants
were tried (update from the first iteration; the LANCELOT η-gate; a two-phase
schedule) and all did worse than continuation in μ. The experiments and the
comparison table are in `adaptive_regularization.ipynb` §9. Do not try to
"switch AL on" in the loop without second derivatives.

## λ and μ schedules

`run_optimization_adaptive` is the only loop, and it needs no manual μ₀,
`mu_dec` or λ tuning.

| | how it is chosen |
|---|---|
| λ (Marquardt damper) | gain ratio, Nielsen's scheme |
| μ (continuity weight) | starts at the curvature ratio ‖J_G‖²_F / tr(H), tightened by Powell's rule |
| step acceptance | ρ > 0 measured on Φ_μ — the same function the step was computed for |
| stopping | automatic: a run of rejections, stalling, or pred ≈ 0 |

**The μ gate.** Powell's rule alone (tighten whenever the constraint violation
fails to drop by `viol_target`) fires on early iterations, when `rss` is still
falling by orders of magnitude and the junction merely oscillates. μ is then
driven to ~1e-7 within a few steps, the constraints start to dominate, and the
solution locks onto a consistent trajectory far away from the measurements —
exactly the "does not converge with few shooting intervals" symptom. The gate
`rss_stall_tol` (default 0.99) only allows tightening when the measurement
residual has also stalled. With it the Lorenz attractor converges from θ₀ = 0
with `N_shoot=5` (rel. error 2.6e-4, ‖R_G‖² ≈ 4e-10) where before it stalled at
rel. error 1.18; `N_shoot` of 10 and 20 are unaffected. See
`pytests/adaptive_test.py::test_attractor_converges_with_few_shoots`.

## Sensitivities

**Variational equations** (the explicit path): `Ṡ_θ = f_x S_θ + f_θ` and
`Ṡ_c = f_x S_c`, integrated together with the state. Both back ends are
explicit, so this path does not apply to stiff systems.

**Collocation** (the stiff path). On one element the stage equations are

```
z = A x_prev + h·B·F(z, θ),   A = 1_K ⊗ I,   B = a ⊗ I,   a = D1⁻¹
```

with `a` the Radau IIA Butcher table (K = 3: order 5, L-stable). Newton uses
`M = I − h·B·blkdiag(f_x)`. Sensitivities come from the implicit function
theorem — internal numerical differentiation, i.e. exact derivatives of the
discrete scheme:

```
Ψ = (e_Kᵀ ⊗ I) M⁻¹ A,        Γ = (e_Kᵀ ⊗ I) M⁻¹ h B F_θ
S_c ← Ψ S_c    (S_c(0) = I),  S_θ ← Ψ S_θ + Γ   (S_θ(0) = 0)
```

In the code Ψ and Γ are obtained as `ca.jacobian` of the rootfinder output —
CasADi differentiates through it by the same theorem.

## Covariance of θ

`σ²` times the θ-block of the inverse KKT matrix

```
[[H, J_Gᵀ], [J_G, 0]]
```

computed with a sparse LU. That block is the projection of `H⁻¹` onto the
tangent subspace of the constraints; it is the same matrix `gn_step` solves at
μ = 0, so the cost is one `splu` plus `n_theta` back-substitutions.
`σ² = ‖R‖²/dof` uses the measurement residual only — continuity rows are not
observations and do not enter the residual sum. `dof = n_rows − n_theta − nx`,
that is, measurements minus *free* unknowns.

Computing it as `(H + J_GᵀJ_G)⁻¹` is wrong: that treats continuity as an
observation with an arbitrary weight. The symptoms are a standard error that
depends on the scale of `J_G` (multiplying `J_G` by 1000 changed it by 2.4×)
and disagreement with single shooting, which has no constraints at all. The
old formula inflated the intervals by 2.4× at `N_shoot=5` and 6.1× at
`N_shoot=20`. External checks: `pytests/covariance_test.py` compares against
single shooting and against Monte-Carlo coverage of the 95% intervals.

Confidence intervals (`confidence_intervals`) are the usual two-sided Student
intervals: `se_i = √Cov_ii`, `CI_i = θ_i ± t_{α/2, dof}·se_i`.

**Identifiability diagnostics.** `correlation_theta` normalizes the same
covariance; `|corr|` close to 1 means the parameters are only distinguishable
in combination (Lotka–Volterra: `corr(α, β) = 0.997`). Its condition number is
recorded per iteration as `hist['corr_cond']` — growth means the parameters are
drifting into a flat valley.

## State scaling

The `−μI` block weights the junction residuals of all states equally, so the
method is not invariant to the units of the state. The same problem with one
coordinate expressed in units 1000× larger gives rel. error 2.8e-1 instead of
9.5e-3, and a junction of 1.1e-5 instead of 1.7e-12.

`cont_scale` (on `MultipleShooting` and `CollocationShooting`) divides the
continuity rows by a state scale: `None` (default, weights exactly 1), an
`(nx,)` array, or `'auto'` — the RMS of the measurements, available only when
`h(x) = x`. The weights are fixed once: changing them between iterations would
change the merit function itself, and the gain ratio would be comparing
different things. Checked by `pytests/scaling_test.py`.

## Conditioning

Measured, and not a problem on these tasks: `cond(J) ≈ 1.8e2` gives
`cond(H) ≈ 3.3e4`, i.e. about 2 digits lost out of 16. Moving to a sparse QR or
LSMR is not warranted.
