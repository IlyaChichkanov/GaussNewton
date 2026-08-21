# Performance

Measured on real data: 8000 measurement points, 10 shots.

| Path | One `solve` |
|---|---|
| `MultipleShooting` (JAX) | ~0.2–0.4 s |
| `CollocationShooting` | ~0.33–0.43 s |

Breakdown of the collocation number: ~0.2 s marching in C++, ~0.05 s for the
numpy sensitivity recursions, ~0.03 s for the observations.

One-off build costs: the `mapaccum` functions take ~0.6 s per distinct grid
length; the JAX JIT takes ~5 s.

Things that turned out to matter:

- The ODE right-hand side is evaluated tens of thousands of times per solve.
  Computing the input signals once per call instead of three times (via `f`,
  `df_dx`, `df_dtheta`) gave 1.7× on the numpy path — 3982 → 2294 ms on
  `Integrator` with 16 points and 2 shots.
- `map('thread')` beats `map('openmp')` for the collocation march: CasADi
  releases the GIL, so threads are enough.
- Accumulating `H` and `g` avoids building the sparse `J` at all; `solve()` is
  kept only as the reference assembly used by the finite-difference test.
