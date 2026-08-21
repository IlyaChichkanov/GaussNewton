"""Multiple shooting driven by the collocation integrator (Radau IIA + IND)."""
from commom_utils.collocation import CollocationIntegrator
from gauss_newton.problem import MultipleShooting


class CollocationShooting(MultipleShooting):
    """MultipleShooting with self.integrator replaced; suitable for stiff systems.

    Everything else - assembly of J/R/J_G/R_G, the mu machinery, the compiled
    model in self.system - is inherited unchanged. See docs/architecture.md.
    """

    def __init__(self, system, N_shoot, gamma=None, c0_cost=1, verbose=False,
                 K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25,
                 rootfinder_plugin='newton', rootfinder_options=None,
                 n_threads=None, cont_scale=None):
        # use_jax=True routes shoot_rows through the batched entry point
        # get_jacobian_solution_jax_batch, which the collocation integrator
        # implements with threads (no JAX involved)
        super().__init__(system, N_shoot, gamma=gamma, c0_cost=c0_cost,
                         use_jax=True, verbose=verbose, cont_scale=cont_scale)
        self.integrator = CollocationIntegrator(
            self.system, K=K, n_sub=n_sub,
            newton_tol=newton_tol, newton_maxiter=newton_maxiter,
            rootfinder_plugin=rootfinder_plugin,
            rootfinder_options=rootfinder_options,
            n_threads=n_threads)
