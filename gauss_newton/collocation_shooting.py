# -*- coding: utf-8 -*-
"""Multiple shooting с коллокационным интегратором (Радо IIA + IND).

CollocationShooting отличается от MultipleShooting только интегратором:
вместо адаптивного solve_ivp/odeint расширенной системы — марш по элементам
с Ньютоном и рекурсиями чувствительностей (см. collocation.ipynb). Вся
сборка J/R/J_G/R_G, mu-машинерия и run_optimization наследуются как есть.
Пригоден для жёстких систем (L-устойчивость Радо IIA).
"""
from commom_utils.collocation import CollocationSystemJacobian
from gauss_newton.gauss_newton_math import MultipleShooting


class CollocationShooting(MultipleShooting):

    def __init__(self, system, N_shoot, gamma=None, c0_cost=1, verbose=False,
                 K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25,
                 rootfinder_plugin='newton', rootfinder_options=None,
                 n_threads=None):
        # use_jax=True направляет _solve_batch в батчевый вход
        # get_jacobian_solution_jax_batch — здесь он реализован коллокационным
        # маршем с потоковым распараллеливанием по шутам (JAX не используется)
        super().__init__(system, N_shoot, gamma=gamma, c0_cost=c0_cost,
                         use_jax=True, verbose=verbose)
        # Родитель уже создал SystemJacobian; заменяем на коллокационный
        # (та же символика CasADi, переопределены только интеграторы)
        self.system = CollocationSystemJacobian(
            system, K=K, n_sub=n_sub,
            newton_tol=newton_tol, newton_maxiter=newton_maxiter,
            rootfinder_plugin=rootfinder_plugin,
            rootfinder_options=rootfinder_options,
            n_threads=n_threads)
