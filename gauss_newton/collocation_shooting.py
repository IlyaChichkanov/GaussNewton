# -*- coding: utf-8 -*-
"""Multiple shooting с коллокационным интегратором (Радо IIA + IND).

CollocationShooting отличается от MultipleShooting только интегратором:
вместо адаптивного solve_ivp/odeint расширенной системы — марш по элементам
с Ньютоном и рекурсиями чувствительностей (см. collocation.ipynb). Модель
(self.system, CompiledModel) остаётся общей — подменяется ТОЛЬКО
self.integrator; вся сборка J/R/J_G/R_G и вся mu-машинерия наследуются
как есть. Пригоден для жёстких систем (L-устойчивость Радо IIA).
"""
from commom_utils.collocation import CollocationIntegrator
from gauss_newton.problem import MultipleShooting


class CollocationShooting(MultipleShooting):

    def __init__(self, system, N_shoot, gamma=None, c0_cost=1, verbose=False,
                 K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25,
                 rootfinder_plugin='newton', rootfinder_options=None,
                 n_threads=None):
        # use_jax=True направляет shoot_rows в батчевый вход
        # get_jacobian_solution_jax_batch — у коллокационного интегратора он
        # реализован потоковым маршем по шутам (JAX не используется)
        super().__init__(system, N_shoot, gamma=gamma, c0_cost=c0_cost,
                         use_jax=True, verbose=verbose)
        # Родитель уже скомпилировал модель (self.system) и создал
        # вариационный интегратор; заменяем только интегратор, модель общая
        self.integrator = CollocationIntegrator(
            self.system, K=K, n_sub=n_sub,
            newton_tol=newton_tol, newton_maxiter=newton_maxiter,
            rootfinder_plugin=rootfinder_plugin,
            rootfinder_options=rootfinder_options,
            n_threads=n_threads)
