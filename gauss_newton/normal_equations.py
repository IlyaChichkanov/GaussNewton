# -*- coding: utf-8 -*-
"""Слой нормальных уравнений: как получить H и g в точке.

Что такое H и g. Линеаризованная подзадача ГН на итерации — это МНК по
приращению delta = [d_theta; d_c_1..d_c_T]; шагу нужны только матрица
нормальных уравнений H = J^T J и градиент g = J^T r, а не сама J.
Документ «MS and Orthogonal Collocations» показывает, что их можно копить
по измерениям (rank-m обновления), и большая J вообще не нужна:

    g_{k+1} = g_k + J_{k+1}^T r_{k+1},      H_{k+1} = H_k + J_{k+1}^T J_{k+1}.

Обозначения — как в документе (в коде массивы, в документе матрицы):

    документ                    здесь                смысл
    -------------------------   ------------------   ------------------------------
    s                           c_j                  начальное состояние (у нас — шута j)
    J^(theta)_i, J^(s)_i        S_theta, S_c         чувствительности x(t_i) по theta и c_j
    J^(s)_{k+1|k}               Psi  (collocation)   переход состояния за элемент
    J^(theta)_{k+1|k}           Gamma(collocation)   вклад параметров за элемент
    h_x, h_theta                dh_dx, dh_dtheta     якобианы наблюдения
    r_i = y_i - h(x_i, theta)   ShootRows.r          невязка (в коде уже взвешена)
    J_i строка = [h_x S_theta_i + h_theta | h_x S_c_i]
                                ShootRows.J_theta,   (см. gauss_newton.problem.ShootRows)
                                ShootRows.J_c
    H^(theta), H^(theta s), H^(s)  H_theta, H_theta_c, H_c   блоки H
    g                           g_theta, g_c         блоки градиента

Отличие от документа — multiple shooting: неизвестных не [theta; s], а
[theta; c_1..c_T], поэтому H «стрелочная»: измерения шута j затрагивают только
столбцы theta и c_j, значит блоки H^(theta s) и H^(s) — свои у каждого шута,
а между разными шутами нули. Невязки стыковки в H не сворачиваются: строки
(J_G, R_G) нужны отдельно — они стоят в седловой системе рядом с блоком -mu I.

Слой оптимизации (шаг и цикл) живёт в gauss_newton/adaptive.py и знает только
класс NormalEquations — как именно получены H и g, ему безразлично.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.sparse import bmat, csr_matrix, vstack, eye as speye
from scipy.sparse.linalg import splu

from gauss_newton.problem import MultipleShooting
from gauss_newton.collocation_shooting import CollocationShooting


@dataclass
class NormalEquations:
    """Нормальные уравнения ГН в точке (без самой J).

    H, g       : H = J^T J, g = J^T r по измерительным невязкам;
    J_G, R_G   : строки непрерывности шутов (в H не свёрнуты, см. модульный docstring);
    rss        : ||r||^2 по измерениям; n_rows: число измерительных строк.
    """
    H: object
    g: np.ndarray
    J_G: object
    R_G: np.ndarray
    rss: float
    n_rows: int

    @classmethod
    def from_jacobian(cls, J, R, J_G, R_G):
        """Путь через большую J: то же самое, но J сначала построена."""
        return cls(H=(J.T @ J).tocsr(), g=J.T @ R, J_G=J_G, R_G=R_G,
                   rss=float(R @ R), n_rows=J.shape[0])

    @property
    def n_cont(self):
        return self.J_G.shape[0]

    def cont_sq(self):
        """||R_G||^2 — квадрат невязки стыковки."""
        return float(self.R_G @ self.R_G) if self.R_G.size else 0.0

    def cost(self):
        """||r||^2 + ||R_G||^2 — полная стоимость (обе группы невязок)."""
        return self.rss + self.cont_sq()

    def merit(self, mu):
        """Phi_mu = ||r||^2 + (1/mu)||R_G||^2 — то, что минимизирует шаг ГН."""
        return self.rss + self.cont_sq() / mu if self.R_G.size else self.rss

    def mu_curvature(self):
        """Стартовое mu: кривизна штрафа сопоставима кривизне измерений.

        ||J_G||_F^2 / ||J||_F^2, где ||J||_F^2 = tr(J^T J) = tr(H) —
        сама J для этого не нужна.
        """
        if not self.R_G.size:
            return 1.0
        return float((self.J_G.multiply(self.J_G)).sum()
                     / max(self.H.diagonal().sum(), 1e-300))

    def covariance_theta(self, n_theta, ridge=1e-8):
        """Маргинальная ковариация theta прямо из H, без построения J.

        J_full = [J; J_G], J_full^T J_full = H + J_G^T J_G; возвращается
        theta-блок обратной матрицы (Шур-комплемент по блоку c).
        """
        A = self.H + (self.J_G.T @ self.J_G) if self.R_G.size else self.H
        n_params = A.shape[0]
        dof = max(self.n_rows + self.n_cont - n_params, 1)
        sigma2 = self.cost() / dof

        rhs = np.zeros((n_params, n_theta))
        rhs[:n_theta, :] = np.eye(n_theta)
        A_reg = (A + ridge * speye(n_params)).tocsc()
        try:
            X = splu(A_reg).solve(rhs)
        except RuntimeError:
            X = np.linalg.pinv(A_reg.toarray()) @ rhs
        return sigma2 * X[:n_theta, :], sigma2, dof


class AccumulateMixin:
    """Собирает H и g накоплением по измерениям — большая J не строится."""

    def normal_equations(self, theta_full):
        n_state, n_theta, _ = self.system.dims()
        H_theta = np.zeros((n_theta, n_theta))   # документ: H^(theta)
        g_theta = np.zeros(n_theta)              # документ: g, theta-часть
        H_theta_c, H_c, g_c = [], [], []         # по одному блоку на шут
        J_G_batches, R_G_batches = [], []
        rss = 0.0
        n_rows = 0

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches,
                    self.t_eval_measurements_batches)):
            rows = self.shoot_rows(theta_full, state_measured, t_meas, batch)

            for shoot in rows:
                J_theta, J_c, r = shoot.J_theta, shoot.J_c, shoot.r
                # Суммы по точкам шута ('m') — векторизованная запись
                # рекурсий H_{k+1} = H_k + J_{k+1}^T J_{k+1}, g_{k+1} = g_k + J_{k+1}^T r_{k+1}
                H_theta += np.einsum('mop,moq->pq', J_theta, J_theta)
                H_theta_c.append(np.einsum('mop,moc->pc', J_theta, J_c))
                H_c.append(np.einsum('moc,mod->cd', J_c, J_c))
                g_theta += np.einsum('mop,mo->p', J_theta, r)
                g_c.append(np.einsum('moc,mo->c', J_c, r))
                rss += float((r * r).sum())
                n_rows += r.size

            J_G, R_G = self.continuity_rows(rows)
            J_G_batches.append(J_G)
            R_G_batches.append(R_G)

        # Стрелочная сборка: theta-строка сверху, по блоку на шут по диагонали
        T = len(H_c)
        blocks = [[csr_matrix(H_theta)] + [csr_matrix(B) for B in H_theta_c]]
        for j in range(T):
            row = [csr_matrix(H_theta_c[j].T)] + [None] * T
            row[1 + j] = csr_matrix(H_c[j])
            blocks.append(row)

        return NormalEquations(
            H=bmat(blocks, format='csr'),
            g=np.concatenate([g_theta] + g_c),
            J_G=self._concatenate_jacobian_batches(J_G_batches),
            R_G=np.concatenate(R_G_batches),
            rss=rss, n_rows=n_rows)


class MultipleShootingAccum(AccumulateMixin, MultipleShooting):
    """Multiple shooting (вариационные уравнения) с накоплением H и g."""


class CollocationShootingAccum(AccumulateMixin, CollocationShooting):
    """Multiple shooting на коллокациях Радо IIA с накоплением H и g."""


def normal_equations_of(problem, theta_full):
    """H и g для любой задачи: накоплением, если умеет, иначе через J."""
    if hasattr(problem, 'normal_equations'):
        return problem.normal_equations(theta_full)
    return NormalEquations.from_jacobian(*problem.solve(theta_full))


def confidence_intervals(theta_opt, cov, dof, alpha=0.05):
    """Двусторонние доверительные интервалы по t-распределению Стьюдента.

    Живёт рядом с ковариацией (covariance_theta), которая их и питает:
    se_i = sqrt(Cov_ii), CI_i = theta_i +- t_{alpha/2, dof} * se_i.
    Вывод — theory_gauss_newton.ipynb, раздел «Доверительные интервалы».
    """
    se = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - alpha / 2, df=dof)
    return theta_opt - t_crit * se, theta_opt + t_crit * se
