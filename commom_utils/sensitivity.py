# -*- coding: utf-8 -*-
"""Чувствительности траектории: именованный вид вместо плоского вектора.

Интеграторы вынуждены отдавать результат ПЛОСКИМ: и scipy solve_ivp, и jax
odeint умеют интегрировать только вектор, поэтому расширенное состояние
укладывается в строку

    [x (nx); S_theta.flatten() (nx*n_theta); S_c.flatten() (nx*nx)]

по одному столбцу на точку сетки. Это контракт интеграторов
(get_jacobian_solution), и он не меняется.

Проблема была в том, что этот layout протекал НАРУЖУ: каждый потребитель
распаковывал его вручную через слайсы и moveaxis, размерности жили в
комментариях, а не в коде. Ровно там и завёлся баг с лишней осью df_dx_jax.

SensitivityTrajectory — граница: плоский вид существует только между
интегратором и unpack(), дальше идут именованные массивы с проверяемыми
формами.

Обозначения — из theory_gauss_newton.ipynb:
    S_theta(t) = dx(t)/dtheta   (nx, n_theta), S_theta(t_0) = 0
    S_c(t)     = dx(t)/dc_j     (nx, nx),      S_c(t_0)     = I
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityTrajectory:
    """Решение расширенной системы на сетке из m точек.

    x       : (m, nx)              состояние
    S_theta : (m, nx, n_theta)     чувствительность по параметрам
    S_c     : (m, nx, nx)          чувствительность по начальному состоянию шута
    """
    x: np.ndarray
    S_theta: np.ndarray
    S_c: np.ndarray

    def __post_init__(self):
        m, nx = self.x.shape
        if self.S_theta.shape[:2] != (m, nx) or self.S_c.shape != (m, nx, nx):
            raise ValueError(
                f"несогласованные формы: x {self.x.shape}, "
                f"S_theta {self.S_theta.shape}, S_c {self.S_c.shape}")

    @property
    def n_points(self):
        return self.x.shape[0]

    @classmethod
    def unpack(cls, flat, nx, n_theta):
        """Разбор плоского выхода интегратора (nx + nx*n_theta + nx*nx, m)."""
        flat = np.asarray(flat)
        m = flat.shape[1]
        end_theta = nx + nx * n_theta
        return cls(
            x=flat[:nx, :].T,
            S_theta=flat[nx:end_theta, :].reshape(nx, n_theta, m).transpose(2, 0, 1),
            S_c=flat[end_theta:, :].reshape(nx, nx, m).transpose(2, 0, 1),
        )

    def pack(self):
        """Обратно в плоский контракт интеграторов — обратная к unpack.

        Круговой обход проверяется в pytests/sensitivity_test.py: именно он
        фиксирует, что порядок осей в unpack прочитан верно.
        """
        m = self.x.shape[0]
        return np.concatenate([
            self.x.T,
            self.S_theta.transpose(1, 2, 0).reshape(-1, m),
            self.S_c.transpose(1, 2, 0).reshape(-1, m),
        ])

    def head(self, m):
        """Первые m точек — измерения интервала без стыковочной точки."""
        return SensitivityTrajectory(self.x[:m], self.S_theta[:m], self.S_c[:m])


def initial_flat_row(c0, n_theta):
    """Начальное расширенное состояние в плоском layout.

    [c0; S_theta(t0) = 0; S_c(t0) = I] — начальные условия вариационных
    уравнений и рекурсий коллокаций. Раньше эта строка была написана в
    четырёх местах (ode_system x2, collocation x2), каждый раз заново
    вспоминая порядок блоков.
    """
    c0 = np.asarray(c0, dtype=float)
    nx = c0.shape[0]
    return np.concatenate([c0, np.zeros(nx * n_theta), np.eye(nx).ravel()])


def split_row(y, nx, n_theta):
    """Одна точка плоского layout -> (x, S_theta, S_c) — точечный unpack.

    Только срезы и reshape, поэтому работает и с numpy-, и с jax-массивами
    (правые части вариационных уравнений зовут её под трассировкой).
    """
    end_theta = nx + nx * n_theta
    return (y[:nx],
            y[nx:end_theta].reshape((nx, n_theta)),
            y[end_theta:end_theta + nx * nx].reshape((nx, nx)))



def group_by_grid_length(t_grids):
    """Индексы шутов, сгруппированные по длине временной сетки.

    Оба батчевых интегратора требуют одинаковых форм внутри группы: у jax
    это условие vmap, у коллокаций — общая mapaccum-функция на n_elems.
    Раньше эта группировка была написана дважды, в ode_system и в collocation.

    Возвращает список списков индексов (порядок групп — по первому вхождению).
    """
    groups = {}
    for i, grid in enumerate(t_grids):
        groups.setdefault(len(grid), []).append(i)
    return list(groups.values())
