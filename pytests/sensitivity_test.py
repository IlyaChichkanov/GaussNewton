# -*- coding: utf-8 -*-
"""SensitivityTrajectory: круговой обход плоского контракта.

Плоский layout `[x; S_theta.flatten(); S_c.flatten()]` × точки сетки — это
контракт `SystemJacobian.get_jacobian_solution`, и `unpack` читает его по
осям вручную. Ошибка в порядке осей здесь не падает, а тихо перемешивает
чувствительности по параметрам с чувствительностями по начальному состоянию.

Эталон тут независимый: массив заполняется УНИКАЛЬНЫМИ числами, поэтому
любая перестановка осей меняет результат и ловится.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.sensitivity import SensitivityTrajectory, group_by_grid_length

SHAPES = [(2, 4, 5), (3, 1, 2), (1, 1, 1), (4, 3, 7)]


@pytest.mark.parametrize("nx,n_theta,m", SHAPES)
def test_pack_is_inverse_of_unpack(nx, n_theta, m):
    rows = nx + nx * n_theta + nx * nx
    flat = np.arange(rows * m, dtype=float).reshape(rows, m)

    traj = SensitivityTrajectory.unpack(flat, nx, n_theta)
    assert traj.x.shape == (m, nx)
    assert traj.S_theta.shape == (m, nx, n_theta)
    assert traj.S_c.shape == (m, nx, nx)
    np.testing.assert_array_equal(traj.pack(), flat)


@pytest.mark.parametrize("nx,n_theta,m", SHAPES)
def test_unpack_reads_the_documented_layout(nx, n_theta, m):
    """Блоки лежат в порядке [x; S_theta (C-order); S_c (C-order)]."""
    rows = nx + nx * n_theta + nx * nx
    flat = np.arange(rows * m, dtype=float).reshape(rows, m)
    traj = SensitivityTrajectory.unpack(flat, nx, n_theta)

    for i in range(m):
        np.testing.assert_array_equal(traj.x[i], flat[:nx, i])
        np.testing.assert_array_equal(
            traj.S_theta[i], flat[nx:nx + nx * n_theta, i].reshape(nx, n_theta))
        np.testing.assert_array_equal(
            traj.S_c[i], flat[nx + nx * n_theta:, i].reshape(nx, nx))


def test_head_keeps_first_points():
    nx, n_theta, m = 2, 3, 6
    rows = nx + nx * n_theta + nx * nx
    flat = np.arange(rows * m, dtype=float).reshape(rows, m)
    traj = SensitivityTrajectory.unpack(flat, nx, n_theta)

    head = traj.head(m - 1)          # шут без стыковочной точки
    assert head.n_points == m - 1
    np.testing.assert_array_equal(head.pack(), flat[:, :m - 1])


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError, match="несогласованные формы"):
        SensitivityTrajectory(x=np.zeros((5, 2)),
                              S_theta=np.zeros((5, 3, 4)),   # nx=3 против 2
                              S_c=np.zeros((5, 2, 2)))


def test_group_by_grid_length():
    grids = [np.zeros(10), np.zeros(7), np.zeros(10), np.zeros(7), np.zeros(3)]
    groups = group_by_grid_length(grids)
    assert groups == [[0, 2], [1, 3], [4]]
    # каждый шут попал ровно в одну группу
    assert sorted(i for g in groups for i in g) == list(range(len(grids)))
