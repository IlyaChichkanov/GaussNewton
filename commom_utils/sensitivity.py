"""Named view of a sensitivity trajectory over the flat integrator layout.

The flat layout `[x; S_theta.flatten(); S_c.flatten()]` is the integrator
contract; it exists only between an integrator and `unpack()`.
See docs/architecture.md.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityTrajectory:
    """Solution of the extended system on a grid of m points.

    x       : (m, nx)              state
    S_theta : (m, nx, n_theta)     sensitivity to the parameters
    S_c     : (m, nx, nx)          sensitivity to the shot's initial state
    """
    x: np.ndarray
    S_theta: np.ndarray
    S_c: np.ndarray

    def __post_init__(self):
        m, nx = self.x.shape
        if self.S_theta.shape[:2] != (m, nx) or self.S_c.shape != (m, nx, nx):
            raise ValueError(
                f"inconsistent shapes: x {self.x.shape}, "
                f"S_theta {self.S_theta.shape}, S_c {self.S_c.shape}")

    @property
    def n_points(self):
        return self.x.shape[0]

    @classmethod
    def unpack(cls, flat, nx, n_theta):
        """Read the flat integrator output (nx + nx*n_theta + nx*nx, m)."""
        flat = np.asarray(flat)
        m = flat.shape[1]
        end_theta = nx + nx * n_theta
        return cls(
            x=flat[:nx, :].T,
            S_theta=flat[nx:end_theta, :].reshape(nx, n_theta, m).transpose(2, 0, 1),
            S_c=flat[end_theta:, :].reshape(nx, nx, m).transpose(2, 0, 1),
        )

    def pack(self):
        """Back to the flat layout — the inverse of `unpack`."""
        m = self.x.shape[0]
        return np.concatenate([
            self.x.T,
            self.S_theta.transpose(1, 2, 0).reshape(-1, m),
            self.S_c.transpose(1, 2, 0).reshape(-1, m),
        ])

    def head(self, m):
        """First m points — a shot's measurements without the junction point."""
        return SensitivityTrajectory(self.x[:m], self.S_theta[:m], self.S_c[:m])


def initial_flat_row(c0, n_theta):
    """Initial extended state [c0; S_theta = 0; S_c = I] in the flat layout."""
    c0 = np.asarray(c0, dtype=float)
    nx = c0.shape[0]
    return np.concatenate([c0, np.zeros(nx * n_theta), np.eye(nx).ravel()])


def split_row(y, nx, n_theta):
    """One point of the flat layout -> (x, S_theta, S_c).

    Slices and reshapes only, so it also works on jax arrays under tracing.
    """
    end_theta = nx + nx * n_theta
    return (y[:nx],
            y[nx:end_theta].reshape((nx, n_theta)),
            y[end_theta:end_theta + nx * nx].reshape((nx, nx)))


def group_by_grid_length(t_grids):
    """Shot indices grouped by grid length, ordered by first appearance.

    Both batch integrators need equal shapes within a group: it is a vmap
    precondition for jax and a shared mapaccum function for collocation.
    """
    groups = {}
    for i, grid in enumerate(t_grids):
        groups.setdefault(len(grid), []).append(i)
    return list(groups.values())
