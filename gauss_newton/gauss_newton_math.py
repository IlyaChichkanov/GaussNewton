import numpy as np
from scipy.integrate import solve_ivp
from commom_utils.ode_system import ODESystem, SystemJacobian
import casadi as ca
ATOL = 1e-5
RTOL = 1e-5



class TimeIntervalManager:
    def __init__(self, N_shoot, t_eval_measurements):
        self.t_eval_measurements = t_eval_measurements
        N_measurement = len(t_eval_measurements)
        self.measurement_indexes = np.arange(N_measurement)
        shoot_indexes = self.measurement_indexes[0:-1:int(len(self.measurement_indexes)/N_shoot)]
        self.shoot_indexes = np.append(shoot_indexes, self.measurement_indexes[-1])
        self.N_shoot = len(self.shoot_indexes) - 1

    def get_time_interval(self, shoot):
        meas_idx = self.measurement_indexes[
            (self.measurement_indexes >= self.shoot_indexes[shoot]) &
            (self.measurement_indexes < self.shoot_indexes[shoot + 1])
        ]
        t_interval = self.t_eval_measurements[meas_idx]
        t_interval = np.append(t_interval, self.t_eval_measurements[meas_idx[-1] + 1])
        return t_interval, meas_idx
    



class MultipleShooting:
    """
    Multiple shooting implementation for parameter identification.
    """

    def __init__(self, system: ODESystem, N_shoot: int, gamma: np.ndarray = np.nan,
                 c0_cost: float = 1, use_jax: bool = False):
        self.system = SystemJacobian(system)
        self.N_shoot = N_shoot
        self.gamma = gamma
        self.c0_cost = c0_cost
        self.use_jax = use_jax

        # Data storage
        self.state_measured_batches = []
        self.state_full_batches = []
        self.t_eval_measurements_batches = []

        # Optional: store full trajectory for debugging
        self.full_trajectory = None

    def add_batch(self, state_full, state_measured, t_eval_measurements):
        """Add a new batch of experimental data."""
        self.state_measured_batches.append(state_measured)
        self.state_full_batches.append(state_full)
        self.t_eval_measurements_batches.append(t_eval_measurements)

    def get_time_interval(self, shoot, batch):
        """Get time interval and measurement indices for a given shoot and batch."""
        tm = TimeIntervalManager(self.N_shoot, self.t_eval_measurements_batches[batch])
        return tm.get_time_interval(shoot)

    def make_full_theta(self, theta0):
        """
        Build the full parameter vector (theta + initial conditions for all shoots).
        """
        theta_full = np.copy(theta0)
        for state_meas, state_full, t_meas in zip(
                self.state_measured_batches, self.state_full_batches, self.t_eval_measurements_batches):
            n_meas = len(t_meas)
            meas_idx = np.arange(n_meas, dtype=int)
            shoot_idx = meas_idx[0:-1:int(len(meas_idx) / self.N_shoot)]
            shoot_idx = np.append(shoot_idx, meas_idx[-1])
            for i in range(len(shoot_idx) - 1):
                id_ = shoot_idx[i]
                c0_ = state_full[id_]   # initial guess from the provided full state
                theta_full = np.concatenate((theta_full, c0_))
        return theta_full

    def _concatenate_jacobians(self, J1, J2):
        """
        Concatenate two jacobian matrices in the block structure required by multiple shooting.
        """
        n_state, n_theta, _ = self.system.get_dimentions()
        J2_theta = J2[:, :n_theta]
        J2_c0 = J2[:, n_theta:]
        zeros1 = np.zeros((J1.shape[0], J2_c0.shape[1]))
        zeros2 = np.zeros((J2_theta.shape[0], J1.shape[1] - n_theta))
        return np.block([[J1, zeros1], [J2_theta, zeros2, J2_c0]])

    def solve(self, theta_full):
        """
        Assemble the full Jacobians and residuals for all batches.
        Returns (J_meas, R_meas, J_cont, R_cont).
        """
        J_total = None
        J_G_total = None
        R_total = None
        R_G_total = None

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches, self.t_eval_measurements_batches)):
            print(f"Solve batch {batch}")
            J_batch, J_G_batch, R_batch, R_G_batch = self._solve_batch(
                theta_full, state_measured, t_meas, batch
            )
            if J_total is None:
                J_total, J_G_total, R_total, R_G_total = J_batch, J_G_batch, R_batch, R_G_batch
            else:
                J_total = self._concatenate_jacobians(J_total, J_batch)
                J_G_total = self._concatenate_jacobians(J_G_total, J_G_batch)
                R_total = np.hstack((R_total, R_batch))
                R_G_total = np.hstack((R_G_total, R_G_batch))

        return J_total, R_total, J_G_total, R_G_total

    def _solve_batch(self, theta_full, state_measured, t_meas, batch_idx):
        """
        Process one experimental batch.
        Returns (J_meas, J_cont, R_meas, R_cont) for this batch.
        """
        n_state, n_theta, n_meas = self.system.get_dimentions()

        # Indices within the sensitivity solution
        idx_theta = slice(0, n_state * n_theta)
        idx_c = slice(n_state * n_theta, n_state * (n_theta + n_state))

        n_measurements = state_measured.shape[0]
        tm = TimeIntervalManager(self.N_shoot, t_meas)
        n_shoot = tm.N_shoot

        # Preallocate arrays
        J = np.zeros((n_measurements, n_meas, n_theta + n_shoot * n_state))
        J_G = np.zeros((n_shoot - 1, n_state, n_theta + n_shoot * n_state))
        R = np.zeros((n_measurements, n_meas))
        R_G = np.zeros((n_shoot - 1, n_state))

        meas_row = 0          # current row index for measurements
        cont_row = 0          # current row index for continuity constraints
        Jx_prev = None
        Jc_prev = None
        state_prev = None

        for shoot in range(n_shoot):
            # Extract initial condition for this shoot
            start_idx = n_theta + n_shoot * batch_idx * n_state + shoot * n_state
            c0 = theta_full[start_idx: start_idx + n_state]

            t_interval, meas_idx = tm.get_time_interval(shoot)
            if self.use_jax:
                sol = self.system.get_jacobian_solution_jax(c0, theta_full[:n_theta], t_interval)
            else:
                sol = self.system.get_jacobian_solution(c0, theta_full[:n_theta], t_interval)

            # Extract state and sensitivity parts
            state_traj = sol[:n_state, :]          # (n_state, n_time)
            J_raw = sol[n_state:, :]               # (n_state*(n_theta+n_state), n_time)

            # Process each measurement point in this interval
            for i, idx in enumerate(meas_idx):
                state = state_traj[:, i]
                t = t_interval[i]

                dh_dx = self.system.dh_dx(state, t, theta_full[:n_theta])
                dh_dtheta = self.system.dh_dtheta(state, t, theta_full[:n_theta])

                dx_dtheta = J_raw[idx_theta, i].reshape(n_state, n_theta)
                dx_dc = J_raw[idx_c, i].reshape(n_state, n_state)

                J_theta = dh_dx @ dx_dtheta + dh_dtheta
                J_c = dh_dx @ dx_dc

                # Normalize by number of points in this interval
                norm = len(meas_idx)
                J[meas_row, :, :n_theta] = J_theta / norm
                J[meas_row, :, n_theta + shoot * n_state : n_theta + (shoot + 1) * n_state] = J_c / norm
                y_meas = state_measured[idx]
                y_pred = self.system.h_x(state, t, theta_full[:n_theta])
                R[meas_row] = (y_meas - y_pred) / norm

                # Apply weights if gamma is provided
                if np.any(~np.isnan(self.gamma)):
                    R[meas_row] *= self.gamma
                    J[meas_row] *= self.gamma[:, np.newaxis]
                    if i == 0:   # first point in interval gets extra weight
                        R[meas_row] *= self.c0_cost
                        J[meas_row] *= self.c0_cost

                meas_row += 1

            # Continuity constraints (skip for first shoot)
            if shoot > 0:
                J_G[cont_row, :, :n_theta] = Jx_prev
                J_G[cont_row, :, n_theta + (shoot - 1) * n_state : n_theta + shoot * n_state] = Jc_prev
                J_G[cont_row, :, n_theta + shoot * n_state : n_theta + (shoot + 1) * n_state] = -np.eye(n_state)
                R_G[cont_row] = -(state_prev - c0)
                # Note: original code also applied gamma to R_G/J_G in a strange way (used R[ind], J[ind] with ind out of range).
                # This part is omitted because it was likely a bug. We keep original logic only for measurements.
                # If needed, we can add similar weight application for continuity, but original did not.
                cont_row += 1

            # Store data for next shoot
            Jx_prev = J_raw[idx_theta, -1].reshape(n_state, n_theta)
            Jc_prev = J_raw[idx_c, -1].reshape(n_state, n_state)
            state_prev = state_traj[:, -1]

        # Flatten the arrays
        J_flat = J.reshape(n_measurements * n_meas, -1)
        R_flat = R.reshape(n_measurements * n_meas)
        J_G_flat = J_G.reshape((n_shoot - 1) * n_state, -1)
        R_G_flat = R_G.reshape((n_shoot - 1) * n_state)

        return J_flat, J_G_flat, R_flat, R_G_flat

    # Keep original method name for backward compatibility
    def concantenate_jacobian(self, J1, J2):
        """Alias for _concatenate_jacobians (maintains original method name)."""
        return self._concatenate_jacobians(J1, J2)