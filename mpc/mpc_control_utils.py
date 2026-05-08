import numpy as np

from abc import ABC, abstractmethod


def reset_solver_initial_guess(solver_):
    """Reset solver to zero initial guess"""
    N = solver_.acados_ocp.dims.N
    nx = solver_.acados_ocp.model.x.shape[0]
    nu = solver_.acados_ocp.model.u.shape[0]
    
    for i in range(N + 1):
        solver_.set(i, "x", np.zeros(nx))
    for i in range(N):
        solver_.set(i, "u", np.zeros(nu))


class Controller(ABC):
    @abstractmethod
    def set_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_control(self, state, t, dt):
        """Возвращает оптимальное управление (np.array)"""
        pass

    def mpc_control(self) -> bool:
        return False

class LateralPIDController(Controller):
    def __init__(self, K_fb=np.array([0.5**2, 0.5]), wheelbase=2.5,
                 yaw_rate_compensation=False):
        self.K_fb = K_fb
        self.wheelbase = wheelbase
        self.yaw_rate_compensation = yaw_rate_compensation
        self.curv = 0.0
        self.v = 0.0

    def set_params(self, params):
        self.v, self.curv = params[0], params[1]

    def compute_control(self, state, t, dt):
        d, psi = state[0], state[1]
        # Форвардный угол: δ_ff = arctan(L * c * cos(ψ) / (1 - c*d))
        # (безопаснее использовать arctan, для малых углов эквивалентно выражению)
        uff = np.arctan(
            self.wheelbase * self.curv * np.cos(psi) / (1 - self.curv * d + 1e-12)
        )
        # Обратная связь по ошибкам
        u_fb = -np.dot(self.K_fb[:2], np.array([d/self.v, psi]))
        # Компенсация угловой скорости (если модель содержит w)
        yaw_rate_fb = 0.0
        if self.yaw_rate_compensation and len(self.K_fb) > 2 and len(state) > 3:
            w = state[3]
            # простая линеаризация: w_ref ≈ curv * v
            yaw_rate_fb = self.K_fb[2] * (w - self.curv * self.v)
        u_opt = u_fb + yaw_rate_fb + uff
        return np.array([u_opt])


# ----------------------------------------------------------------------
# 6. MPC-контроллер
# ----------------------------------------------------------------------
class LateralMPCController(Controller):
    def __init__(self, solver, trajectory):  # траектория теперь обязательна
        self.solver = solver
        self.trajectory = trajectory

    def mpc_control(self) -> bool:
        return True

    def set_params(self, stage, param):
        # Установка параметров для конкретного шага горизонта
        self.solver.set(stage, 'p', param)

    def set_horizon_params(self, t, dt, N, model_param):
        """Заполняет параметры на весь горизонт по траектории."""
        assert self.solver.acados_ocp.model.p.shape[0] == len(model_param) + 2
        horizon_times = t + np.arange(N+1) * dt
        # обрезаем, если вышли за пределы траектории
        max_time = self.trajectory.t_list[-1]  # предполагаем, что есть атрибут
        horizon_times = np.clip(horizon_times, 0, max_time)
        vel = self.trajectory.vel_profile(horizon_times)
        curv = self.trajectory.curv_profile(horizon_times)
        for stage in range(N+1):
            self.set_params(stage, np.array([vel[stage], curv[stage], *model_param]))

    def get_initial_state(self, x_current, t):
        curr_vel = self.trajectory.vel_profile(t)
        d_over_v = x_current[0] / curr_vel
        psi = x_current[1]
        prev_x = self.solver.get(1, 'x')
        tail = prev_x[2:] if len(prev_x) > 2 else np.array([])
        return np.hstack([d_over_v, psi, tail])

    def compute_control(self, state, t, dt):
        x0_solver = self.get_initial_state(state, t)
        self.solver.set(0, "lbx", x0_solver)
        self.solver.set(0, "ubx", x0_solver)
        print(x0_solver)
        status = self.solver.solve()
        if status != 0:
            raise RuntimeError(f"MPC solver failed with status {status}")
        u_opt = np.array(self.solver.get(1, 'x'))[2:3]
        return u_opt
    



