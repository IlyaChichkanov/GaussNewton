
from abc import ABC, abstractmethod
import numpy as np
from scipy import interpolate
from commom_utils.ode_system import ODESystem, SystemItegrator
from mpc.mpc_control_utils import Controller
import matplotlib.pyplot as plt


class TrajectoryTimeProfile:
    def __init__(self, f_vel, f_curv, t_list=None):
        self.vel_profile = f_vel
        self.curv_profile = f_curv
        self.t_list = t_list if t_list is not None else f_vel.x

class Scenario(ABC):
    @abstractmethod
    def create_trajectory(self, t_list: np.ndarray) -> TrajectoryTimeProfile:
        """
        Принимает временной массив, возвращает готовый профиль траектории.
        """
        pass


class ConstantCurveScenario(Scenario):
    """Постоянная кривизна на всей длительности."""
    def __init__(self, velocity=10.0, curvature=0.01):
        self.v = velocity
        self.c = curvature

    def create_trajectory(self, t_list):
        vel_arr = np.full_like(t_list, self.v)
        curv_arr = np.full_like(t_list, self.c)
        f_vel = interpolate.interp1d(t_list, vel_arr, kind='linear')
        f_curv = interpolate.interp1d(t_list, curv_arr, kind='linear')
        return TrajectoryTimeProfile(f_vel, f_curv, t_list)


class SinusoidalCurveScenario(Scenario):
    """Синусоидальная кривизна с заданной амплитудой и частотой."""
    def __init__(self, velocity=10.0, curv_amplitude=0.02, frequency=0.5):
        self.v = velocity
        self.A = curv_amplitude
        self.freq = frequency

    def create_trajectory(self, t_list):
        vel_arr = np.full_like(t_list, self.v)
        curv_arr = self.A * np.sin(2 * np.pi * self.freq * t_list)
        f_vel = interpolate.interp1d(t_list, vel_arr, kind='linear')
        f_curv = interpolate.interp1d(t_list, curv_arr, kind='linear')
        return TrajectoryTimeProfile(f_vel, f_curv, t_list)


class DoubleLaneChangeScenario(Scenario):
    """
    Классическое перестроение: два коротких импульса кривизны разного знака.
    Форма задаётся суммой двух `tanh`-функций.
    """
    def __init__(self, velocity=10.0, curv_max=0.02, t_center=5.0, duration=2.0):
        self.v = velocity
        self.c_max = curv_max
        self.t0 = t_center
        self.T = duration

    def create_trajectory(self, t_list):
        vel_arr = np.full_like(t_list, self.v)
        # переход из одной полосы в другую и обратно
        t_rel = (t_list - self.t0) / self.T
        curv_arr = self.c_max * (np.tanh(t_rel + 1) - np.tanh(t_rel - 1)) / 2
        f_vel = interpolate.interp1d(t_list, vel_arr, kind='linear')
        f_curv = interpolate.interp1d(t_list, curv_arr, kind='linear')
        return TrajectoryTimeProfile(f_vel, f_curv, t_list)


class CustomSequenceScenario(Scenario):
    """
    Позволяет передать произвольные массивы скорости и кривизны (аналог вашего текущего подхода).
    """
    def __init__(self, velocity_array, curvature_array):
        self.vel = np.asarray(velocity_array)
        self.curv = np.asarray(curvature_array)
        assert len(self.vel) == len(self.curv)

    def create_trajectory(self, t_list):
        if len(t_list) != len(self.vel):
            raise ValueError("t_list must match length of provided arrays")
        f_vel = interpolate.interp1d(t_list, self.vel, kind='linear')
        f_curv = interpolate.interp1d(t_list, self.curv, kind='linear')
        return TrajectoryTimeProfile(f_vel, f_curv, t_list)


class VariableSpeedScenario(Scenario):
    """
    Декорирует любой другой сценарий, изменяя его профиль скорости.
    Полезно для тестирования на разных скоростях без дублирования кода.
    """
    def __init__(self, base_scenario: Scenario, speed_profile_func):
        self.base = base_scenario
        self.speed_func = speed_profile_func

    def create_trajectory(self, t_list):
        base_traj = self.base.create_trajectory(t_list)
        # Переопределяем скорость, сохраняя кривизну
        v_new = self.speed_func(t_list)
        f_vel = interpolate.interp1d(t_list, v_new, kind='linear')
        return TrajectoryTimeProfile(f_vel, base_traj.curv_profile, t_list)
    




class Simulator:
    def __init__(self, model: ODESystem, controller: Controller,
                 trajectory, delay_cycles=0, use_jax=True):
        self.integrator = SystemItegrator(model)
        self.controller = controller
        self.trajectory = trajectory
        self.delay_cycles = delay_cycles
        self.use_jax = use_jax
        self.nx = model.nx

    def run(self, t_list, x0=None):
        """
        Запуск симуляции.
        t_list: массив времён (N_sim+1 точек, включая начальное время)
        x0: начальное состояние (если None, то [0.1, 0.02])
        Возвращает states (N_sim+1 x nx), controls (N_sim x nu)
        """
        if x0 is None:
            x0 = np.zeros(self.nx)


        N_sim = len(t_list) - 1
        states = np.zeros((N_sim + 1, self.nx))
        controls = np.zeros((N_sim, 1))  # предполагаем nu=1
        states[0] = x0
        x_current = x0.copy()

        # Буфер для учёта задержки управления
        u_history = [np.zeros(1)] * max(self.delay_cycles, 1)

        # Если MPC, нужно предварительно настроить горизонт
        if self.controller.mpc_control():
            N = self.controller.solver.acados_ocp.dims.N
    
        for i in range(N_sim):
            t = t_list[i]
            dt = t_list[i+1] - t  # фактический шаг

            # Параметры траектории в текущий момент
            curr_vel = self.trajectory.vel_profile(t)
            curr_curv = self.trajectory.curv_profile(t)
            params = np.hstack([curr_vel, curr_curv])

            # Если MPC – обновляем параметры на всём горизонте
            if self.controller.mpc_control():
                model_params = np.array([0.2, 0])  #delay offset
                self.controller.set_horizon_params(t, dt, N, model_params)
            else:
                self.controller.set_params(params)

            # Получение управления
            u_opt = self.controller.compute_control(x_current, t, dt)
            # Задержка: берём управление из истории
            if self.delay_cycles > 0:
                u_applied = u_history[-self.delay_cycles]
                u_history.append(u_opt.copy())
                u_history.pop(0)
            else:
                u_applied = u_opt

            # Интегрирование
            if self.use_jax:
                x_current = self.integrator.step_jax(x_current, u_applied, params, dt)
            else:
                x_current = self.integrator.step(x_current, u_applied, params, dt)

            # Сохранение
            controls[i] = u_opt
            states[i+1] = x_current

        return states, controls



class SimulationPlotter:
    def __init__(self, t_list, states, controls, trajectory=None, model=None):
        """
        t_list : array (N+1,) — время для состояний
        states : array (N+1, nx)
        controls : array (N, nu) — управление (первый канал = rwa)
        trajectory : TrajectoryTimeProfile (должен иметь vel_profile)
        model : ODESystem (например, KinematicBicycleErrors) — нужен wheelbase
        """
        self.t_full = np.asarray(t_list)
        self.states = np.asarray(states)
        self.controls = np.asarray(controls)
        self.trajectory = trajectory
        self.model = model

        # Время для управления: если длины совпадают, управление дискретизировано так же,
        # иначе стандартно — на один шаг меньше.
        if len(self.t_full) == len(self.controls):
            self.t_control = self.t_full
        else:
            self.t_control = self.t_full[:-1]

        # Извлекаем wheelbase
        if self.model is not None:
            self.wheelbase = getattr(self.model, 'wheelbase', None)
        else:
            self.wheelbase = None

    def get_velocity_at_controls(self):
        """Скорость на моментах выдачи управления."""
        if self.trajectory is None:
            raise ValueError("Trajectory required to compute velocity")
        return self.trajectory.vel_profile(self.t_control)

    def compute_jerk(self):
        """Jerk = v^2 * (d(rwa)/dt) / wheelbase"""
        if self.wheelbase is None:
            raise ValueError("Model with wheelbase required for jerk computation")
        v = self.get_velocity_at_controls()
        # производная управления по времени
        drwa_dt = np.gradient(self.controls[:, 0], self.t_control)
        jerk = v**2 * drwa_dt / self.wheelbase
        return jerk

    def compute_comfort_acceleration(self):
        """a_comf = v^2 * rwa / wheelbase"""
        if self.wheelbase is None:
            raise ValueError("Model with wheelbase required for comfort acceleration")
        v = self.get_velocity_at_controls()
        rwa = self.controls[:, 0]
        a_comf = v**2 * rwa / self.wheelbase
        return a_comf

    def plot_states(self, axs=None):
        if axs is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        else:
            ax1, ax2 = axs
        ax1.plot(self.t_full, self.states[:, 0], 'b-', linewidth=2, label='d')
        ax1.set_ylabel('Lateral deviation [m]')
        ax1.grid(True)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax2.plot(self.t_full, self.states[:, 1], 'g-', linewidth=2, label='psi')
        ax2.set_ylabel('Heading angle [rad]')
        ax2.grid(True)
        ax2.axhline(y=0, color='r', linestyle='--')
        return ax1, ax2

    def plot_control(self, ax=None, control_idx=0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(self.t_control, self.controls[:, control_idx], 'k', linewidth=2, label='steering (rwa)')
        ax.set_ylabel('Steering angle [rad]')
        ax.grid(True)
        ax.axhline(y=0, color='r', linestyle='--')
        return ax

    def plot_jerk(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        jerk = self.compute_jerk()
        ax.plot(self.t_control, jerk, 'm-', linewidth=2, label='jerk')
        ax.set_ylabel('Jerk [m/s^3]') 
        ax.grid(True)
        ax.axhline(y=0, color='r', linestyle='--')
        return ax

    def plot_comfort_acceleration(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        a_comf = self.compute_comfort_acceleration()
        ax.plot(self.t_control, a_comf, 'c-', linewidth=2, label='comfort acc')
        ax.set_ylabel('Comfort acceleration [m/s²]')
        ax.grid(True)
        ax.axhline(y=0, color='r', linestyle='--')
        return ax

    def plot_all(self, include_jerk=False, include_comfort=False):
        n_extra = int(include_jerk) + int(include_comfort)
        n_rows = 2 + 1 + n_extra  # states (2), control (1)
        fig, axs = plt.subplots(n_rows, 1, figsize=(15, 3*n_rows), sharex=True)

        self.plot_states(axs=(axs[0], axs[1]))
        self.plot_control(ax=axs[2])
        idx = 3
        if include_jerk:
            self.plot_jerk(ax=axs[idx])
            idx += 1
        if include_comfort:
            self.plot_comfort_acceleration(ax=axs[idx])
            idx += 1

        axs[-1].set_xlabel('Time [s]')
        for ax in axs:
            ax.legend(loc='best')
        plt.tight_layout()
        return fig, axs
    
