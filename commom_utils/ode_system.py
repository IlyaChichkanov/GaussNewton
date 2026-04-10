import numpy as np
import jax
from jax.experimental.ode import odeint
from jax import numpy as jnp
from scipy.integrate import solve_ivp
from casadi import SX, vertcat, Function, jacobian, vertcat
import casadi as ca
from abc import ABC, abstractmethod
from jaxadi import convert



class ODESystem:
    def __init__(self, nx, np, nu):
        self.nx = nx
        self.np = np
        self.nu = nu
        self.state = ca.SX.sym("x", nx)
        self.theta = ca.SX.sym("theta", np)
        self.u = ca.SX.sym("u", nu)
        self.n_obs = self.observation(self.state, self.theta, self.u).shape[0]

    @abstractmethod
    def get_derivative(self, state: SX, theta: SX, u: SX) -> SX:
        pass

    def get_system(self):
        f = self.get_derivative(self.state, self.theta, self.u)
        return self.state, self.theta, self.u, f
    
    def observation(self, state: SX, theta: SX, u: SX):
        return state
    
    def get_input_signals(self, t):
        return []


class SystemJacobian:
    """
    Класс для вычисления правых частей, якобианов и интегрирования системы.
    Поддерживает как обычный режим (NumPy + SciPy), так и JAX-режим (jaxadi.convert).
    """

    def __init__(self, f_sym: ODESystem, method: str = 'RK45'):
        """
        f_sym: объект System, предоставляющий символьное описание модели.
        method: метод интегрирования для solve_ivp (например, 'RK45').
        """
        self.ATOL = 1e-5
        self.RTOL = 1e-5
        self.f_sym = f_sym
        self.method = method

        # Получаем символьные переменные и выражения
        state_var, theta_var, inp_signal_var, f = self.f_sym.get_system()
        h_observ = self.f_sym.observation(state_var, theta_var, inp_signal_var)

        # Списки элементов для CasADi функций
        state_list = state_var.elements()
        inp_list = inp_signal_var.elements()
        theta_list = theta_var.elements()

        self.inp_signal_len = len(inp_list)
        self.nx = len(state_list)
        self.n_p = len(theta_list)
        self.n_obs = len(h_observ.elements())

        # --- Создание CasADi функций ---
        self.res_f = Function('func', [*state_list, *inp_list, *theta_list], [f])
        self.res_f_jax = convert(self.res_f, compile=True)

        self.res_h = Function('h_x', [*state_list, *inp_list, *theta_list], [h_observ])

        # Якобианы
        J_h_x = jacobian(h_observ, vertcat(*state_list))
        self.compute_jacobian_h_x = Function('J_h_x', [*state_list, *inp_list, *theta_list], [J_h_x])

        J_h_theta = jacobian(h_observ, vertcat(*theta_list))
        self.compute_jacobian_h_theta = Function('J_h_theta', [*state_list, *inp_list, *theta_list], [J_h_theta])

        J_p = jacobian(f, vertcat(*theta_list))
        self.compute_jacobian_theta = Function('J_p', [*state_list, *inp_list, *theta_list], [J_p])
        self.compute_jacobian_theta_jax = convert(Function('J_p', [*state_list, *inp_list, *theta_list], [J_p]), compile=True)

        J_x = jacobian(f, vertcat(*state_list))
        self.compute_jacobian_x = Function('J_x', [*state_list, *inp_list, *theta_list], [J_x])
        self.compute_jacobian_x_jax = convert(Function('J_x', [*state_list, *inp_list, *theta_list], [J_x]), compile=True)

        # Константы для индексации расширенного состояния (чувствительности)
        self._IDX_JX = slice(self.nx, self.nx + self.nx * self.n_p)
        self._IDX_JC = slice(self._IDX_JX.stop, self._IDX_JX.stop + self.nx * self.nx)

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    def _get_inp_signals(self, t):
        """Возвращает массив входных сигналов в момент t."""
        try:
            return self.f_sym.get_input_signals(t)
        except Exception as e:
            print(f'Ошибка получения входов при t={t}: {e}')
            return np.zeros(self.inp_signal_len)

    def get_dimentions(self):
        """Возвращает размерности: (state_len, theta_len, meas_len)."""
        return self.nx, self.n_p, self.n_obs

    # ----------------------------------------------------------------------
    # Методы для обычного режима (NumPy)
    # ----------------------------------------------------------------------
    def h_x(self, state, t, theta):
        """Вычисляет выход системы в момент t."""
        inp = self._get_inp_signals(t)
        return np.array(self.res_h(*state, *inp, *theta)).flatten()

    def inverse_h(self, y, t, theta, x_guess=None, n_iter=1):
        """
        Приближённо решает уравнение h(x, theta) = y относительно x.
        Параметры:
            y: измерение (meas_len,)
            t: время
            theta: параметры (theta_len,)
            x_guess: начальное приближение (state_len,). Если None, то нули.
            n_iter: число итераций Гаусса–Ньютона (обычно 1-2).
        Возвращает:
            x: оценка состояния (state_len,)
        """
        if x_guess is None:
            x_guess = np.zeros(self.nx)
        
        x = x_guess.copy()
        for _ in range(n_iter):
            dh_dx = self.dh_dx(x, t, theta)        # (meas_len, state_len)
            h_val = self.h_x(x, t, theta)          # (meas_len,)
            residual = y - h_val
            # Решаем линейную систему (least squares)
            delta_x = np.linalg.lstsq(dh_dx, residual, rcond=None)[0]
            x = x + delta_x
        return x

    def f_x_theta(self, state, t, theta):
        """Вычисляет правую часть системы в момент t."""
        inp = self._get_inp_signals(t)
        return np.array(self.res_f(*state, *inp, *theta)).flatten()

    def dh_dx(self, state, t, theta):
        """Якобиан выхода по состоянию."""
        inp = self._get_inp_signals(t)
        return np.array(self.compute_jacobian_h_x(*state, *inp, *theta))

    def dh_dtheta(self, state, t, theta):
        """Якобиан выхода по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self.compute_jacobian_h_theta(*state, *inp, *theta)).squeeze()

    def df_dtheta(self, state, t, theta):
        """Якобиан правой части по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self.compute_jacobian_theta(*state, *inp, *theta))

    def df_dx(self, state, t, theta):
        """Якобиан правой части по состоянию."""
        inp = self._get_inp_signals(t)
        return np.array(self.compute_jacobian_x(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # JAX-методы (используют скомпилированные функции из jaxadi)
    # ----------------------------------------------------------------------
    def f_x_theta_jax(self, y, t, *theta):
        """JAX-совместимая правая часть."""
        inp = self._get_inp_signals(t)
        return jnp.array(self.res_f_jax(*y, *inp, *theta)[0].flatten())

    def df_dtheta_jax(self, state, t, theta):
        """JAX-совместимый якобиан по параметрам."""
        inp = self._get_inp_signals(t)
        return jnp.array(self.compute_jacobian_theta_jax(*state, *inp, *theta))[0]

    def df_dx_jax(self, state, t, theta):
        """JAX-совместимый якобиан по состоянию."""
        inp = self._get_inp_signals(t)
        return jnp.array(self.compute_jacobian_x_jax(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # Интегрирование
    # ----------------------------------------------------------------------
    def get_solution(self, c0, theta, t_eval):
        """Интегрирование только состояния (обычный режим)."""
        def system(t, y):
            return self.f_x_theta(y, t, theta[:self.n_p])

        sol = solve_ivp(system, (t_eval[0], t_eval[-1]), c0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Интегрирование не сошлось: {sol.message}")
        return sol.y

    def get_jacobian_solution(self, c0, theta, t_eval):
        """Интегрирование расширенной системы (состояние + чувствительности) (обычный режим)."""
        n = self.nx
        p = self.n_p

        J0 = np.concatenate([np.zeros((n, p)).flatten(), np.eye(n).flatten()])
        y0 = np.concatenate([c0, J0])

        def full_ode(t, y):
            x = y[:n]
            Jx = y[self._IDX_JX].reshape((n, p))
            Jc = y[self._IDX_JC].reshape((n, n))

            dx = self.f_x_theta(x, t, theta[:p])
            dJx = self.df_dx(x, t, theta[:p]) @ Jx + self.df_dtheta(x, t, theta[:p])
            dJc = self.df_dx(x, t, theta[:p]) @ Jc

            return np.concatenate([dx, dJx.flatten(), dJc.flatten()])

        sol = solve_ivp(full_ode, (t_eval[0], t_eval[-1]), y0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Интегрирование чувствительности не сошлось: {sol.message}")
        return sol.y

    # ----------------------------------------------------------------------
    # JAX-интегрирование (через odeint)
    # ----------------------------------------------------------------------
    def get_solution_jax(self, c0, theta, t_eval):
        """JAX-интегрирование только состояния."""
        sol = odeint(self.f_x_theta_jax,
                     jnp.array(c0),
                     jnp.array(t_eval),
                     *theta[:self.n_p])
        return np.array(sol).T

    def get_jacobian_solution_jax(self, c0, theta, t_eval):
        """JAX-интегрирование расширенной системы (состояние + чувствительности)."""
        n = self.nx
        p = self.n_p

        J0 = jnp.concatenate([jnp.zeros((n, p)).flatten(), jnp.eye(n).flatten()])
        y0 = jnp.concatenate([jnp.array(c0), J0])

        sol = odeint(self.make_full_system_jax, y0, jnp.array(t_eval), *theta[:p])
        return np.array(sol).T

    # ----------------------------------------------------------------------
    # Вспомогательные методы для расширенной системы (обычный и JAX)
    # ----------------------------------------------------------------------
    def _jacobian_x(self, state, t, theta):
        """Вычисляет производную матрицы Jx (чувствительности по параметрам)."""
        x = state[:self.nx]
        Jx = state[self._IDX_JX].reshape((self.nx, self.n_p))
        dJx = self.df_dx(x, t, theta) @ Jx + self.df_dtheta(x, t, theta)
        return dJx.flatten()

    def _jacobian_c(self, state, t, theta):
        """Вычисляет производную матрицы Jc (чувствительности по начальным условиям)."""
        x = state[:self.nx]
        Jc = state[self._IDX_JC].reshape((self.nx, self.nx))
        dJc = self.df_dx(x, t, theta) @ Jc
        return dJc.flatten()

    def make_full_system(self, state, t, theta):
        """Расширенная система для обычного режима."""
        x = state[:self.nx]
        dx = self.f_x_theta(x, t, theta)
        dJx = self._jacobian_x(state, t, theta)
        dJc = self._jacobian_c(state, t, theta)
        return np.concatenate([dx, dJx, dJc])

    def _jacobian_x_jax(self, state, t, theta):
        """JAX-версия производной Jx."""
        x = state[:self.nx]
        Jx = state[self._IDX_JX].reshape((self.nx, self.n_p))
        dJx = self.df_dx_jax(x, t, theta) @ Jx + self.df_dtheta_jax(x, t, theta)
        return dJx.flatten()

    def _jacobian_c_jax(self, state, t, theta):
        """JAX-версия производной Jc."""
        x = state[:self.nx]
        Jc = state[self._IDX_JC].reshape((self.nx, self.nx))
        dJc = self.df_dx_jax(x, t, theta) @ Jc
        return dJc.flatten()

    def make_full_system_jax(self, state, t, *theta):
        """JAX-версия расширенной системы."""
        x = state[:self.nx]
        dx = self.f_x_theta_jax(x, t, *theta)
        dJx = self._jacobian_x_jax(state, t, theta)
        dJc = self._jacobian_c_jax(state, t, theta)
        return jnp.concatenate([dx, dJx, dJc])

    # ----------------------------------------------------------------------
    # Обратная совместимость (старые названия методов, если они использовались)
    # ----------------------------------------------------------------------
    def JacobianX(self, state, t, theta):
        """Старое название для _jacobian_x. Оставлено для совместимости."""
        return self._jacobian_x(state, t, theta)

    def JacobianC(self, state, t, theta):
        """Старое название для _jacobian_c. Оставлено для совместимости."""
        return self._jacobian_c(state, t, theta)

    def JacobianX_jax(self, state, t, theta):
        """Старое название для _jacobian_x_jax. Оставлено для совместимости."""
        return self._jacobian_x_jax(state, t, theta)

    def JacobianC_jax(self, state, t, theta):
        """Старое название для _jacobian_c_jax. Оставлено для совместимости."""
        return self._jacobian_c_jax(state, t, theta)
    


class SyntheticDataGenerator:
    """
    Генератор синтетических данных для динамической системы.

    Параметры
    ----------
    system : object
        Система с методами:
            - get_dimentions() -> (state_len, theta_len, meas_len)
            - get_solution(c0, theta, t_eval) -> array (state_len, n_t) (обычный режим)
            - get_solution_jax(c0, theta, t_eval) -> array (n_t, state_len) (JAX-режим)
            - h_x(state, t, theta) -> измерение в момент t
    sigma : float, default=0.01
        Стандартное отклонение аддитивного гауссовского шума.
    perturb_initial : bool, default=True
        Флаг: если True, начальные условия для каждого батча возмущаются.
    perturbation_scale : float, default=0.1
        Масштаб возмущения (c0_true = c0 * (1 + scale * (rand - 0.5))).
    use_jax : bool, default=True
        Использовать JAX-интегрирование (get_solution_jax) или обычное.
    """

    def __init__(self, system_ode: ODESystem, sigma=0.01, perturb_initial=False, perturbation_scale=0.1, use_jax=True):
        self.system = SystemJacobian(system_ode)
        self.sigma = sigma
        self.perturb_initial = perturb_initial
        self.perturbation_scale = perturbation_scale
        self.use_jax = use_jax



        self.state_len, self.theta_len, self.meas_len = self.system.get_dimentions()

    def generate_batch(self, c0, theta, t_start, t_end, n_measurements, seed=None):
        """
        Генерирует один батч данных на интервале [t_start, t_end].

        Параметры
        ----------
        c0 : array_like (state_len,)
            Номинальное начальное состояние.
        theta : array_like (theta_len,)
            Параметры системы.
        t_start : float
            Начало интервала.
        t_end : float
            Конец интервала.
        n_measurements : int
            Количество точек измерений.
        seed : int, optional
            Seed для генератора случайных чисел (для воспроизводимости).

        Возвращает
        ----------
        t_eval : np.ndarray (n_measurements,)
            Временные точки.
        measurements : np.ndarray (n_measurements, meas_len)
            Зашумлённые измерения.
        states : np.ndarray (n_measurements, state_len)
            Зашумлённые состояния.
        """
        if seed is not None:
            np.random.seed(seed)

        # Возмущение начального условия
        if self.perturb_initial:
            c0_true = c0 * (1 + self.perturbation_scale * (np.random.random(self.state_len) - 0.5))
        else:
            c0_true = c0

        t_eval = np.linspace(t_start, t_end, n_measurements)

        # Интегрирование
        if self.use_jax:
            # get_solution_jax возвращает (n_measurements, state_len)
            solution = self.system.get_solution_jax(c0_true, theta, t_eval)
        else:
            # get_solution возвращает (state_len, n_measurements)
            sol = self.system.get_solution(c0_true, theta, t_eval)
            solution = sol.T  # приводим к (n_measurements, state_len)

        # Добавляем шум к состояниям
        noise = self.sigma * np.random.normal(size=(self.state_len, n_measurements))
        noisy_states = (solution + noise).T  # (n_measurements, state_len)

        # Вычисляем измерения
        measurements = np.zeros((n_measurements, self.meas_len))
        for i, state in enumerate(noisy_states):
            measurements[i] = self.system.h_x(state, t_eval[i], theta)

        return t_eval, measurements, noisy_states

    def generate(self, c0, theta, time_intervals, n_measurements, seeds=None):
        """
        Генерирует несколько батчей данных для заданных временных интервалов.

        Параметры
        ----------
        c0 : array_like (state_len,)
            Номинальное начальное состояние.
        theta : array_like (theta_len,)
            Параметры системы.
        time_intervals : list of (t_start, t_end)
            Список интервалов времени для каждого батча.
        n_measurements : int
            Количество точек измерений в каждом батче.
        seeds : list of int, optional
            Список seed'ов для каждого батча (должен быть той же длины).

        Возвращает
        ----------
        t_batches : list of np.ndarray
            Временные точки для каждого батча.
        measured_batches : list of np.ndarray
            Зашумлённые измерения для каждого батча.
        state_batches : list of np.ndarray
            Зашумлённые состояния для каждого батча.
        """
        if seeds is not None and len(seeds) != len(time_intervals):
            raise ValueError("Длина seeds должна совпадать с количеством интервалов")

        t_batches = []
        measured_batches = []
        state_batches = []

        for idx, (t_start, t_end) in enumerate(time_intervals):
            seed = seeds[idx] if seeds is not None else None
            t_eval, meas, states = self.generate_batch(
                c0, theta, t_start, t_end, n_measurements, seed=seed
            )
            t_batches.append(t_eval)
            measured_batches.append(meas)
            state_batches.append(states)

        return t_batches, measured_batches, state_batches
    





class MHESyntheticDataGenerator:
    """
    Generates synthetic measurement data for a dynamical system,
    including control inputs.
    """

    def __init__(self, system_ode: ODESystem, sigma=1e-3):
        self.system = SystemJacobian(system_ode)
        self.sigma = sigma
        self.state_dim, self.param_dim, self.meas_dim = self.system.get_dimentions()
        # Determine control dimension (adjust if your system has a different attribute)

        self.control_dim = system_ode.nu


    def _generate_trajectory(self, c0, theta, t, sigma=None):
        """
        Generate a trajectory at given time points.

        Args:
            c0 (array): Initial state.
            theta (array): Model parameters.
            t (array): Time points (must be strictly increasing).
            sigma (float, optional): Noise std.

        Returns:
            tuple: (t, u, full_states, measured_states)
        """
        if sigma is None:
            sigma = self.sigma

        # Get control inputs at each time point
        u = np.zeros((len(t), self.control_dim))
        for i, ti in enumerate(t):
            u[i] = self.system.f_sym.get_input_signals(ti)

        # Integrate system to obtain full states
        full_states = self.system.get_solution(c0, theta, t).T  # shape (len(t), state_dim)

        # Add measurement noise
        mean = np.zeros(self.state_dim)
        cov = np.diag([sigma**2] * self.state_dim)
        noise = np.random.multivariate_normal(mean, cov, len(t))
        noisy_full = full_states + noise

        # Compute measured outputs
        measured = np.zeros((len(t), self.meas_dim))
        for i, state in enumerate(noisy_full):
            
            measured[i] = self.system.h_x(state, t[i], theta)
           

        return t, u, noisy_full, measured

    def generate_sliding_windows_exact(self, c0, theta, t0, tf, num_windows,
                                       n_measurement, overlap_points=1, sigma=None):
        """
        Generate overlapping windows that each cover exactly T_f seconds.

        Args:
            c0 (array): Initial state.
            theta (array): Model parameters.
            t0 (float): Start time of the first window.
            t_f (float): Time span of each window.
            num_windows (int): Number of overlapping windows.
            n_measurement (int): Number of points per window.
            overlap_points (int): Number of points that overlap between consecutive windows.
                                   Default 1 gives windows like [1-8],[2-9] if step = N_measurement-1.
            sigma (float, optional): Noise std.

        Returns:
            tuple: (list_of_time_arrays, list_of_control_inputs,
                    list_of_measured_states, list_of_full_states)
        """

        assert(len(c0) == self.state_dim)
        assert(len(theta) == self.param_dim)
        # Time step between consecutive points inside a window
        dt = tf / (n_measurement - 1)

        # Step in points between window starts
        step = n_measurement - overlap_points

        # Build the overall time vector
        total_points = 1 + (num_windows - 1) * step + (n_measurement - 1)
        t_long = np.linspace(t0, t0 + (num_windows - 1) * step * dt + tf, total_points)

        # Generate the long trajectory
        t_long, u_long, full_long, meas_long = self._generate_trajectory(
            c0, theta, t_long, sigma
        )

        # Extract windows
        t_windows = []
        u_windows = []
        meas_windows = []
        full_windows = []

        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + n_measurement
            t_windows.append(t_long[start_idx:end_idx])
            u_windows.append(u_long[start_idx:end_idx])
            meas_windows.append(meas_long[start_idx:end_idx])
            full_windows.append(full_long[start_idx:end_idx])

        return t_windows, u_windows, meas_windows, full_windows
    

def check_system_ok(system_ode : ODESystem):
            # Проверяем наличие необходимых методов
    system = SystemJacobian(system_ode)
    if not hasattr(system, 'get_dimentions'):
        raise AttributeError("system должен иметь метод get_dimentions()")

    if not hasattr(system, 'h_x'):
        raise AttributeError("system должен иметь метод h_x(state, t, theta)")
    
    return True