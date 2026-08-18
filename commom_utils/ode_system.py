import numpy as np
import jax
from jax.experimental.ode import odeint
from jax import numpy as jnp
from scipy.integrate import solve_ivp
from casadi import SX, vertcat, Function, jacobian
import casadi as ca
from abc import abstractmethod
from typing import NamedTuple
from jaxadi import convert

from commom_utils.sensitivity import group_by_grid_length


class Dims(NamedTuple):
    """Размерности задачи. NamedTuple, чтобы работала и распаковка
    `nx, n_theta, n_obs = system.dims()`, и обращение по имени."""
    nx: int          # размерность состояния
    n_theta: int     # число оцениваемых параметров
    n_obs: int       # размерность наблюдения h(x, theta, u)


class ODESystem:
    def __init__(self, nx, n_theta, nu):
        self.nx = nx
        self.n_theta = n_theta
        self.nu = nu
        self.state = ca.SX.sym("x", nx)
        self.theta = ca.SX.sym("theta", n_theta)
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

    def __init__(self, model: ODESystem, method: str = 'RK45'):
        """
        model: объект System, предоставляющий символьное описание модели.
        method: метод интегрирования для solve_ivp (например, 'RK45').
        """
        self.ATOL = 1e-5
        self.RTOL = 1e-5
        self.model = model
        self.method = method

        # Получаем символьные переменные и выражения
        state_var, theta_var, inp_signal_var, f = self.model.get_system()
        h_observ = self.model.observation(state_var, theta_var, inp_signal_var)

        # Списки элементов для CasADi функций
        state_list = state_var.elements()
        inp_list = inp_signal_var.elements()
        theta_list = theta_var.elements()

        self.nu = len(inp_list)
        self.nx = len(state_list)
        self.n_theta = len(theta_list)
        self.n_obs = len(h_observ.elements())

        # --- Создание CasADi функций ---
        self._f_ca = Function('func', [*state_list, *inp_list, *theta_list], [f])
        self._f_jax_ca = convert(self._f_ca, compile=True)

        self._h_ca = Function('h', [*state_list, *inp_list, *theta_list], [h_observ])

        # Якобианы
        J_h_x = jacobian(h_observ, vertcat(*state_list))
        self._h_x_ca = Function('J_h_x', [*state_list, *inp_list, *theta_list], [J_h_x])

        J_h_theta = jacobian(h_observ, vertcat(*theta_list))
        self._h_theta_ca = Function('J_h_theta', [*state_list, *inp_list, *theta_list], [J_h_theta])

        J_p = jacobian(f, vertcat(*theta_list))
        self._f_theta_ca = Function('J_p', [*state_list, *inp_list, *theta_list], [J_p])
        self._f_theta_jax_ca = convert(self._f_theta_ca, compile=True)

        J_x = jacobian(f, vertcat(*state_list))
        self._f_x_ca = Function('J_x', [*state_list, *inp_list, *theta_list], [J_x])
        self._f_x_jax_ca = convert(self._f_x_ca, compile=True)

        # Слайсы блоков S_theta и S_c внутри плоского расширенного состояния
        # [x; S_theta.flatten(); S_c.flatten()] — см. commom_utils/sensitivity.py
        self._IDX_S_THETA = slice(self.nx, self.nx + self.nx * self.n_theta)
        self._IDX_S_C = slice(self._IDX_S_THETA.stop, self._IDX_S_THETA.stop + self.nx * self.nx)

        # Тождественное наблюдение h(x) = x: dh/dx = I, dh/dθ = 0 —
        # позволяет полностью пропустить вычисление якобианов наблюдения
        state_vec = vertcat(*state_list)
        try:
            self.identity_observation = (h_observ.shape == state_vec.shape
                                         and bool(ca.is_equal(h_observ, state_vec, 20)))
        except Exception:
            self.identity_observation = False

        # Кэш map-версий функций наблюдения (ключ: (имя, N точек))
        self._obs_map_cache = {}

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    def _get_inp_signals(self, t):
        """Входные сигналы модели в момент t.

        Ошибку НЕ глушим: раньше любое исключение печаталось и подменялось
        нулями, то есть опечатка в get_input_signals давала не падение, а
        молча неверные чувствительности и неверную оценку параметров.
        """
        try:
            return self.model.get_input_signals(t)
        except Exception as exc:
            raise RuntimeError(
                f"{type(self.model).__name__}.get_input_signals(t) упала при "
                f"t={t}. Метод зовётся ИЗНУТРИ правой части ОДУ, в том числе "
                f"с трассируемым временем под jax odeint и с массивом времени "
                f"в коллокациях: допустим только jnp, без math.* и без "
                f"питоновского `if t < ...` (нужен jnp.where).") from exc

    def dims(self):
        """Размерности задачи: (nx, n_theta, n_obs)."""
        return Dims(self.nx, self.n_theta, self.n_obs)

    # ----------------------------------------------------------------------
    # Методы для обычного режима (NumPy)
    # ----------------------------------------------------------------------
    def h(self, state, t, theta):
        """Вычисляет выход системы в момент t."""
        inp = self._get_inp_signals(t)
        return np.array(self._h_ca(*state, *inp, *theta)).flatten()

    # def inverse_h(self, y, t, theta, x_guess=None, n_iter=1):
    #     """
    #     Приближённо решает уравнение h(x, theta) = y относительно x.
    #     Параметры:
    #         y: измерение (meas_len,)
    #         t: время
    #         theta: параметры (theta_len,)
    #         x_guess: начальное приближение (state_len,). Если None, то нули.
    #         n_iter: число итераций Гаусса–Ньютона (обычно 1-2).
    #     Возвращает:
    #         x: оценка состояния (state_len,)
    #     """
    #     if x_guess is None:
    #         x_guess = np.zeros(self.nx)
        
    #     x = x_guess.copy()
    #     for _ in range(n_iter):
    #         dh_dx = self.dh_dx(x, t, theta)        # (meas_len, state_len)
    #         h_val = self.h(x, t, theta)          # (meas_len,)
    #         residual = y - h_val
    #         # Решаем линейную систему (least squares)
    #         delta_x = np.linalg.lstsq(dh_dx, residual, rcond=None)[0]
    #         x = x + delta_x
    #     return x
    
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
            dh_dx = self.dh_dx(x, t, theta)
            h_val = self.h(x, t, theta)
            residual = y - h_val
            # Проверка на некорректные значения
            if np.any(np.isnan(dh_dx)) or np.any(np.isnan(residual)):
                raise ValueError("NaN encountered in inverse_h iteration")
            # Используем псевдообращение с автоматическим выбором порога
            try:
                delta_x = np.linalg.lstsq(dh_dx, residual, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback на псевдообращение через pinv с явным rcond
                delta_x = np.linalg.pinv(dh_dx, rcond=1e-6) @ residual
            x = x + delta_x
        return x
    
    def f(self, state, t, theta):
        """Вычисляет правую часть системы в момент t."""
        inp = self._get_inp_signals(t)
        return np.array(self._f_ca(*state, *inp, *theta)).flatten()

    def dh_dx(self, state, t, theta):
        """Якобиан выхода по состоянию."""
        inp = self._get_inp_signals(t)
        return np.array(self._h_x_ca(*state, *inp, *theta))

    def dh_dtheta(self, state, t, theta):
        """Якобиан выхода по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self._h_theta_ca(*state, *inp, *theta)).squeeze()

    def df_dtheta(self, state, t, theta):
        """Якобиан правой части по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self._f_theta_ca(*state, *inp, *theta))

    def df_dx(self, state, t, theta):
        """Якобиан правой части по состоянию."""
        inp = self._get_inp_signals(t)
        return np.array(self._f_x_ca(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # Батчевые вычисления наблюдений (CasADi Function.map)
    # ----------------------------------------------------------------------
    def _obs_mapped(self, name, n_points):
        """Map-версия функции наблюдения на n_points точек (с кэшем)."""
        key = (name, n_points)
        if key not in self._obs_map_cache:
            base = {'h': self._h_ca,
                    'dh_dx': self._h_x_ca,
                    'dh_dtheta': self._h_theta_ca}[name]
            self._obs_map_cache[key] = base.map(n_points)
        return self._obs_map_cache[key]

    def observation_batch(self, states, t_array, theta):
        """h, dh/dx и dh/dθ сразу для всех точек (3 вызова CasADi вместо 3N).

        Параметры: states (nx, N), t_array (N,), theta (nθ,).
        Возвращает: h (N, n_obs), dh_dx (N, n_obs, nx), dh_dtheta (N, n_obs, nθ).
        """
        n_points = states.shape[1]
        try:
            # Интерполяторы обычно принимают массив времени целиком
            inp = np.array([np.asarray(s, dtype=float).reshape(n_points)
                            for s in self.model.get_input_signals(np.asarray(t_array))]).T
        except Exception:
            inp = np.array([np.asarray(self._get_inp_signals(t), dtype=float).reshape(self.nu)
                            for t in t_array]).reshape(n_points, self.nu)

        # Каждый скалярный вход map-функции — строка (1, N); theta транслируется
        args = [states[i, :].reshape(1, n_points) for i in range(self.nx)]
        args += [inp[:, j].reshape(1, n_points) for j in range(self.nu)]
        args += [float(theta[k]) for k in range(self.n_theta)]

        h = np.array(self._obs_mapped('h', n_points)(*args))  # (n_obs, N)
        # map конкатенирует выходы по столбцам: (n_obs, nx*N) -> (N, n_obs, nx)
        dh_dx = np.array(self._obs_mapped('dh_dx', n_points)(*args))
        dh_dx = dh_dx.reshape(self.n_obs, n_points, self.nx).transpose(1, 0, 2)
        dh_dtheta = np.array(self._obs_mapped('dh_dtheta', n_points)(*args))
        dh_dtheta = dh_dtheta.reshape(self.n_obs, n_points, self.n_theta).transpose(1, 0, 2)
        return h.T, dh_dx, dh_dtheta

    # ----------------------------------------------------------------------
    # JAX-методы (используют скомпилированные функции из jaxadi)
    # ----------------------------------------------------------------------
    def f_jax(self, y, t, *theta):
        """JAX-совместимая правая часть."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._f_jax_ca(*y, *inp, *theta)[0].flatten())

    def df_dtheta_jax(self, state, t, theta):
        """JAX-совместимый якобиан по параметрам."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._f_theta_jax_ca(*state, *inp, *theta))[0]

    def df_dx_jax(self, state, t, theta):
        """JAX-совместимый якобиан по состоянию."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._f_x_jax_ca(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # Интегрирование
    # ----------------------------------------------------------------------
    def get_solution(self, c0, theta, t_eval):
        """Интегрирование только состояния (обычный режим)."""
        def system(t, y):
            return self.f(y, t, theta[:self.n_theta])

        sol = solve_ivp(system, (t_eval[0], t_eval[-1]), c0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Интегрирование не сошлось: {sol.message}")
        return sol.y

    def get_jacobian_solution(self, c0, theta, t_eval):
        """Интегрирование расширенной системы (состояние + чувствительности) (обычный режим)."""
        n = self.nx
        p = self.n_theta

        J0 = np.concatenate([np.zeros((n, p)).flatten(), np.eye(n).flatten()])
        y0 = np.concatenate([c0, J0])

        def full_ode(t, y):
            return self._variational_rhs(y, t, theta[:p])

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
        sol = odeint(self.f_jax,
                     jnp.array(c0),
                     jnp.array(t_eval),
                     *theta[:self.n_theta],
                     rtol=self.RTOL, atol=self.ATOL)
        return np.array(sol).T

    def get_jacobian_solution_jax(self, c0, theta, t_eval):
        """JAX-интегрирование расширенной системы (состояние + чувствительности)."""
        n = self.nx
        p = self.n_theta

        J0 = jnp.concatenate([jnp.zeros((n, p)).flatten(), jnp.eye(n).flatten()])
        y0 = jnp.concatenate([jnp.array(c0), J0])

        sol = odeint(self._variational_rhs_jax, y0, jnp.array(t_eval), *theta[:p],
                     rtol=self.RTOL, atol=self.ATOL)
        return np.array(sol).T

    def _vmapped_full_integrator(self):
        """JIT+vmap-обёртка интегратора расширенной системы (кэшируется).

        jax.jit сам перекомпилирует при смене формы (число шутов, длина сетки).
        """
        if getattr(self, '_jax_vmap_full', None) is None:
            n, p = self.nx, self.n_theta
            J0 = jnp.concatenate([jnp.zeros(n * p), jnp.eye(n).flatten()])

            def integrate_one(c0, t_grid, theta):
                y0 = jnp.concatenate([c0, J0])
                return odeint(self._variational_rhs_jax, y0, t_grid, *theta,
                              rtol=self.RTOL, atol=self.ATOL)

            self._jax_vmap_full = jax.jit(
                jax.vmap(integrate_one, in_axes=(0, 0, None)))
        return self._jax_vmap_full

    def get_jacobian_solution_jax_batch(self, c0_list, theta, t_grids):
        """Батчевое интегрирование расширенной системы сразу для всех шутов.

        Шуты группируются по длине временной сетки (vmap требует одинаковых
        форм), каждая группа интегрируется одним vmap-вызовом odeint.

        c0_list: список/массив (S, nx); t_grids: список массивов времени.
        Возвращает список из S матриц (nx + nx·np + nx·nx, L_i) — как
        get_jacobian_solution_jax для каждого шута.
        """
        theta_j = jnp.array(np.asarray(theta[:self.n_theta], dtype=float))

        results = [None] * len(t_grids)
        integrate = self._vmapped_full_integrator()
        for idxs in group_by_grid_length(t_grids):
            ts_stack = jnp.array(np.stack([np.asarray(t_grids[i]) for i in idxs]))
            c0_stack = jnp.array(np.stack([np.asarray(c0_list[i]) for i in idxs]))
            sols = np.array(integrate(c0_stack, ts_stack, theta_j))  # (k, L, dim)
            for j, i in enumerate(idxs):
                results[i] = sols[j].T
        return results

    # ----------------------------------------------------------------------
    # Вариационные уравнения (theory_gauss_newton.ipynb, раздел 3)
    #
    #     J_theta' = f_x J_theta + f_theta,   J_theta(t_0) = 0
    #     J_c'     = f_x J_c,                 J_c(t_0)     = I
    #
    # Записаны ровно ДВАЖДЫ — numpy и jax, — одинаковой схемой: f_x общий
    # множитель обоих уравнений, поэтому считается один раз и применяется к
    # склейке [S_theta | S_c]. Разница между версиями только в бэкенде.
    # ----------------------------------------------------------------------
    def _variational_rhs(self, y, t, theta):
        """Правая часть расширенной системы (numpy).

        Входные сигналы вычисляются ОДИН раз за вызов: раньше путь шёл через
        f/df_dx/df_dtheta, и пользовательская get_input_signals звалась трижды
        на каждый шаг интегратора.
        """
        n, p = self.nx, self.n_theta
        x = y[:n]
        S_theta = y[self._IDX_S_THETA].reshape((n, p))
        S_c = y[self._IDX_S_C].reshape((n, n))

        args = (*x, *self._get_inp_signals(t), *theta)
        dx = np.array(self._f_ca(*args)).ravel()
        f_x = np.array(self._f_x_ca(*args))
        dS = f_x @ np.concatenate([S_theta, S_c], axis=1)
        dS[:, :p] += np.array(self._f_theta_ca(*args))
        # layout плоский: сначала весь S_theta, потом весь S_c (C-order каждый),
        # поэтому склейку приходится снова разрезать, а не ravel'ить целиком
        return np.concatenate([dx, dS[:, :p].ravel(), dS[:, p:].ravel()])

    def _variational_rhs_jax(self, y, t, *theta):
        """Правая часть расширенной системы (jax) — та же схема."""
        n, p = self.nx, self.n_theta
        x = y[:n]
        S_theta = y[self._IDX_S_THETA].reshape((n, p))
        S_c = y[self._IDX_S_C].reshape((n, n))

        # df_dx_jax отдаёт (1, n, n) — приводим к (n, n) явно, а не полагаемся
        # на broadcast (раньше он молча давал (1, n, p) на выходе матумножения)
        f_x = self.df_dx_jax(x, t, theta).reshape((n, n))
        dx = self.f_jax(x, t, *theta)
        dS = f_x @ jnp.concatenate([S_theta, S_c], axis=1)
        dS_theta = dS[:, :p] + self.df_dtheta_jax(x, t, theta)
        return jnp.concatenate([dx, dS_theta.flatten(), dS[:, p:].flatten()])


class SystemIntegrator(SystemJacobian):
    """Интегрирование с УДЕРЖИВАЕМЫМ входом u, заданным вызывающей стороной.

    Отличие от SystemJacobian: там входы берутся из модели
    (get_input_signals(t)), а здесь u — аргумент и держится постоянным на
    шаге. Именно это нужно симуляции MPC: вход выдаёт регулятор, а не модель.

    Раньше это был отдельный класс, целиком повторявший компиляцию CasADi из
    SystemJacobian, причём с ДРУГИМ порядком аргументов — Function объявлялся
    как [state, theta, u], а get_lin_system_dynamics звал его как
    (state, u, theta), то есть подставлял вход в слоты параметров. Теперь
    функции берутся у родителя, и такой рассинхрон невозможен по построению.
    """

    def __init__(self, model: ODESystem, method: str = 'RK45'):
        super().__init__(model, method=method)
        # Якобиан по входу — единственное, чего нет у родителя
        state_var, theta_var, inp_var, f = model.get_system()
        self._f_u_ca = Function(
            'J_u', [*state_var.elements(), *inp_var.elements(),
                    *theta_var.elements()],
            [jacobian(f, vertcat(*inp_var.elements()))])

    def f_of_u(self, state, u, theta):
        """Правая часть при явно заданном входе u."""
        return np.array(self._f_ca(*state, *u, *theta)).ravel()

    def f_of_u_jax(self, state, t, u, theta):
        return jnp.array(self._f_jax_ca(*state, *u, *theta)[0]).flatten()

    def _check(self, c0, u, theta):
        if not (len(c0) == self.nx and len(u) == self.nu
                and len(theta) == self.n_theta):
            raise ValueError(
                f"ожидалось x({self.nx}), u({self.nu}), theta({self.n_theta}); "
                f"получено x({len(c0)}), u({len(u)}), theta({len(theta)})")

    def integrate(self, c0, u, theta, t_span):
        """Траектория на t_span при постоянном u."""
        self._check(c0, u, theta)
        sol = solve_ivp(lambda t, y: self.f_of_u(y, u, theta), t_span, c0,
                        method=self.method)
        if not sol.success:
            raise RuntimeError(f"Интегрирование не сошлось: {sol.message}")
        return sol.y.T

    def step(self, c0, u, theta, dt):
        """Один шаг длиной dt при постоянном u."""
        return self.integrate(c0, u, theta, (0.0, dt))[-1]

    def step_jax(self, c0, u, theta, dt):
        self._check(c0, u, theta)
        # Допуски НЕ передаём: у odeint по умолчанию 1.4e-8, и симуляция MPC
        # исторически считалась с ними. Ослаблять их здесь — отдельное решение,
        # а не побочный эффект рефакторинга (см. RTOL/ATOL для якобианов).
        sol = odeint(self.f_of_u_jax, jnp.array(c0), jnp.array([0.0, dt]),
                     u, theta)
        return np.array(sol[-1])

    def get_lin_system_dynamics(self, state, u, theta):
        """Линеаризация (A, B, D) = (df/dx, df/du, df/dtheta) в точке."""
        self._check(state, u, theta)
        args = (*state, *u, *theta)
        return (np.array(self._f_x_ca(*args)),
                np.array(self._f_u_ca(*args)),
                np.array(self._f_theta_ca(*args)))


class SyntheticDataGenerator:
    """
    Генератор синтетических данных для динамической системы.

    Параметры
    ----------
    system : object
        Система с методами:
            - dims() -> (state_len, theta_len, meas_len)
            - get_solution(c0, theta, t_eval) -> array (state_len, n_t) (обычный режим)
            - get_solution_jax(c0, theta, t_eval) -> array (n_t, state_len) (JAX-режим)
            - h(state, t, theta) -> измерение в момент t
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



        self.state_len, self.theta_len, self.meas_len = self.system.dims()

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
            solution = sol  # приводим к (n_measurements, state_len)

        # Добавляем шум к состояниям
        noise = self.sigma * np.random.normal(size=(self.state_len, n_measurements))
        noisy_states = (solution + noise).T  # (n_measurements, state_len)

        # Вычисляем измерения
        measurements = np.zeros((n_measurements, self.meas_len))
        inp_signal = np.zeros((n_measurements, self.system.nu))
        for i, state in enumerate(noisy_states):
            measurements[i] = self.system.h(state, t_eval[i], theta)
            inp_signal[i] = self.system.model.get_input_signals(t_eval[i])
        return t_eval, measurements, noisy_states, inp_signal

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
        inp_signals = []
        for idx, (t_start, t_end) in enumerate(time_intervals):
            seed = seeds[idx] if seeds is not None else None
            t_eval, meas, states, inp_signal = self.generate_batch(
                c0, theta, t_start, t_end, n_measurements, seed=seed
            )
            t_batches.append(t_eval)
            measured_batches.append(meas)
            state_batches.append(states)
            inp_signals.append(inp_signal)
            
        return t_batches, measured_batches, state_batches, inp_signals
    





class MHESyntheticDataGenerator:
    """
    Generates synthetic measurement data for a dynamical system,
    including control inputs.
    """

    def __init__(self, system_ode: ODESystem, sigma=1e-3):
        self.system = SystemJacobian(system_ode)
        self.sigma = sigma
        self.state_dim, self.param_dim, self.meas_dim = self.system.dims()
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
            sigma = [0] * self.meas_dim
        # Get control inputs at each time point
        u = np.zeros((len(t), self.control_dim))
        for i, ti in enumerate(t):
            u[i] = self.system.model.get_input_signals(ti)

        # Integrate system to obtain full states
        full_states = self.system.get_solution(c0, theta, t).T  # shape (len(t), state_dim)

        # Compute measured outputs
        measured = np.zeros((len(t), self.meas_dim))
        for i, state in enumerate(full_states):
            measured[i] = self.system.h(state, t[i], theta)
           
        mean = np.zeros(self.meas_dim)
        cov = np.diag(sigma**2)
        noise = np.random.multivariate_normal(mean, cov, len(t))
        measured += noise
        return t, u, full_states, measured

    def generate_sliding_windows_exact(self, c0, theta, t0, tf, num_windows,
                                       n_measurement, overlap_points=1):
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
            c0, theta, t_long, self.sigma
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

    system = SystemJacobian(system_ode)
    assert system.nu == len(system_ode.get_input_signals(0))
    if not hasattr(system, 'dims'):
        raise AttributeError("system должен иметь метод dims()")

    if not hasattr(system, 'h'):
        raise AttributeError("system должен иметь метод h(state, t, theta)")
    
    return True