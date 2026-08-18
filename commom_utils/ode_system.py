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

from commom_utils.sensitivity import (group_by_grid_length, initial_flat_row,
                                      split_row)


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


class CompiledModel:
    """Скомпилированная модель: f, h и их якобианы, numpy- и jax-бэкенды.

    ТОЛЬКО модель, без интегрирования: компиляция CasADi/jaxadi, поточечный
    фасад (f, h, df_dx, ...), батчевые наблюдения и обращение наблюдения.
    Интегрирование чувствительностей — отдельные классы по композиции:
    VariationalIntegrator (здесь же) и CollocationIntegrator
    (commom_utils/collocation.py) — оба держат CompiledModel и делят одну
    компиляцию.
    """

    def __init__(self, model: ODESystem):
        """model: объект ODESystem, предоставляющий символьное описание."""
        self.model = model

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

        # --- Компиляция CasADi-функций (у всех один список аргументов) ---
        args = [*state_list, *inp_list, *theta_list]
        state_vec, theta_vec = vertcat(*state_list), vertcat(*theta_list)

        def _fn(name, expr):
            return Function(name, args, [expr])

        # Имена приватных хендлов повторяют публичные методы-обёртки:
        # _f_ca <-> f, _dh_dx_ca <-> dh_dx и т.д. (одна величина = одно имя)
        self._f_ca = _fn('f', f)
        self._f_jax_ca = convert(self._f_ca, compile=True)
        self._h_ca = _fn('h', h_observ)
        self._dh_dx_ca = _fn('dh_dx', jacobian(h_observ, state_vec))
        self._dh_dtheta_ca = _fn('dh_dtheta', jacobian(h_observ, theta_vec))
        self._df_dtheta_ca = _fn('df_dtheta', jacobian(f, theta_vec))
        self._df_dtheta_jax_ca = convert(self._df_dtheta_ca, compile=True)
        self._df_dx_ca = _fn('df_dx', jacobian(f, state_vec))
        self._df_dx_jax_ca = convert(self._df_dx_ca, compile=True)

        self.identity_observation = self._is_identity_observation(h_observ,
                                                                  state_vec)

        # Кэши: map-версии функций наблюдения (ключ: (имя, N точек))
        # и vmap-обёртка интегратора расширенной системы
        self._obs_map_cache = {}
        self._jax_vmap_full = None

    @staticmethod
    def _is_identity_observation(h_observ, state_vec):
        """h(x) = x? Тогда dh/dx = I, dh/dtheta = 0, и якобианы наблюдения
        можно не вычислять вовсе.

        20 — глубина структурного сравнения выражений в ca.is_equal
        (не допуск!); is_equal может бросить на несравнимых формах —
        тогда считаем наблюдение нетождественным.
        """
        try:
            return (h_observ.shape == state_vec.shape
                    and bool(ca.is_equal(h_observ, state_vec, 20)))
        except Exception:
            return False

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

    def _inp_on_times(self, t_flat):
        """Входные сигналы на векторе времени -> (len(t_flat), nu).

        Быстрый путь — один вызов get_input_signals с массивом времени
        (интерполяторы обычно его принимают); если функция входов массив не
        переваривает — семантически эквивалентный обход по точкам.
        Общий помощник обоих батчевых потребителей: observation_batch и
        подготовки узловых входов в коллокациях.
        """
        t_flat = np.asarray(t_flat)
        n = t_flat.size
        if self.nu == 0:
            return np.zeros((n, 0))
        try:
            sigs = self.model.get_input_signals(t_flat)
            return np.array([np.asarray(s, dtype=float).reshape(n)
                             for s in sigs]).T
        except Exception:
            return np.array(
                [np.asarray(self._get_inp_signals(t), dtype=float)
                 .reshape(self.nu) for t in t_flat]).reshape(n, self.nu)

    # ----------------------------------------------------------------------
    # Методы для обычного режима (NumPy)
    # ----------------------------------------------------------------------
    def h(self, state, t, theta):
        """Вычисляет выход системы в момент t."""
        inp = self._get_inp_signals(t)
        return np.array(self._h_ca(*state, *inp, *theta)).flatten()

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
        return np.array(self._dh_dx_ca(*state, *inp, *theta))

    def dh_dtheta(self, state, t, theta):
        """Якобиан выхода по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self._dh_dtheta_ca(*state, *inp, *theta)).squeeze()

    def df_dtheta(self, state, t, theta):
        """Якобиан правой части по параметрам."""
        inp = self._get_inp_signals(t)
        return np.array(self._df_dtheta_ca(*state, *inp, *theta))

    def df_dx(self, state, t, theta):
        """Якобиан правой части по состоянию."""
        inp = self._get_inp_signals(t)
        return np.array(self._df_dx_ca(*state, *inp, *theta))

    # ----------------------------------------------------------------------
    # Батчевые вычисления наблюдений (CasADi Function.map)
    # ----------------------------------------------------------------------
    def _obs_mapped(self, name, n_points):
        """Map-версия функции наблюдения на n_points точек (с кэшем)."""
        key = (name, n_points)
        if key not in self._obs_map_cache:
            base = {'h': self._h_ca,
                    'dh_dx': self._dh_dx_ca,
                    'dh_dtheta': self._dh_dtheta_ca}[name]
            self._obs_map_cache[key] = base.map(n_points)
        return self._obs_map_cache[key]

    def observation_batch(self, states, t_array, theta):
        """h, dh/dx и dh/dθ сразу для всех точек (3 вызова CasADi вместо 3N).

        Параметры: states (nx, N), t_array (N,), theta (nθ,).
        Возвращает: h (N, n_obs), dh_dx (N, n_obs, nx), dh_dtheta (N, n_obs, nθ).
        """
        n_points = states.shape[1]
        inp = self._inp_on_times(t_array)

        # Каждый скалярный вход map-функции — строка (1, N); theta транслируется
        args = [states[i, :].reshape(1, n_points) for i in range(self.nx)]
        args += [inp[:, j].reshape(1, n_points) for j in range(self.nu)]
        args += [float(theta[k]) for k in range(self.n_theta)]

        def unstack(mat, width):
            # map стыкует матричные выходы по столбцам:
            # (n_obs, width*N) -> (N, n_obs, width)
            return np.array(mat).reshape(self.n_obs, n_points,
                                         width).transpose(1, 0, 2)

        h = np.array(self._obs_mapped('h', n_points)(*args))  # (n_obs, N)
        dh_dx = unstack(self._obs_mapped('dh_dx', n_points)(*args), self.nx)
        dh_dtheta = unstack(self._obs_mapped('dh_dtheta', n_points)(*args),
                            self.n_theta)
        return h.T, dh_dx, dh_dtheta

    # ----------------------------------------------------------------------
    # JAX-методы (используют скомпилированные функции из jaxadi)
    # ----------------------------------------------------------------------
    def f_jax(self, y, t, *theta):
        """JAX-совместимая правая часть."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._f_jax_ca(*y, *inp, *theta)[0].flatten())

    def df_dtheta_jax(self, state, t, theta):
        """JAX-совместимый якобиан по параметрам, (nx, n_theta)."""
        inp = self._get_inp_signals(t)
        return jnp.array(self._df_dtheta_jax_ca(*state, *inp, *theta))[0]

    def df_dx_jax(self, state, t, theta):
        """JAX-совместимый якобиан по состоянию, (nx, nx).

        jaxadi отдаёт список выходов — без [0] здесь была бы форма
        (1, nx, nx), и матумножение на неё молча давало лишнюю ось.
        """
        inp = self._get_inp_signals(t)
        return jnp.array(self._df_dx_jax_ca(*state, *inp, *theta))[0]

class VariationalIntegrator:
    """Интегратор чувствительностей: вариационные уравнения, scipy/jax.

    Модель (CompiledModel) держится по КОМПОЗИЦИИ: она отвечает за f, h и их
    якобианы, интегратор — за уравнения, сетку и допуски. Тот же контракт
    выхода реализует CollocationIntegrator (commom_utils/collocation.py) —
    для жёстких систем; здесь оба бэкенда ЯВНЫЕ (solve_ivp RK45 / jax dopri).

    Вариационные уравнения (theory_gauss_newton.ipynb, раздел 3):

        J_theta' = f_x J_theta + f_theta,   J_theta(t_0) = 0
        J_c'     = f_x J_c,                 J_c(t_0)     = I

    Правые части записаны ровно ДВАЖДЫ — numpy и jax, — одинаковой схемой:
    f_x общий множитель обоих уравнений, поэтому считается один раз и
    применяется к склейке [S_theta | S_c]. Разница только в бэкенде.
    """

    def __init__(self, model, method: str = 'RK45'):
        # Удобство ноутбуков и тестов: принимает и сырую ODESystem,
        # и уже скомпилированную модель (тогда компиляция общая)
        self.model = (model if isinstance(model, CompiledModel)
                      else CompiledModel(model))
        self.method = method
        self.ATOL = 1e-5
        self.RTOL = 1e-5
        self._jax_vmap_full = None   # кэш vmap-обёртки батчевого интегратора

    def get_solution(self, c0, theta, t_eval):
        """Интегрирование только состояния (обычный режим)."""
        m = self.model

        def system(t, y):
            return m.f(y, t, theta[:m.n_theta])

        sol = solve_ivp(system, (t_eval[0], t_eval[-1]), c0,
                        t_eval=t_eval, method=self.method,
                        atol=self.ATOL, rtol=self.RTOL)
        if not sol.success:
            raise RuntimeError(f"Интегрирование не сошлось: {sol.message}")
        return sol.y

    def get_jacobian_solution(self, c0, theta, t_eval):
        """Интегрирование расширенной системы (состояние + чувствительности) (обычный режим)."""
        p = self.model.n_theta
        y0 = initial_flat_row(c0, p)

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
        sol = odeint(self.model.f_jax,
                     jnp.array(c0),
                     jnp.array(t_eval),
                     *theta[:self.model.n_theta],
                     rtol=self.RTOL, atol=self.ATOL)
        return np.array(sol).T

    def _vmapped_full_integrator(self):
        """JIT+vmap-обёртка интегратора расширенной системы (кэшируется).

        jax.jit сам перекомпилирует при смене формы (число шутов, длина сетки).
        """
        if self._jax_vmap_full is None:
            n, p = self.model.nx, self.model.n_theta
            # Хвост начальной строки (S_theta = 0, S_c = I) — константа
            sens0 = jnp.array(initial_flat_row(np.zeros(n), p)[n:])

            def integrate_one(c0, t_grid, theta):
                y0 = jnp.concatenate([c0, sens0])
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
        get_jacobian_solution для каждого шута.
        """
        theta_j = jnp.array(np.asarray(theta[:self.model.n_theta], dtype=float))

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
    # Вариационные уравнения — правые части
    # ----------------------------------------------------------------------
    def _variational_rhs(self, y, t, theta):
        """Правая часть расширенной системы (numpy).

        Входные сигналы вычисляются ОДИН раз за вызов: раньше путь шёл через
        f/df_dx/df_dtheta, и пользовательская get_input_signals звалась трижды
        на каждый шаг интегратора.
        """
        m = self.model
        n, p = m.nx, m.n_theta
        x, S_theta, S_c = split_row(y, n, p)

        args = (*x, *m._get_inp_signals(t), *theta)
        dx = np.array(m._f_ca(*args)).ravel()
        f_x = np.array(m._df_dx_ca(*args))
        f_theta = np.array(m._df_dtheta_ca(*args))

        dS = f_x @ np.concatenate([S_theta, S_c], axis=1)
        # layout плоский: сначала весь S_theta, потом весь S_c (C-order каждый),
        # поэтому склейку приходится снова разрезать, а не ravel'ить целиком
        return np.concatenate([dx, (dS[:, :p] + f_theta).ravel(),
                               dS[:, p:].ravel()])

    def _variational_rhs_jax(self, y, t, *theta):
        """Правая часть расширенной системы (jax) — та же схема.

        Как и numpy-версия, зовёт скомпилированные функции напрямую с общим
        args: путь через обёртки f_jax/df_dx_jax/df_dtheta_jax вычислял бы
        входные сигналы трижды за вызов (под jit это цена трассировки, но
        пара версий обязана читаться как одно и то же дважды).
        """
        m = self.model
        n, p = m.nx, m.n_theta
        x, S_theta, S_c = split_row(y, n, p)

        args = (*x, *m._get_inp_signals(t), *theta)
        dx = jnp.array(m._f_jax_ca(*args)[0].flatten())
        f_x = jnp.array(m._df_dx_jax_ca(*args))[0]
        f_theta = jnp.array(m._df_dtheta_jax_ca(*args))[0]

        dS = f_x @ jnp.concatenate([S_theta, S_c], axis=1)
        return jnp.concatenate([dx, (dS[:, :p] + f_theta).flatten(),
                                dS[:, p:].flatten()])


class SystemIntegrator(CompiledModel):
    """Интегрирование с УДЕРЖИВАЕМЫМ входом u, заданным вызывающей стороной.

    Отличие от остальной библиотеки: там входы берутся из модели
    (get_input_signals(t)), а здесь u — аргумент и держится постоянным на
    шаге. Именно это нужно симуляции MPC: вход выдаёт регулятор, а не модель.

    Раньше это был отдельный класс, целиком повторявший компиляцию CasADi,
    причём с ДРУГИМ порядком аргументов — Function объявлялся как
    [state, theta, u], а get_lin_system_dynamics звал его как
    (state, u, theta), то есть подставлял вход в слоты параметров. Теперь
    функции берутся у CompiledModel, и такой рассинхрон невозможен по
    построению.
    """

    def __init__(self, model: ODESystem, method: str = 'RK45'):
        super().__init__(model)
        self.method = method
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
        # t не используется (u удерживается), но обязателен: odeint зовёт
        # правую часть как f(y, t, *args)
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
        return (np.array(self._df_dx_ca(*args)),
                np.array(self._f_u_ca(*args)),
                np.array(self._df_dtheta_ca(*args)))


class SyntheticDataGenerator:
    """
    Генератор синтетических данных для динамической системы.

    Параметры
    ----------
    system : object
        Система с методами:
            - dims() -> (state_len, theta_len, meas_len)
            - get_solution / get_solution_jax (c0, theta, t_eval)
              -> array (state_len, n_t)
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
        self.system = CompiledModel(system_ode)
        self.integrator = VariationalIntegrator(self.system)
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

        # Обе версии интегратора возвращают (state_len, n_measurements)
        integrate = (self.integrator.get_solution_jax if self.use_jax
                     else self.integrator.get_solution)
        solution = integrate(c0_true, theta, t_eval)

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
        self.system = CompiledModel(system_ode)
        self.integrator = VariationalIntegrator(self.system)
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
        # sigma: скаляр (один на все каналы) или вектор (meas_dim,);
        # прежний дефолт [0]*meas_dim падал бы на sigma**2 (list ** int)
        sigma = np.atleast_1d(np.asarray(
            0.0 if sigma is None else sigma, dtype=float))
        if sigma.size == 1:
            sigma = np.full(self.meas_dim, sigma[0])
        # Get control inputs at each time point
        u = np.zeros((len(t), self.control_dim))
        for i, ti in enumerate(t):
            u[i] = self.system.model.get_input_signals(ti)

        # Integrate system to obtain full states
        full_states = self.integrator.get_solution(c0, theta, t).T  # shape (len(t), state_dim)

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
    

def check_system_ok(system_ode: ODESystem):
    """Дешёвая проверка согласованности модели: компиляция + число входов.

    Сама компиляция CompiledModel уже валидирует размерности символов;
    отдельно проверяем только то, что она проверить не может — что
    get_input_signals возвращает ровно nu сигналов.
    """
    system = CompiledModel(system_ode)
    n_inp = len(system_ode.get_input_signals(0))
    if system.nu != n_inp:
        raise ValueError(
            f"{type(system_ode).__name__}: объявлено nu={system.nu}, но "
            f"get_input_signals(t) возвращает {n_inp} сигналов")
    return True