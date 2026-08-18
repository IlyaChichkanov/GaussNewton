# -*- coding: utf-8 -*-
"""Коллокационный интегратор (Радо IIA + IND) на CasADi.

Теория: collocation.ipynb, раздел «Альтернатива: коллокации как неявный
интегратор». Коллокационные уравнения элемента записываются как одношаговая
неявная схема z = A x_prev + h B F(z, theta) (стадийные уравнения Радо IIA)
и решаются CasADi-rootfinder'ом (Ньютон с line search и точным якобианом).
Чувствительности решения по theta и по начальному состоянию — производные
выхода rootfinder'а по теореме о неявной функции (internal numerical
differentiation, точные производные дискретной схемы); по элементам они
накапливаются рекурсиями S_c <- Psi S_c, S_th <- Psi S_th + Gamma.

CollocationSystemJacobian — drop-in замена SystemJacobian: переопределены
только методы интегрирования, контракт выхода (layout строк
[x; J_theta.flatten(); J_c.flatten()]) сохранён, поэтому MultipleShooting,
run_optimization_adaptive и plot_solution работают без изменений.

Несходимость Ньютона НЕ бросает исключений внутри CasADi (error_on_fail
отключён — иначе C++ печатает многострочные дампы входов): каждый элемент
дополнительно возвращает масштабированную невязку в найденной точке, и марш
проверяет её сам, поднимая обычный RuntimeError одной строкой. В пробной
точке оптимизации это штатный отказ шага — цикл его перехватывает и откатывает.
"""
import os

import numpy as np
import casadi as ca

from commom_utils.ode_system import SystemJacobian
from commom_utils.sensitivity import group_by_grid_length


class RadauTables:
    """Таблицы коллокаций Радо IIA: узлы, матрица дифференцирования, Бутчер.

    Полином степени K строится по точкам {0, tau_1..tau_K}; tilde_D — матрица
    производных базиса в узлах коллокации, разбитая на столбец левого края d0
    и невырожденный блок D1. Таблица Бутчера a = D1^{-1}.
    """

    _NODES = {
        1: [1.0],
        2: [1.0 / 3.0, 1.0],
        3: [(4.0 - np.sqrt(6.0)) / 10.0, (4.0 + np.sqrt(6.0)) / 10.0, 1.0],
    }

    def __init__(self, K=3):
        if K not in self._NODES:
            raise ValueError(f"K={K} не поддерживается (доступно {sorted(self._NODES)})")
        self.K = K
        self.tau = np.array(self._NODES[K])
        self.nodes = np.concatenate([[0.0], self.tau])   # интерполяционные точки

        tilde_D = self._differentiation_matrix(self.nodes, self.tau)  # K x (K+1)
        self.d0 = tilde_D[:, 0]
        self.D1 = tilde_D[:, 1:]
        self.butcher_a = np.linalg.inv(self.D1)

        # Самопроверки (см. collocation.ipynb): точность на константах
        # даёт нулевые суммы строк и A = 1 (x) I
        assert np.allclose(tilde_D.sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose(-self.butcher_a @ self.d0, np.ones(K), atol=1e-12)

    @staticmethod
    def _differentiation_matrix(nodes, at):
        """D[k, j] = dl_j/dtau в точках at[k]."""
        n = len(nodes)
        D = np.zeros((len(at), n))
        for k, x in enumerate(at):
            for j in range(n):
                s = 0.0
                for m in range(n):
                    if m == j:
                        continue
                    prod = 1.0 / (nodes[j] - nodes[m])
                    for r in range(n):
                        if r != j and r != m:
                            prod *= (x - nodes[r]) / (nodes[j] - nodes[r])
                    s += prod
                D[k, j] = s
        return D


class CollocationSystemJacobian(SystemJacobian):
    """SystemJacobian с коллокационным интегратором вместо solve_ivp/odeint.

    Параметры:
        K: число стадий Радо IIA (порядок 2K-1).
        n_sub: число элементов на интервал между соседними точками t_eval.
        newton_tol: допуск Ньютона по стадиям — и на невязку max|Phi| (abstol),
            и на шаг max|dz| (abstolStep); критерии работают как ИЛИ.
        newton_maxiter: жёсткий предел итераций Ньютона.
        rootfinder_plugin: плагин ca.rootfinder ('newton'; установлены также
            'kinsol', 'fast_newton').
        rootfinder_options: dict, переопределяющий любые опции rootfinder
            (у kinsol свои имена опций — этот словарь и есть путь их задать).
    """

    # Допуск проверки сходимости: масштабированная невязка элемента
    # не должна превышать RES_SAFETY * newton_tol (запас на остановку по шагу)
    RES_SAFETY = 10.0

    def __init__(self, model, K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=25,
                 rootfinder_plugin='newton', rootfinder_options=None,
                 n_threads=None, method='RK45'):
        super().__init__(model, method=method)
        if int(n_sub) < 1:
            raise ValueError(f"n_sub={n_sub}: должно быть >= 1")
        if float(newton_tol) <= 0:
            raise ValueError(f"newton_tol={newton_tol}: должен быть > 0")
        if int(newton_maxiter) < 1:
            raise ValueError(f"newton_maxiter={newton_maxiter}: должно быть >= 1")
        self.colloc = RadauTables(K)
        self.n_sub = int(n_sub)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.rootfinder_plugin = rootfinder_plugin
        self.rootfinder_options = dict(rootfinder_options or {})
        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 1)
        self.n_threads = int(n_threads)

        # B (x) I — действие таблицы Бутчера на стековый вектор стадий
        self._B_kron = np.kron(self.colloc.butcher_a, np.eye(self.nx))
        # Кэши: входные сигналы на сетке, шаговые функции (лениво),
        # mapaccum по числу элементов, map('thread') по размеру группы шутов
        self._node_inp_cache = {}
        self._step_fns = None
        self._accum_cache = {}
        self._map_cache = {}

    # ------------------------------------------------------------------
    # Помощники, общие для одиночного и батчевого марша
    # ------------------------------------------------------------------
    def _pack_row(self, x, S_th, S_c):
        """Столбец выхода в контракте SystemJacobian.

        Layout (C-order): [x (nx); J_theta.flatten() (nx*np); J_c.flatten()
        (nx*nx)] — ровно то, что ждёт SensitivityTrajectory.unpack.
        """
        return np.concatenate([np.asarray(x, float), S_th.ravel(), S_c.ravel()])

    def _accumulate_sens(self, c0, x_all, Psi_all, Gamma_all, n_pts):
        """Рекурсии чувствительностей по элементам -> выход в layout родителя.

        S_c <- Psi_e S_c (S_c[0] = I), S_th <- Psi_e S_th + Gamma_e (S_th[0]=0).
        x_all: (nx, n_elems); Psi_all: (n_elems, nx, nx);
        Gamma_all: (n_elems, nx, np). Точки t_eval — концы каждого n_sub-го
        элемента, поэтому столбец пишется раз в n_sub элементов.
        """
        nx, n_theta = self.nx, self.n_theta
        out = np.zeros((nx + nx * n_theta + nx * nx, n_pts))
        S_th = np.zeros((nx, n_theta))
        S_c = np.eye(nx)
        out[:, 0] = self._pack_row(c0, S_th, S_c)
        col = 1
        for e in range(x_all.shape[1]):
            S_c = Psi_all[e] @ S_c
            S_th = Psi_all[e] @ S_th + Gamma_all[e]
            if (e + 1) % self.n_sub == 0:
                out[:, col] = self._pack_row(x_all[:, e], S_th, S_c)
                col += 1
        return out

    def _node_inputs(self, t_eval):
        """Входы u в узлах всех под-элементов сетки: (N-1, n_sub, K, nu)."""
        # Массив нехешируем, поэтому ключ — его байты; безопасно, т.к. сюда
        # t_eval всегда приходит уже нормализованным (asarray(..., float))
        key = (t_eval.tobytes(), self.n_sub)
        cached = self._node_inp_cache.get(key)
        if cached is not None:
            return cached

        K, nu = self.colloc.K, self.nu
        n_int = len(t_eval) - 1
        h_int = (t_eval[1:] - t_eval[:-1]) / self.n_sub          # (n_int,)
        # t_nodes[i, s, k] = t_eval[i] + (s + tau_k) * h_int[i]
        offs = (np.arange(self.n_sub)[:, None] + self.colloc.tau[None, :])
        t_nodes = t_eval[:-1, None, None] + offs[None, :, :] * h_int[:, None, None]
        flat_t = t_nodes.ravel()

        if nu == 0:
            # Автономная система: пустой массив с правильной формой, чтобы
            # дальнейшие reshape по осям (n_int, n_sub, K, nu) не ветвились
            inp = np.zeros((n_int, self.n_sub, K, 0))
        else:
            try:
                # Быстрый путь: get_input_signals векторизован по времени
                sigs = self.model.get_input_signals(flat_t)
                inp = np.array([np.asarray(s, dtype=float).reshape(flat_t.size)
                                for s in sigs]).T
            except Exception:
                # Fallback: по точке за раз (семантически эквивалентен,
                # просто медленнее — для входов, не принимающих массив t)
                inp = np.array([np.asarray(self._get_inp_signals(t), dtype=float)
                                .reshape(nu) for t in flat_t])
            inp = inp.reshape(n_int, self.n_sub, K, nu)

        self._node_inp_cache[key] = inp
        return inp

    # ------------------------------------------------------------------
    # Шаг элемента: rootfinder по стадиям + Psi/Gamma через теорему
    # о неявной функции (ca.jacobian дифференцирует сквозь rootfinder)
    # ------------------------------------------------------------------
    def _build_step_functions(self):
        if self._step_fns is not None:
            return self._step_fns

        K, nx, n_theta, nu = self.colloc.K, self.nx, self.n_theta, self.nu

        z = ca.MX.sym('z', K * nx)
        x_prev = ca.MX.sym('x_prev', nx)
        theta = ca.MX.sym('theta', n_theta)
        u_flat = ca.MX.sym('u_flat', K * nu)
        h = ca.MX.sym('h')

        # F(z): правая часть во всех стадиях (SX-функция _f_ca на MX-аргументах)
        F_stages = []
        for k in range(K):
            xk = [z[k * nx + i] for i in range(nx)]
            uk = [u_flat[k * nu + j] for j in range(nu)]
            th = [theta[i] for i in range(n_theta)]
            F_stages.append(self._f_ca(*xk, *uk, *th))
        F = ca.vcat(F_stages)

        # Резидуал стадийных уравнений: Phi = z - A x_prev - h (B (x) I) F(z)
        B_kron = ca.DM(self._B_kron)
        step_param = ca.vertcat(x_prev, theta, u_flat, h)
        Phi = z - ca.repmat(x_prev, K, 1) - h * ca.mtimes(B_kron, F)
        phi_fn = ca.Function('colloc_res', [z, step_param], [Phi])

        rf_opts = {
            'abstol': self.newton_tol,       # невязка: max|Phi| < tol
            'abstolStep': self.newton_tol,   # шаг: max|dz| < tol (критерии - ИЛИ);
                                             # спасает большие масштабы состояний,
                                             # где abstol упирается в floor округления
            'max_iter': self.newton_maxiter,
            'line_search': True,
            'error_on_fail': False,          # тихо: сходимость проверяет марш
                                             # по stage_res, без C++ дампов
        }
        rf_opts.update(self.rootfinder_options)
        stage_solver = ca.rootfinder('colloc_rf', self.rootfinder_plugin,
                                     phi_fn, rf_opts)

        z0 = ca.repmat(x_prev, K, 1)          # производная решения по z0 нулевая
        z_sol = stage_solver(z0, step_param)
        # Радо IIA stiffly accurate (tau_K = 1): последний узел совпадает
        # с правым краем элемента, поэтому x_next — последний стадийный блок
        x_next = z_sol[-nx:]
        Psi = ca.jacobian(x_next, x_prev)     # через rootfinder — неявная функция
        Gamma = ca.jacobian(x_next, theta)
        # Масштабированная невязка в найденной точке — маркер сходимости;
        # при расходимости (inf/nan в z) она тоже inf/nan
        stage_res = ca.norm_inf(phi_fn(z_sol, step_param)) \
            / (1.0 + ca.norm_inf(z_sol))

        step_sens = ca.Function('colloc_step_sens', [x_prev, theta, u_flat, h],
                                [x_next, Psi, Gamma, stage_res])
        step_x = ca.Function('colloc_step_x', [x_prev, theta, u_flat, h],
                             [x_next, stage_res])
        self._step_fns = (step_sens, step_x)
        return self._step_fns

    def _get_mapaccum(self, n_elems, with_sens):
        """Марш по n_elems элементам одного шута (аккумулируется только x)."""
        key = (n_elems, with_sens)
        fn = self._accum_cache.get(key)
        if fn is None:
            step_sens, step_x = self._build_step_functions()
            base = step_sens if with_sens else step_x
            fn = base.mapaccum(n_elems)
            self._accum_cache[key] = fn
        return fn

    def _get_mapmarch(self, n_elems, group_size):
        """map('thread') поверх mapaccum: параллельный марш группы шутов."""
        key = (n_elems, group_size)
        fn = self._map_cache.get(key)
        if fn is None:
            march = self._get_mapaccum(n_elems, with_sens=True)
            fn = march.map(group_size, 'thread', min(self.n_threads, group_size))
            self._map_cache[key] = fn
        return fn

    def _march_inputs(self, theta, t_eval):
        """Входы mapaccum (столбец на элемент): theta, u, h."""
        K, nu = self.colloc.K, self.nu
        n_elems = (len(t_eval) - 1) * self.n_sub
        inp_nodes = self._node_inputs(t_eval)                  # (n_int, n_sub, K, nu)
        u_cols = inp_nodes.reshape(n_elems, K * nu).T          # (K*nu, n_elems)
        h_cols = np.repeat((t_eval[1:] - t_eval[:-1]) / self.n_sub,
                           self.n_sub).reshape(1, n_elems)
        theta_cols = np.tile(theta.reshape(-1, 1), (1, n_elems))
        return theta_cols, u_cols, h_cols

    def _unstack_mapaccum(self, x_stack, Psi_stack, Gamma_stack, n_elems):
        """Разбор выходов mapaccum одного шута.

        mapaccum стыкует матричные выходы элементов горизонтально:
        Psi_stack = [Psi_1 | Psi_2 | ...] размера (nx, n_elems*nx), аналогично
        Gamma. Возвращает (x_all (nx, n_elems), Psi_all (n_elems, nx, nx),
        Gamma_all (n_elems, nx, np)).
        """
        nx, n_theta = self.nx, self.n_theta
        x_all = np.asarray(x_stack)
        Psi_all = np.asarray(Psi_stack).reshape(nx, n_elems, nx).transpose(1, 0, 2)
        Gamma_all = np.asarray(Gamma_stack).reshape(nx, n_elems, n_theta) \
            .transpose(1, 0, 2)
        return x_all, Psi_all, Gamma_all

    def _check_converged(self, stage_res):
        """Проверка сходимости Ньютона по всем элементам (тихая замена
        error_on_fail: без исключений из C++ и без дампов CasADi)."""
        stage_res = np.asarray(stage_res, float).ravel()
        if stage_res.size == 0:
            return
        # nan -> inf, чтобы и расходимость, и NaN попали в ветку отказа
        clean = np.nan_to_num(stage_res, nan=np.inf)
        worst = float(np.max(clean))
        limit = self.RES_SAFETY * self.newton_tol
        if worst > limit:
            n_bad = int(np.sum(clean > limit))
            raise RuntimeError(
                f"Коллокации: Ньютон по стадиям не сошёлся в {n_bad} из "
                f"{stage_res.size} элементов (max масштабированной невязки "
                f"{worst:.2e} > {limit:.1e}; max_iter={self.newton_maxiter}). "
                f"Увеличьте n_sub (шаг элемента h = dt/n_sub станет меньше), "
                f"ослабьте newton_tol или поднимите newton_maxiter. В пробной "
                f"точке оптимизации это штатный отказ шага — цикл его откатит.")

    # ------------------------------------------------------------------
    # Марш по сетке: один шут и батч шутов
    # ------------------------------------------------------------------
    def _march_compiled(self, c0, theta, t_eval, with_sens):
        nx = self.nx
        n_pts = len(t_eval)
        n_elems = (n_pts - 1) * self.n_sub
        theta_cols, u_cols, h_cols = self._march_inputs(theta, t_eval)

        march = self._get_mapaccum(n_elems, with_sens)

        if not with_sens:
            x_stack, stage_res = march(c0, theta_cols, u_cols, h_cols)
            self._check_converged(stage_res)
            out = np.empty((nx, n_pts))
            out[:, 0] = c0
            # mapaccum отдаёт состояние после КАЖДОГО элемента; точки t_eval —
            # концы каждого n_sub-го
            out[:, 1:] = np.asarray(x_stack)[:, self.n_sub - 1::self.n_sub]
            return out

        x_stack, Psi_stack, Gamma_stack, stage_res = march(
            c0, theta_cols, u_cols, h_cols)
        self._check_converged(stage_res)
        x_all, Psi_all, Gamma_all = self._unstack_mapaccum(
            x_stack, Psi_stack, Gamma_stack, n_elems)
        return self._accumulate_sens(c0, x_all, Psi_all, Gamma_all, n_pts)

    def _march_batch_compiled(self, c0_list, theta, t_grids):
        """Все шуты батча: группировка по длине сетки + потоковый map."""
        nx, n_theta = self.nx, self.n_theta

        results = [None] * len(t_grids)
        for idxs in group_by_grid_length(t_grids):
            n_pts = len(t_grids[idxs[0]])
            n_elems = (n_pts - 1) * self.n_sub
            if len(idxs) == 1 or self.n_threads == 1:
                for i in idxs:
                    results[i] = self._march_compiled(
                        np.asarray(c0_list[i], float), theta,
                        np.asarray(t_grids[i], float), with_sens=True)
                continue

            x0_mat = np.column_stack([np.asarray(c0_list[i], float) for i in idxs])
            theta_cols, u_cols, h_cols = zip(
                *(self._march_inputs(theta, np.asarray(t_grids[i], float))
                  for i in idxs))
            marchmap = self._get_mapmarch(n_elems, len(idxs))
            x_stack, Psi_stack, Gamma_stack, stage_res = marchmap(
                x0_mat, np.hstack(theta_cols), np.hstack(u_cols),
                np.hstack(h_cols))
            self._check_converged(stage_res)

            # map стыкует выходы шутов горизонтально: блок шута j — столбцы
            # [j*n_elems*(...), (j+1)*n_elems*(...))
            x_stack = np.asarray(x_stack)         # (nx, G*n_elems)
            Psi_stack = np.asarray(Psi_stack)     # (nx, G*n_elems*nx)
            Gamma_stack = np.asarray(Gamma_stack)  # (nx, G*n_elems*np)
            for j, i in enumerate(idxs):
                x_all, Psi_all, Gamma_all = self._unstack_mapaccum(
                    x_stack[:, j * n_elems:(j + 1) * n_elems],
                    Psi_stack[:, j * n_elems * nx:(j + 1) * n_elems * nx],
                    Gamma_stack[:, j * n_elems * n_theta:(j + 1) * n_elems * n_theta],
                    n_elems)
                results[i] = self._accumulate_sens(
                    np.asarray(c0_list[i], float), x_all, Psi_all, Gamma_all,
                    n_pts)
        return results

    def _march(self, c0, theta, t_eval, with_sens):
        """Нормализация входов + вырожденные сетки + запуск марша."""
        c0 = np.asarray(c0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)
        theta = np.asarray(theta, dtype=float)[:self.n_theta]
        if len(t_eval) < 2:                    # нет ни одного элемента
            nx, n_theta = self.nx, self.n_theta
            if with_sens:
                out = np.zeros((nx + nx * n_theta + nx * nx, len(t_eval)))
                if len(t_eval) == 1:
                    out[:, 0] = self._pack_row(c0, np.zeros((nx, n_theta)),
                                               np.eye(nx))
            else:
                out = np.tile(c0.reshape(-1, 1), (1, len(t_eval)))
            return out
        return self._march_compiled(c0, theta, t_eval, with_sens)

    # ------------------------------------------------------------------
    # Переопределённые интеграторы (контракт SystemJacobian сохранён)
    # ------------------------------------------------------------------
    def get_jacobian_solution(self, c0, theta, t_eval):
        """Состояние + чувствительности: строки [x; J_th.flatten(); J_c.flatten()]."""
        return self._march(c0, theta, t_eval, with_sens=True)

    def get_solution(self, c0, theta, t_eval):
        """Только состояние (nx, N)."""
        return self._march(c0, theta, t_eval, with_sens=False)

    # jax-имена маршрутизируются на ту же реализацию: для коллокационного
    # интегратора флаг use_jax не имеет значения
    def get_jacobian_solution_jax_batch(self, c0_list, theta, t_grids):
        theta = np.asarray(theta, dtype=float)[:self.n_theta]
        return self._march_batch_compiled(c0_list, theta, t_grids)

    def get_solution_jax(self, c0, theta, t_eval):
        return self._march(c0, theta, t_eval, with_sens=False)
