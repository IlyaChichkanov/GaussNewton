# -*- coding: utf-8 -*-
"""Рекурсивный коллокационный интегратор (Радо IIA + IND).

Теория: collocation.ipynb, раздел «Альтернатива: коллокации как неявный
интегратор». Коллокационные уравнения элемента записываются как одношаговая
неявная схема z = A x_prev + h B F(z, theta) (стадийные уравнения Радо IIA),
решаются Ньютоном, а чувствительности решения по theta и по начальному
состоянию накапливаются рекурсиями S_c <- Psi S_c, S_th <- Psi S_th + Gamma,
где Psi и Gamma получаются из того же LU-разложения, что и шаг Ньютона
(internal numerical differentiation — точные производные дискретной схемы).

CollocationSystemJacobian — drop-in замена SystemJacobian: переопределены
только методы интегрирования, контракт выхода (layout строк
[x; J_theta.flatten(); J_c.flatten()]) сохранён, поэтому MultipleShooting,
run_optimization и plot_solution работают без изменений.
"""
import os

import numpy as np
import casadi as ca
from scipy.linalg import lu_factor, lu_solve

from commom_utils.ode_system import SystemJacobian


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
        newton_tol, newton_maxiter: критерий остановки Ньютона по стадиям.
    """

    def __init__(self, f_sym, K=3, n_sub=1, newton_tol=1e-10, newton_maxiter=15,
                 use_compiled=True, n_threads=None, method='RK45'):
        super().__init__(f_sym, method=method)
        self.colloc = RadauTables(K)
        self.n_sub = int(n_sub)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.use_compiled = bool(use_compiled)
        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 1)
        self.n_threads = int(n_threads)

        K_, nx = self.colloc.K, self.nx
        # B (x) I — действие таблицы Бутчера на стековый вектор стадий
        self._B_kron = np.kron(self.colloc.butcher_a, np.eye(nx))
        self._eyeK = np.eye(K_ * nx)
        # Кэш входных сигналов в узлах: ключ — сетка t_eval
        self._node_inp_cache = {}
        # Компилированный путь: шаговые функции (лениво) и кэш mapaccum по N
        self._step_fns = None
        self._mapaccum_cache = {}

    # ------------------------------------------------------------------
    # Входные сигналы в узлах коллокации (предвычисляются на сетку)
    # ------------------------------------------------------------------
    def _node_inputs(self, t_eval):
        """Входы u в узлах всех под-элементов сетки: (N-1, n_sub, K, nu)."""
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
            inp = np.zeros((n_int, self.n_sub, K, 0))
        else:
            try:
                sigs = self.f_sym.get_input_signals(flat_t)
                inp = np.array([np.asarray(s, dtype=float).reshape(flat_t.size)
                                for s in sigs]).T
            except Exception:
                inp = np.array([np.asarray(self._get_inp_signals(t), dtype=float)
                                .reshape(nu) for t in flat_t])
            inp = inp.reshape(n_int, self.n_sub, K, nu)

        self._node_inp_cache[key] = inp
        return inp

    # ------------------------------------------------------------------
    # Один элемент: Ньютон по стадиям (+ опционально Psi/Gamma)
    # ------------------------------------------------------------------
    def _stage_eval(self, z, theta, inp_k, with_fth=False):
        """f, f_x (и f_theta) во всех K стадиях. z — стековый вектор (K*nx,)."""
        K, nx = self.colloc.K, self.nx
        F = np.empty(K * nx)
        fx = np.empty((K, nx, nx))
        fth = np.empty((K, nx, self.np)) if with_fth else None
        for k in range(K):
            xk = z[k * nx:(k + 1) * nx]
            args = (*xk, *inp_k[k], *theta)
            F[k * nx:(k + 1) * nx] = np.array(self.res_f(*args)).ravel()
            fx[k] = np.array(self.compute_jacobian_x(*args))
            if with_fth:
                fth[k] = np.array(self.compute_jacobian_theta(*args)).reshape(nx, self.np)
        return F, fx, fth

    def _newton_matrix(self, h, fx):
        """M = I - h (B (x) I) blkdiag(f_x): блок (k, j) = d_kj I - h a_kj f_x^j."""
        K, nx = self.colloc.K, self.nx
        M = self._eyeK.copy()
        a = self.colloc.butcher_a
        for k in range(K):
            for j in range(K):
                M[k * nx:(k + 1) * nx, j * nx:(j + 1) * nx] -= h * a[k, j] * fx[j]
        return M

    def _element_step(self, x_prev, theta, h, inp_k, with_sens):
        """Шаг по одному элементу.

        Возвращает (x_next, Psi, Gamma); при with_sens=False вместо Psi/Gamma — None.
        """
        K, nx = self.colloc.K, self.nx
        Ax = np.tile(x_prev, K)                       # A = 1 (x) I
        z = Ax.copy()                                 # старт: константа

        converged = False
        for _ in range(self.newton_maxiter):
            F, fx, _ = self._stage_eval(z, theta, inp_k)
            res = z - Ax - h * (self._B_kron @ F)
            M = self._newton_matrix(h, fx)
            dz = lu_solve(lu_factor(M), -res)
            z += dz
            if np.max(np.abs(dz)) < self.newton_tol * (1.0 + np.max(np.abs(z))):
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"Коллокации: Ньютон не сошёлся за {self.newton_maxiter} итераций "
                f"(h={h:.3e}, |dz|={np.max(np.abs(dz)):.3e})")

        if not with_sens:
            return z[-nx:], None, None

        # Psi/Gamma из того же M (в сошедшейся точке), см. collocation.ipynb
        F, fx, fth = self._stage_eval(z, theta, inp_k, with_fth=True)
        lu = lu_factor(self._newton_matrix(h, fx))
        rhs_c = np.tile(np.eye(nx), (K, 1))                       # столбцы A
        rhs_th = h * (self._B_kron @ fth.reshape(K * nx, self.np))
        Psi = lu_solve(lu, rhs_c)[-nx:]
        Gamma = lu_solve(lu, rhs_th)[-nx:]
        return z[-nx:], Psi, Gamma

    # ------------------------------------------------------------------
    # Компилированный шаг элемента: rootfinder + производные через
    # теорему о неявной функции (это в точности Psi/Gamma, но в C++)
    # ------------------------------------------------------------------
    def _build_step_functions(self):
        if self._step_fns is not None:
            return self._step_fns

        K, nx, np_, nu = self.colloc.K, self.nx, self.np, self.nu

        z = ca.MX.sym('z', K * nx)
        x_prev = ca.MX.sym('x_prev', nx)
        theta = ca.MX.sym('theta', np_)
        u_flat = ca.MX.sym('u_flat', K * nu)
        h = ca.MX.sym('h')

        # F(z): правая часть во всех стадиях (SX-функция res_f на MX-аргументах)
        F_stages = []
        for k in range(K):
            xk = [z[k * nx + i] for i in range(nx)]
            uk = [u_flat[k * nu + j] for j in range(nu)]
            th = [theta[i] for i in range(np_)]
            F_stages.append(self.res_f(*xk, *uk, *th))
        F = ca.vcat(F_stages)

        # Резидуал стадийных уравнений: z - A x_prev - h (B (x) I) F(z)
        B_kron = ca.DM(self._B_kron)
        p = ca.vertcat(x_prev, theta, u_flat, h)
        Phi = z - ca.repmat(x_prev, K, 1) - h * ca.mtimes(B_kron, F)
        G = ca.Function('colloc_res', [z, p], [Phi])

        rf = ca.rootfinder('colloc_rf', 'newton', G, {
            'abstol': self.newton_tol,
            'max_iter': self.newton_maxiter,
            'error_on_fail': True,
        })

        z0 = ca.repmat(x_prev, K, 1)          # производная решения по z0 нулевая
        z_sol = rf(z0, p)
        x_next = z_sol[-nx:]
        Psi = ca.jacobian(x_next, x_prev)     # через rootfinder — неявная функция
        Gamma = ca.jacobian(x_next, theta)

        step_sens = ca.Function('colloc_step_sens', [x_prev, theta, u_flat, h],
                                [x_next, Psi, Gamma])
        step_x = ca.Function('colloc_step_x', [x_prev, theta, u_flat, h],
                             [x_next])
        self._step_fns = (step_sens, step_x)
        return self._step_fns

    def _get_mapaccum(self, n_elems, with_sens):
        key = ('accum', n_elems, with_sens)
        fn = self._mapaccum_cache.get(key)
        if fn is None:
            step_sens, step_x = self._build_step_functions()
            base = step_sens if with_sens else step_x
            fn = base.mapaccum(n_elems)
            self._mapaccum_cache[key] = fn
        return fn

    def _march_inputs(self, theta, t_eval):
        """Входы mapaccum (столбец на элемент): theta, u, h."""
        K, nu = self.colloc.K, self.nu
        n_elems = (len(t_eval) - 1) * self.n_sub
        inp_nodes = self._node_inputs(t_eval)                  # (n_int, n_sub, K, nu)
        u_mat = inp_nodes.reshape(n_elems, K * nu).T           # (K*nu, n_elems)
        h_mat = np.repeat((t_eval[1:] - t_eval[:-1]) / self.n_sub,
                          self.n_sub).reshape(1, n_elems)
        theta_mat = np.tile(theta.reshape(-1, 1), (1, n_elems))
        return theta_mat, u_mat, h_mat

    def _newton_fail(self, exc):
        return RuntimeError(
            f"Коллокации: Ньютон по стадиям не сошёлся "
            f"(abstol={self.newton_tol}, max_iter={self.newton_maxiter}); "
            f"уменьшите шаг (n_sub) или ослабьте допуск. CasADi: {exc}")

    def _accumulate_sens(self, c0, x_all, Psi_all, Gamma_all, n_pts):
        """Рекурсии S_c/S_theta по элементам -> выход в layout родителя.

        x_all: (nx, n_elems); Psi_all: (n_elems, nx, nx); Gamma_all: (n_elems, nx, np).
        """
        nx, np_ = self.nx, self.np
        out = np.zeros((nx + nx * np_ + nx * nx, n_pts))
        S_th = np.zeros((nx, np_))
        S_c = np.eye(nx)
        out[:, 0] = np.concatenate([np.asarray(c0, float), S_th.ravel(), S_c.ravel()])
        col = 1
        for e in range(x_all.shape[1]):
            S_c = Psi_all[e] @ S_c
            S_th = Psi_all[e] @ S_th + Gamma_all[e]
            if (e + 1) % self.n_sub == 0:
                out[:, col] = np.concatenate([x_all[:, e], S_th.ravel(), S_c.ravel()])
                col += 1
        return out

    def _march_compiled(self, c0, theta, t_eval, with_sens):
        nx, np_ = self.nx, self.np
        n_pts = len(t_eval)
        n_elems = (n_pts - 1) * self.n_sub
        theta_mat, u_mat, h_mat = self._march_inputs(theta, t_eval)

        march = self._get_mapaccum(n_elems, with_sens)
        try:
            res = march(c0, theta_mat, u_mat, h_mat)
        except RuntimeError as exc:
            raise self._newton_fail(exc) from exc

        if not with_sens:
            x_all = np.array(res)                              # (nx, n_elems)
            out = np.empty((nx, n_pts))
            out[:, 0] = c0
            out[:, 1:] = x_all[:, self.n_sub - 1::self.n_sub]
            return out

        x_all = np.array(res[0])                               # (nx, n_elems)
        # mapaccum стыкует блочные выходы горизонтально: (nx, nx*n_elems)
        Psi_all = np.array(res[1]).reshape(nx, n_elems, nx).transpose(1, 0, 2)
        Gamma_all = np.array(res[2]).reshape(nx, n_elems, np_).transpose(1, 0, 2)
        return self._accumulate_sens(c0, x_all, Psi_all, Gamma_all, n_pts)

    def _get_mapmarch(self, n_elems, group_size):
        """map('thread') поверх mapaccum: параллельный марш группы шутов."""
        key = ('map', n_elems, group_size)
        fn = self._mapaccum_cache.get(key)
        if fn is None:
            march = self._get_mapaccum(n_elems, with_sens=True)
            fn = march.map(group_size, 'thread', min(self.n_threads, group_size))
            self._mapaccum_cache[key] = fn
        return fn

    def _march_batch_compiled(self, c0_list, theta, t_grids):
        """Все шуты батча: группировка по длине сетки + потоковый map."""
        nx, np_ = self.nx, self.np
        groups = {}
        for i, g in enumerate(t_grids):
            groups.setdefault(len(g), []).append(i)

        results = [None] * len(t_grids)
        for L, idxs in groups.items():
            n_pts = L
            n_elems = (L - 1) * self.n_sub
            if len(idxs) == 1 or self.n_threads == 1:
                for i in idxs:
                    results[i] = self._march_compiled(
                        np.asarray(c0_list[i], float), theta,
                        np.asarray(t_grids[i], float), with_sens=True)
                continue

            x0_mat = np.column_stack([np.asarray(c0_list[i], float) for i in idxs])
            th_l, u_l, h_l = zip(*(self._march_inputs(theta,
                                                      np.asarray(t_grids[i], float))
                                   for i in idxs))
            marchmap = self._get_mapmarch(n_elems, len(idxs))
            try:
                res = marchmap(x0_mat, np.hstack(th_l), np.hstack(u_l), np.hstack(h_l))
            except RuntimeError as exc:
                raise self._newton_fail(exc) from exc

            x_st = np.array(res[0])       # (nx, G*n_elems), блоки шутов подряд
            Psi_st = np.array(res[1])     # (nx, G*n_elems*nx)
            Gam_st = np.array(res[2])     # (nx, G*n_elems*np)
            for j, i in enumerate(idxs):
                x_all = x_st[:, j * n_elems:(j + 1) * n_elems]
                Psi = Psi_st[:, j * n_elems * nx:(j + 1) * n_elems * nx] \
                    .reshape(nx, n_elems, nx).transpose(1, 0, 2)
                Gam = Gam_st[:, j * n_elems * np_:(j + 1) * n_elems * np_] \
                    .reshape(nx, n_elems, np_).transpose(1, 0, 2)
                results[i] = self._accumulate_sens(
                    np.asarray(c0_list[i], float), x_all, Psi, Gam, n_pts)
        return results

    # ------------------------------------------------------------------
    # Марш по сетке
    # ------------------------------------------------------------------
    def _march(self, c0, theta, t_eval, with_sens):
        t_eval = np.asarray(t_eval, dtype=float)
        theta = np.asarray(theta, dtype=float)[:self.np]
        if self.use_compiled:
            return self._march_compiled(np.asarray(c0, dtype=float), theta,
                                        t_eval, with_sens)
        nx, np_ = self.nx, self.np
        n_pts = len(t_eval)
        inp_nodes = self._node_inputs(t_eval)

        x = np.asarray(c0, dtype=float).copy()
        if with_sens:
            S_th = np.zeros((nx, np_))
            S_c = np.eye(nx)
            out = np.zeros((nx + nx * np_ + nx * nx, n_pts))
            out[:, 0] = np.concatenate([x, S_th.ravel(), S_c.ravel()])
        else:
            out = np.zeros((nx, n_pts))
            out[:, 0] = x

        for i in range(n_pts - 1):
            h = (t_eval[i + 1] - t_eval[i]) / self.n_sub
            for s in range(self.n_sub):
                x, Psi, Gamma = self._element_step(x, theta, h, inp_nodes[i, s],
                                                   with_sens)
                if with_sens:
                    S_c = Psi @ S_c
                    S_th = Psi @ S_th + Gamma
            if with_sens:
                out[:, i + 1] = np.concatenate([x, S_th.ravel(), S_c.ravel()])
            else:
                out[:, i + 1] = x
        return out

    # ------------------------------------------------------------------
    # Переопределённые интеграторы (контракт SystemJacobian сохранён)
    # ------------------------------------------------------------------
    def get_jacobian_solution(self, c0, theta, t_eval):
        """Состояние + чувствительности: строки [x; J_th.flatten(); J_c.flatten()]."""
        return self._march(c0, theta, t_eval, with_sens=True)

    def get_solution(self, c0, theta, t_eval):
        """Только состояние (nx, N)."""
        return self._march(c0, theta, t_eval, with_sens=False)

    # JAX-варианты маршрутизируются на ту же реализацию: для коллокационного
    # интегратора флаг use_jax не имеет значения
    def get_jacobian_solution_jax(self, c0, theta, t_eval):
        return self._march(c0, theta, t_eval, with_sens=True)

    def get_jacobian_solution_jax_batch(self, c0_list, theta, t_grids):
        theta = np.asarray(theta, dtype=float)[:self.np]
        if self.use_compiled:
            return self._march_batch_compiled(c0_list, theta, t_grids)
        return [self._march(c0, theta, tg, with_sens=True)
                for c0, tg in zip(c0_list, t_grids)]

    def get_solution_jax(self, c0, theta, t_eval):
        return self._march(c0, theta, t_eval, with_sens=False)
