import time

import numpy as np
from commom_utils.ode_system import ODESystem, SystemJacobian
from scipy.sparse import block_diag, csr_matrix, vstack, hstack, eye as speye, diags
from scipy.sparse.linalg import spsolve, splu
from scipy import stats

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

    def __init__(self, system: ODESystem, N_shoot: int, gamma: np.ndarray = None,
                 c0_cost: float = 1, use_jax: bool = False, verbose: bool = False):
        self.system = SystemJacobian(system)
        self.N_shoot = N_shoot
        self.gamma = gamma
        self.c0_cost = c0_cost
        self.use_jax = use_jax
        self.verbose = verbose

        # Data storage
        self.state_measured_batches = []
        self.t_eval_measurements_batches = []
        self.interval_managers = []

    def add_batch(self, state_measured, t_eval_measurements):
        self.state_measured_batches.append(state_measured)
        self.t_eval_measurements_batches.append(t_eval_measurements)
        self.interval_managers.append(TimeIntervalManager(self.N_shoot, t_eval_measurements))


    def make_full_theta(self, theta0, c0_guess = None, c0_init_method='inverse_h', n_iter=1):
        theta_full = np.copy(theta0)
        n_state = self.system.nx

        for state_measured, t_meas, tm in zip(self.state_measured_batches,
                                              self.t_eval_measurements_batches,
                                              self.interval_managers):
            for shoot in range(tm.N_shoot):
                idx = tm.shoot_indexes[shoot]
                y_first = state_measured[idx]
                t_first = t_meas[idx]
                
                if c0_init_method == 'zeros':
                    c0_ = np.zeros(n_state)
                elif c0_init_method == 'measurement_pad':
                    c0_ = np.zeros(n_state)
                    c0_[:n_state] = y_first[:n_state]
                elif c0_init_method == 'inverse_h':
                    x0_guess = np.zeros(n_state) if c0_guess is None or np.any(np.isnan(c0_guess)) else c0_guess
                    
                    c0_ = self.system.inverse_h(y_first, t_first, theta0, 
                                                x_guess=x0_guess, n_iter=n_iter)
                else:
                    raise ValueError(f"Unknown c0_init_method: {c0_init_method}")
                
                theta_full = np.concatenate((theta_full, c0_))
        return theta_full

    def _concatenate_jacobians(self, J1, J2):
        """Совместимость с прежним поэтапным интерфейсом склейки."""
        return self._concatenate_jacobian_batches([J1, J2])

    def _concatenate_jacobian_batches(self, jacobians):
        """Собирает локальные якобианы батчей за один проход.

        Первые ``n_theta`` столбцов общие для всех батчей, а оставшиеся
        столбцы начальных состояний независимы. Поэтому итоговая матрица
        состоит из вертикально склеенного блока параметров и блочно-
        диагонального блока начальных состояний.
        """
        if not jacobians:
            raise ValueError("At least one batch Jacobian is required")

        _, n_theta, _ = self.system.get_dimentions()
        theta_blocks = [jacobian[:, :n_theta] for jacobian in jacobians]
        c0_blocks = [jacobian[:, n_theta:] for jacobian in jacobians]

        theta_block = vstack(theta_blocks, format='csr')
        c0_block = block_diag(c0_blocks, format='csr')
        return hstack([theta_block, c0_block], format='csr')

    def solve(self, theta_full):
        solve_start = time.perf_counter()

        J_batches = []
        J_G_batches = []
        R_batches = []
        R_G_batches = []

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches, self.t_eval_measurements_batches)):
            batch_start = time.perf_counter()
            J_batch, J_G_batch, R_batch, R_G_batch = self._solve_batch(
                theta_full, state_measured, t_meas, batch
            )
            if self.verbose:
                print(f'  Batch {batch}: {time.perf_counter() - batch_start:.3f}s')

            J_batches.append(J_batch)
            J_G_batches.append(J_G_batch)
            R_batches.append(R_batch)
            R_G_batches.append(R_G_batch)

        J_total = self._concatenate_jacobian_batches(J_batches)
        J_G_total = self._concatenate_jacobian_batches(J_G_batches)
        R_total = np.concatenate(R_batches)
        R_G_total = np.concatenate(R_G_batches)

        if self.verbose:
            print(f'Solve total: {time.perf_counter() - solve_start:.3f}s | '
                  f'J: {J_total.shape}, J_G: {J_G_total.shape}')
        return J_total, R_total, J_G_total, R_G_total

    def _solve_batch(self, theta_full, state_measured, t_meas, batch_idx):
        n_state, n_theta, n_obs = self.system.get_dimentions()
        if self.gamma is not None and len(self.gamma) != n_obs:
            raise ValueError(f"gamma length must be {n_obs}, got {len(self.gamma)}")
        idx_theta = slice(0, n_state * n_theta)
        idx_c = slice(n_state * n_theta, n_state * (n_theta + n_state))
        tm = self.interval_managers[batch_idx]
        n_shoot = tm.N_shoot
        theta = theta_full[:n_theta]

        intervals = [tm.get_time_interval(sh) for sh in range(n_shoot)]

        # Смещение c-блока данного батча в theta_full: по фактическому числу
        # шутов предыдущих батчей (оно может отличаться от запрошенного N_shoot)
        c_offset = n_theta + sum(im.N_shoot for im in self.interval_managers[:batch_idx]) * n_state

        # Плотные блоки якобиана: θ-часть общая для всех строк, c-часть
        # блочно-диагональна по шутам — итоговая J собирается одним hstack.
        J_theta_rows = []      # (n_obs, n_theta) на каждое измерение
        J_c_shoot_blocks = []  # вертикальный стек c-блоков каждого шута
        R_blocks = []

        J_G_theta_blocks = []  # (n_state, n_theta) на каждое ограничение
        Jc_prev_blocks = []
        R_G = np.zeros((n_shoot - 1) * n_state)

        Jx_prev = None
        Jc_prev = None
        state_prev = None

        for shoot, (t_interval, meas_idx) in enumerate(intervals):
            start_idx = c_offset + shoot * n_state
            c0 = theta_full[start_idx: start_idx + n_state]

            if self.use_jax:
                sol = self.system.get_jacobian_solution_jax(c0, theta, t_interval)
            else:
                sol = self.system.get_jacobian_solution(c0, theta, t_interval)

            state_traj = sol[:n_state, :]
            J_raw = sol[n_state:, :]

            J_c_rows = []
            for i, idx in enumerate(meas_idx):
                state = state_traj[:, i]
                t = t_interval[i]

                dh_dx = self.system.dh_dx(state, t, theta)
                dh_dtheta = self.system.dh_dtheta(state, t, theta)

                dx_dtheta = J_raw[idx_theta, i].reshape(n_state, n_theta)
                dx_dc = J_raw[idx_c, i].reshape(n_state, n_state)

                # Вес строки: gamma по компонентам наблюдения,
                # c0_cost — дополнительный вес первой точки интервала
                w = self.gamma if self.gamma is not None else np.ones(n_obs)
                if i == 0:
                    w = w * self.c0_cost

                J_theta_rows.append(w[:, None] * (dh_dx @ dx_dtheta + dh_dtheta))
                J_c_rows.append(w[:, None] * (dh_dx @ dx_dc))
                R_blocks.append(w * (state_measured[idx] - self.system.h_x(state, t, theta)))

            J_c_shoot_blocks.append(np.vstack(J_c_rows))

            # Ограничения непрерывности
            if shoot > 0:
                row_idx = (shoot - 1) * n_state
                J_G_theta_blocks.append(Jx_prev)
                Jc_prev_blocks.append(Jc_prev)
                R_G[row_idx:row_idx + n_state] = -(state_prev - c0)

            Jx_prev = J_raw[idx_theta, -1].reshape(n_state, n_theta)
            Jc_prev = J_raw[idx_c, -1].reshape(n_state, n_state)
            state_prev = state_traj[:, -1]

        J = hstack([
            csr_matrix(np.vstack(J_theta_rows)),
            block_diag(J_c_shoot_blocks, format='csr'),
        ], format='csr')
        R = np.concatenate(R_blocks)

        n_cont = (n_shoot - 1) * n_state
        if n_cont > 0:
            # c-часть J_G блочно-бидиагональна: Jc_prev в колонке шута j-1, -I в колонке шута j
            zeros_col = csr_matrix((n_cont, n_state))
            lower = hstack([block_diag(Jc_prev_blocks), zeros_col])
            upper = hstack([zeros_col, -speye(n_cont)])
            J_G = hstack([csr_matrix(np.vstack(J_G_theta_blocks)), lower + upper], format='csr')
        else:
            J_G = csr_matrix((0, n_theta + n_shoot * n_state))
        return J, J_G, R, R_G


def compute_parameter_covariance(J, R, J_G, R_G, n_theta):
    """Маргинальная ковариация θ по полной системе невязок.

    Согласно теории (theory_gauss_newton.ipynb): J_full = [J_meas; J_cont],
    R_full = [R_meas; R_cont], Cov(p) = σ² (J_fullᵀ J_full)⁻¹, dof = n_rows − n_params.
    Возвращается θ-блок обратной матрицы (Шур-комплемент по блоку начальных
    состояний), т.е. корреляция θ с c_j учитывается, а не отбрасывается.
    """
    if J_G is not None and J_G.shape[0] > 0:
        J_full = vstack([J, J_G], format='csr')
        R_full = np.concatenate([R, R_G])
    else:
        J_full = J.tocsr()
        R_full = R

    n_rows, n_params = J_full.shape
    residual_sum_squares = float(R_full @ R_full)
    dof = max(n_rows - n_params, 1)
    sigma2 = residual_sum_squares / dof

    H_reg = (J_full.T @ J_full + 1e-8 * speye(n_params)).tocsc()
    # θ-блок обратной матрицы: решаем H X = E_θ вместо явного обращения
    rhs = np.zeros((n_params, n_theta))
    rhs[:n_theta, :] = np.eye(n_theta)
    try:
        solve = splu(H_reg).solve
        X = solve(rhs)
    except RuntimeError:
        X = np.linalg.pinv(H_reg.toarray()) @ rhs
    cov_theta = sigma2 * X[:n_theta, :]
    return cov_theta, sigma2, dof

def confidence_intervals(theta_opt, cov, dof, alpha=0.05):
    se = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - alpha/2, df=dof)
    ci_low = theta_opt - t_crit * se
    ci_high = theta_opt + t_crit * se
    return ci_low, ci_high

def compute_delta_gn(J, R, J_G, R_G, mu, lambda_, lambda_reg, theta_full, iter_num, mu_dec):
    multiple_shooting = J_G.shape[0] > 0
    n_params = len(theta_full)
    if multiple_shooting:
        H = J.T @ J   # csr
        reg_theta = lambda_reg * speye(n_params) + lambda_ * diags(H.diagonal())
        top = hstack([H + reg_theta, J_G.T])
        n_cont = J_G.shape[0]
        bottom = hstack([J_G, -mu * speye(n_cont)])
        H_full = vstack([top, bottom]).tocsr()     # ← преобразовать в CSR
        rhs = np.concatenate([J.T @ R, R_G])
        delta = spsolve(H_full, rhs)
        delta_theta = delta[:n_params]
        new_mu = mu * mu_dec
    else:
        H = J.T @ J
        H_reg = (H + 1e-8 * speye(n_params)).tocsr()   # ← преобразовать в CSR
        rhs = J.T @ R
        delta_theta = spsolve(H_reg, rhs)
        new_mu = mu
    return delta_theta, new_mu

def run_optimization(problem, config, theta_full, system):
    theta_hist = [theta_full[:].copy()]
    r_meas_hist = []
    r_cont_hist = []
    ci_low_hist = []
    ci_high_hist = []
    consecutive_failures = 0
    mu = config.mu
    n_theta = system.np

    # Минимально допустимое mu (можно задать в config, иначе 1e-6)
    mu_min = getattr(config, 'mu_min', 1e-6)

    # Начальные невязки для сравнения
    J, R, J_G, R_G = problem.solve(theta_full)
    best_cost = np.sum(R**2) + np.sum(R_G**2)  # полная стоимость
    print(f'  J nnz: {J.nnz}, J_G nnz: {J_G.nnz}')
    for it in range(config.n_iter):
        iter_start = time.time()

        # Логирование текущей стоимости
        meas_cost = np.sum(R**2) / max(1, len(R))
        cont_cost = np.sum(R_G**2) / max(1, len(R_G)) if R_G.size > 0 else 0.0
        print(f'Iter {it:3d} | R_meas: {meas_cost:.3e} | R_cont: {cont_cost:.3e} | mu: {mu:.2e}')

        # Ковариация и доверительные интервалы (на текущих J, R)
        cov_theta, _, dof = compute_parameter_covariance(J, R, J_G, R_G, n_theta)
        ci_low_theta, ci_high_theta = confidence_intervals(theta_full[:n_theta], cov_theta, dof, alpha=0.05)
        ci_low_hist.append(ci_low_theta)
        ci_high_hist.append(ci_high_theta)

        delta_theta, new_mu = compute_delta_gn(
            J, R, J_G, R_G, mu,
            config.lambda_, config.lambda_reg,
            theta_full, it, config.mu_dec
        )

        theta_trial = theta_full + delta_theta

        # Проверяем на NaN в параметрах
        if np.any(np.isnan(theta_trial)):
            print("NaN в параметрах, остановка.")
            break


        J_trial, R_trial, J_G_trial, R_G_trial = problem.solve(theta_trial)
        trial_cost = np.sum(R_trial**2) + np.sum(R_G_trial**2)

        # Принимаем шаг только если стоимость уменьшилась (или не изменилась)
        if not np.isnan(trial_cost) and trial_cost <= best_cost:
            # Успех: применяем новые параметры и обновляем mu (но не ниже mu_min)
            theta_full = theta_trial
            best_cost = trial_cost
            J, R, J_G, R_G = J_trial, R_trial, J_G_trial, R_G_trial
            mu = max(new_mu, mu_min)   # не даём mu упасть слишком низко
            consecutive_failures = 0
        else:
            # Шаг плохой: откатываем, mu оставляем прежним
            print(f"  Шаг отклонён (cost {trial_cost:.3e} > {best_cost:.3e}), mu сохранён {mu:.2e}")
            # mu не меняется, оставляем старые J, R и theta_full
            consecutive_failures += 1
            mu = max(mu /config.mu_dec, mu_min)
            if consecutive_failures >= 3:
                print(f"  Остановка после {consecutive_failures} неудачных шагов подряд")
                break
        # Сохраняем историю (текущие theta_full и невязки)
        theta_hist.append(theta_full.copy())
        r_meas_hist.append(meas_cost)
        r_cont_hist.append(cont_cost)

        # Если mu достиг минимума и улучшений нет, можно остановиться
        if mu <= mu_min and trial_cost > best_cost:
            print("mu достиг нижней границы, улучшений нет – остановка.")
            break

        print(f'  Iter time: {time.time() - iter_start:.3f}s')

    ci_low_hist = np.array(ci_low_hist)
    ci_high_hist = np.array(ci_high_hist)

    return theta_hist, r_meas_hist, r_cont_hist, theta_full, ci_low_hist, ci_high_hist
