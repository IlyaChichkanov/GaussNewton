import numpy as np
from commom_utils.ode_system import ODESystem, SystemJacobian
from scipy.sparse import block_diag, lil_matrix, csr_matrix, vstack, hstack, eye as speye, diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags as spdiags
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
                 c0_cost: float = 1, use_jax: bool = False):
        self.system = SystemJacobian(system)
        self.N_shoot = N_shoot
        self.gamma = gamma
        self.c0_cost = c0_cost
        self.use_jax = use_jax

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
        
        for state_measured, t_meas in zip(self.state_measured_batches, self.t_eval_measurements_batches):
            n_meas = len(t_meas)
            meas_idx = np.arange(n_meas, dtype=int)
            shoot_idx = meas_idx[0:-1:int(len(meas_idx) / self.N_shoot)]
            shoot_idx = np.append(shoot_idx, meas_idx[-1])
            
            for i in range(len(shoot_idx) - 1):
                idx = shoot_idx[i]
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
        import time

        solve_start = time.perf_counter()

        J_batches = []
        J_G_batches = []
        R_batches = []
        R_G_batches = []

        for batch, (state_measured, t_meas) in enumerate(
                zip(self.state_measured_batches, self.t_eval_measurements_batches)):
            print(f"Solve batch {batch}")
            batch_solve_start = time.perf_counter()
            J_batch, J_G_batch, R_batch, R_G_batch = self._solve_batch(
                theta_full, state_measured, t_meas, batch
            )
            batch_solve_elapsed = time.perf_counter() - batch_solve_start

            J_batches.append(J_batch)
            J_G_batches.append(J_G_batch)
            R_batches.append(R_batch)
            R_G_batches.append(R_G_batch)
            print(f'  Timing | batch calculation: {batch_solve_elapsed:.3f}s')

        concatenate_start = time.perf_counter()
        J_concatenate_start = time.perf_counter()
        J_total = self._concatenate_jacobian_batches(J_batches)
        J_concatenate_elapsed = time.perf_counter() - J_concatenate_start

        J_G_concatenate_start = time.perf_counter()
        J_G_total = self._concatenate_jacobian_batches(J_G_batches)
        J_G_concatenate_elapsed = time.perf_counter() - J_G_concatenate_start

        residuals_concatenate_start = time.perf_counter()
        R_total = np.concatenate(R_batches)
        R_G_total = np.concatenate(R_G_batches)
        residuals_concatenate_elapsed = time.perf_counter() - residuals_concatenate_start
        concatenate_elapsed = time.perf_counter() - concatenate_start

        solve_elapsed = time.perf_counter() - solve_start
        print(
            f'Concatenation | J: {J_concatenate_elapsed:.3f}s | '
            f'J_G: {J_G_concatenate_elapsed:.3f}s | '
            f'residuals: {residuals_concatenate_elapsed:.3f}s | '
            f'total: {concatenate_elapsed:.3f}s | '
            f'Solve total: {solve_elapsed:.3f}s | '
            f'J_G_total shape: {J_G_total.shape}, R_G_total len: {len(R_G_total)}'
        )
        return J_total, R_total, J_G_total, R_G_total
        
    def _solve_batch(self, theta_full, state_measured, t_meas, batch_idx):
        n_state, n_theta, n_meas = self.system.get_dimentions()
        if self.gamma is not None and len(self.gamma) != n_meas:
            raise ValueError(f"gamma length must be {n_meas}, got {len(self.gamma)}")
        idx_theta = slice(0, n_state * n_theta)
        idx_c = slice(n_state * n_theta, n_state * (n_theta + n_state))
        tm = self.interval_managers[batch_idx]
        n_shoot = tm.N_shoot


        intervals = [tm.get_time_interval(sh) for sh in range(n_shoot)]
        total_meas_points = sum(len(meas_idx) for _, meas_idx in intervals)
        n_params = n_theta + n_shoot * n_state

        J = lil_matrix((total_meas_points * n_meas, n_params))
        R = np.zeros(total_meas_points * n_meas)
        J_G = lil_matrix(((n_shoot - 1) * n_state, n_params))
        R_G = np.zeros((n_shoot - 1) * n_state)

        meas_row = 0  
        cont_row = 0
        Jx_prev = None
        Jc_prev = None
        state_prev = None

        for shoot, (t_interval, meas_idx) in enumerate(intervals):
            start_idx = n_theta + n_shoot * batch_idx * n_state + shoot * n_state
            c0 = theta_full[start_idx: start_idx + n_state]

            if self.use_jax:
                sol = self.system.get_jacobian_solution_jax(c0, theta_full[:n_theta], t_interval)
            else:
                sol = self.system.get_jacobian_solution(c0, theta_full[:n_theta], t_interval)

            state_traj = sol[:n_state, :]
            J_raw = sol[n_state:, :]

            # Обработка измерений интервала (цикл, как и раньше)
            for i, idx in enumerate(meas_idx):
                state = state_traj[:, i]
                t = t_interval[i]

                dh_dx = self.system.dh_dx(state, t, theta_full[:n_theta])
                dh_dtheta = self.system.dh_dtheta(state, t, theta_full[:n_theta])

                dx_dtheta = J_raw[idx_theta, i].reshape(n_state, n_theta)
                dx_dc = J_raw[idx_c, i].reshape(n_state, n_state)

                J_theta = dh_dx @ dx_dtheta + dh_dtheta   # (n_meas, n_theta)
                J_c = dh_dx @ dx_dc                       # (n_meas, n_state)

                # Строки, соответствующие этому измерению
                row_start = meas_row * n_meas
                row_end = row_start + n_meas

                # Заполняем блоки в разреженной матрице
                J[row_start:row_end, :n_theta] = J_theta
                col_start = n_theta + shoot * n_state
                J[row_start:row_end, col_start:col_start + n_state] = J_c

                y_meas = state_measured[idx]
                y_pred = self.system.h_x(state, t, theta_full[:n_theta])
                R[row_start:row_end] = (y_meas - y_pred)

                if self.gamma is not None:
                    R[row_start:row_end] *= self.gamma
                    J[row_start:row_end, :] = spdiags(self.gamma, 0, shape=(n_meas, n_meas)) @ J[row_start:row_end, :]
                    if i == 0:   # первая точка интервала
                        R[row_start:row_end] *= self.c0_cost
                        J[row_start:row_end, :] *= self.c0_cost

                meas_row += 1

            # Ограничения непрерывности
            if shoot > 0:
                row_idx = cont_row * n_state
                J_G[row_idx:row_idx + n_state, :n_theta] = Jx_prev
                col_prev = n_theta + (shoot - 1) * n_state
                J_G[row_idx:row_idx + n_state, col_prev:col_prev + n_state] = Jc_prev
                col_curr = n_theta + shoot * n_state
                J_G[row_idx:row_idx + n_state, col_curr:col_curr + n_state] = -np.eye(n_state)
                R_G[row_idx:row_idx + n_state] = -(state_prev - c0)
                cont_row += 1

            Jx_prev = J_raw[idx_theta, -1].reshape(n_state, n_theta)
            Jc_prev = J_raw[idx_c, -1].reshape(n_state, n_state)
            state_prev = state_traj[:, -1]

        # Преобразуем в CSR для быстрых операций
        J = J.tocsr()
        J_G = J_G.tocsr()
        return J, J_G, R, R_G


def compute_parameter_covariance_measurements_only(J, R, n_theta):
    # J – полный якобиан измерений (n_meas × n_params_full)
    # Берём только первые n_theta столбцов (θ)
    J_theta = J[:, :n_theta]
    n_meas = J.shape[0]
    # Оценка дисперсии шума по измерительным невязкам
    residual_sum_squares = np.dot(R, R)
    dof = max(n_meas - n_theta, 1)
    sigma2 = residual_sum_squares / dof
    H = J_theta.T @ J_theta
    H_reg = H + 1e-8 * np.eye(n_theta)
    try:
        cov_theta = sigma2 * np.linalg.inv(H_reg.toarray() if hasattr(H_reg, 'toarray') else H_reg)
    except np.linalg.LinAlgError:
        cov_theta = sigma2 * np.linalg.pinv(H_reg.toarray() if hasattr(H_reg, 'toarray') else H_reg)
    return cov_theta, sigma2

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
    import time
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
        start = time.time()

        # Логирование текущей стоимости
        meas_cost = np.sum(R**2) / max(1, len(R))
        cont_cost = np.sum(R_G**2) / max(1, len(R_G)) if R_G.size > 0 else 0.0
        print(f'Iter {it:3d} | time: {time.time()-start:.3f}s | '
              f'R_meas: {meas_cost:.3e} | R_cont: {cont_cost:.3e} | mu: {mu:.2e}')

        # Ковариация и доверительные интервалы (на текущих J, R)
        cov_theta, _ = compute_parameter_covariance_measurements_only(J, R, n_theta)
        dof_meas = max(1, J.shape[0] - n_theta)
        ci_low_theta, ci_high_theta = confidence_intervals(theta_full[:n_theta], cov_theta, dof_meas, alpha=0.05)
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
        if not np.isnan(trial_cost) and trial_cost <= 20*best_cost:
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
                break
                mu = max(mu *config.mu_dec, mu_min)
                print(f"  Принудительно уменьшаем mu до {mu:.2e} из-за {consecutive_failures} неудач")
                consecutive_failures = 0
        # Сохраняем историю (текущие theta_full и невязки)
        theta_hist.append(theta_full.copy())
        r_meas_hist.append(meas_cost)
        r_cont_hist.append(cont_cost)

        # Если mu достиг минимума и улучшений нет, можно остановиться
        if mu <= mu_min and trial_cost > best_cost:
            print("mu достиг нижней границы, улучшений нет – остановка.")
            break

        elapsed = time.time() - start

    ci_low_hist = np.array(ci_low_hist)
    ci_high_hist = np.array(ci_high_hist)

    return theta_hist, r_meas_hist, r_cont_hist, theta_full, ci_low_hist, ci_high_hist
