import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from acados_template import AcadosOcpSolver
from mhe.mhe_base_model_interface import MheModel
from matplotlib.widgets import RangeSlider
from tqdm import tqdm
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MheIterationResult:
    """Store results from one MHE iteration."""
    t_batch: np.ndarray
    state_sequence: np.ndarray
    control_sequence: np.ndarray
    state_est: np.ndarray
    noise_est: np.ndarray
    param_est: np.ndarray
    cov_matrix: np.ndarray
    fim: np.ndarray
    eigvals: np.ndarray
    status: int
    cost_value: float
    sqp_iter: int


class ArrivalCostUpdater:
    def __init__(self, mhe_model: MheModel, dt: float,
                 q_state_diag: float = 1e-6,
                 q_param_diag: float = 1e-6,
                 initial_sigma: Optional[np.ndarray] = None):
        self.mhe_model = mhe_model
        self.nx = mhe_model.state_length
        self.np = mhe_model.param_length
        self.n_aug = self.nx + self.np
        self.dt = dt
        self.step_func = mhe_model.create_step_function(dt, "ekf_step")

        if initial_sigma is not None:
            self.Sigma = initial_sigma
        else:
            self.Sigma = np.eye(self.n_aug)

        self.Q = np.diag(np.concatenate([
            np.full(self.nx, q_state_diag),
            np.full(self.np, q_param_diag),
        ]))

    def _compute_full_a(self, x, theta, u):
        res = self.step_func(x=x, theta=theta, u=u)
        J_x = np.array(res['Jx'])
        J_theta = np.array(res['Jtheta'])
        A = np.zeros((self.n_aug, self.n_aug))
        A[:self.nx, :self.nx] = J_x
        A[:self.nx, self.nx:] = J_theta
        A[self.nx:, self.nx:] = np.eye(self.np)
        return A

    def predict_covariance(self, x_seq, theta, u_seq):
        Sigma = self.Sigma.copy()
        for k in range(len(u_seq)):
            A = self._compute_full_a(x_seq[k], theta, u_seq[k])
            #print(A)
            Sigma = A @ Sigma @ A.T + self.Q
        self.Sigma = Sigma

    def correct(self, f_aug):
        Y_prior = np.linalg.inv(self.Sigma)
        Y_post = Y_prior + f_aug
        self.Sigma = np.linalg.inv(Y_post)

    def get_augmented_info_matrix(self):
        return np.linalg.inv(self.Sigma).flatten('F')


def regularize_matrix(m, ridge=1e-6):
    m = (m + m.T) / 2.0
    eigvals, _ = la.eigh(m)
    eigvals = eigvals[::-1]
    m_reg = m + ridge * np.eye(m.shape[0])
    return m_reg, eigvals


def reset_mhe_solver(mhe_model, solver, control_seq,
                     initial_x0, initial_theta, horizon, dt):
    step_func = mhe_model.create_step_function(dt, "reset_traj")
    x_sim = initial_x0.copy()
    for j in range(horizon):
        solver.set(j, "x", np.hstack([x_sim, initial_theta]))
        if j < horizon - 1:
            res = step_func(x=x_sim, theta=initial_theta, u=control_seq[j])
            x_sim = np.array(res['x_next']).flatten()


@dataclass
class MheEstimationData:
    sim_x_est: np.ndarray
    sim_w_est: np.ndarray
    sim_param_est: np.ndarray
    cost_value: float
    sqp_iter: int


# ------------------------------------------------------------
# 4. Главный цикл MHE
# ------------------------------------------------------------
def run_mhe_estimation(
    mhe_model: MheModel,
    acados_solver_factory: AcadosOcpSolver,
    get_window_func,
    get_initial_state_func,
    overlap_points: int,
    initial_theta: np.ndarray,
    mhe_params,
    num_windows: int,
    dt: float,
    r_inv: np.ndarray,
    q_state_diag: float = 1e-6,
    q_param_diag: float = 1e-6,
    initial_sigma: Optional[np.ndarray] = None,
    ridge_reg: float = 1e-6,
    progress_bar: bool = True,
) -> list[MheIterationResult]:
    """
    Запуск MHE с EKF-обновлением ковариации.
    Априорное состояние берётся из оптимальной траектории предыдущего окна.
    """
    N = mhe_params.mhe_horizont          # длина горизонта
    L = N - overlap_points               # сдвиг окна (число новых точек)
    nx = mhe_model.state_length
    results = []

    # ------------------------------------------------
    # Инициализация ковариационного менеджера
    # ------------------------------------------------
    updater = ArrivalCostUpdater(
        mhe_model, dt, q_state_diag=q_state_diag,
        q_param_diag=q_param_diag,
        initial_sigma=initial_sigma
    )

    # Переменные, которые живут между окнами
    x_prior = None              # априорное состояние для текущего окна
    theta_prior = initial_theta.copy()
    control_sequence_prev = None
    prev_sim_x_est = None       # оптимальная траектория предыдущего окна

    first_window = True
    iterator = range(num_windows)
    if progress_bar:
        iterator = tqdm(iterator, desc="MHE windows", unit="window")

    for iter_idx in iterator:

        t_batch, control_sequence, state_sequence, _ = get_window_func(iter_idx)

        if first_window:
            x_prior = get_initial_state_func(
                state_sequence[0], control_sequence[0], initial_theta
            )
            P_aug = updater.get_augmented_info_matrix()   # начальная P
        else:
            u_seq = [control_sequence_prev[i] for i in range(L)]
            x_seq = [prev_sim_x_est[i] for i in range(L + 1)]
            updater.predict_covariance(x_seq, theta_prior, u_seq)
            P_aug = updater.get_augmented_info_matrix()
            x_prior = prev_sim_x_est[L]

        control_sequence_prev = control_sequence.copy()

        set_mhe_solver(
            mhe_model, acados_solver_factory,
            state_sequence, control_sequence,
            x_prior, theta_prior, N, P_aug
        )
        # reset_mhe_solver(
        #     mhe_model, acados_solver_factory,
        #     control_sequence, x_prior, theta_prior, N, dt
        # )

        status = acados_solver_factory.solve()
        if status != 0:
            logger.warning(f"Window {iter_idx}: acados status {status}. Skip.")
            continue

        est = get_mhe_estimated_data(mhe_model, acados_solver_factory, N, use_noise=False)
        sim_x_est = est.sim_x_est
        theta_opt = est.sim_param_est

        x_aug0 = acados_solver_factory.get(0, "x")
        x0_opt = np.array(x_aug0[:nx]).flatten()

        F_aug = mhe_model.compute_augmented_fim(
             dt, control_sequence[:N - overlap_points], x0_opt, theta_opt, r_inv
        )
        F_aug_reg, eigvals = regularize_matrix(F_aug, ridge_reg)
        updater.correct(F_aug_reg)

        theta_prior = theta_opt
        prev_sim_x_est = sim_x_est.copy()
        # Флаг снимается ТОЛЬКО после успешного окна, вместе с prev_sim_x_est.
        # Раньше он снимался сразу в if-ветке, и если окно не решалось (status
        # != 0 -> continue), следующая итерация уходила в else с
        # prev_sim_x_est = None -> TypeError: 'NoneType' is not subscriptable.
        first_window = False

        result = MheIterationResult(
            t_batch=t_batch,
            state_sequence=state_sequence,
            control_sequence=control_sequence,
            state_est=sim_x_est,
            noise_est=est.sim_w_est,
            param_est=theta_opt,
            cov_matrix=updater.Sigma[nx:, nx:].flatten(), # ковариация параметров
            fim=F_aug_reg,
            eigvals=eigvals,
            status=status,
            cost_value=est.cost_value,
            sqp_iter=est.sqp_iter
        )
        results.append(result)

    return results


def plot_mhe_data_windows(t_windows, u_windows, meas_windows, full_windows=None,
                          max_windows=4, state_idx=None):
    """
    Визуализация окон MHE: каждое окно в отдельной строке (subplot).

    Параметры:
    t_windows, u_windows, meas_windows : списки массивов по окнам
    full_windows : список истинных состояний (если есть, для отладки) – форма (N, nx)
    max_windows : число окон для отображения
    state_idx : список индексов состояний для отображения (по умолчанию все)
    """
    n_windows = min(len(t_windows), max_windows)
    n_states = meas_windows[0].shape[1]  # измерений (обычно = nx или меньше)
    n_controls = u_windows[0].shape[1] if u_windows[0].ndim > 1 else 1

    if state_idx is None:
        state_idx = list(range(n_states))

    fig, axs = plt.subplots(n_windows, 1, figsize=(12, 3 * n_windows),
                            sharex=True, squeeze=False)
    axs = axs.flatten()

    for i in range(n_windows):
        ax = axs[i]
        t = t_windows[i]
        u = u_windows[i]
        meas = meas_windows[i]
        full = full_windows[i] if full_windows is not None else None

        for j, s_idx in enumerate(state_idx):
            ax.plot(t, meas[:, s_idx], 'o-', markersize=3,
                    label=f'Meas state {s_idx}', alpha=0.8)

        if full is not None:
            for j, s_idx in enumerate(state_idx):
                ax.plot(t, full[:, s_idx], '--', linewidth=1.5,
                        label=f'True state {s_idx}', alpha=0.7)

        ax2 = ax.twinx()
        for j in range(n_controls):
            u_vals = u[:, j] if u.ndim > 1 else u
            ax2.plot(t, u_vals, '--', color='gray', alpha=0.5,
                     label=f'control_input_{j}')
        ax2.set_ylabel('Control', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_ylabel(f'Window {i}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                      fontsize=18)

    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('MHE data windows', fontsize=18)
    plt.tight_layout()
    plt.show()


def set_mhe_solver(mhe_model: MheModel,
               acados_solver_mhe: AcadosOcpSolver,
               state_sequence: np.array,
               control_sequence: np.array,
               initial_x0: np.array,
               initial_theta: np.array,
               horizon: int,
               p0=np.array) -> tuple:
    assert (len(initial_x0) == mhe_model.state_length)
    assert (len(initial_theta) == mhe_model.param_length)

    x_prior = np.hstack((initial_x0, initial_theta))
    for j in range(horizon):
        p_ext = np.hstack((control_sequence[j, :], state_sequence[j, :], x_prior, p0.flatten()))
        acados_solver_mhe.set(j, "p", p_ext)


def get_mhe_estimated_data(mhe_model: MheModel, acados_solver_mhe: AcadosOcpSolver, horizon: int, use_noise=False):
    """
    Extract estimated states, noise, parameters, cost and iterations from an acados solver.

    Parameters:
        mhe_model: the MheModel instance (provides state_length, param_length)
        acados_solver_mhe: the AcadosOcpSolver after a successful solve
        N: horizon length (number of intervals)

    Returns:
        MheEstimationData object with all collected data.
    """
    nx = mhe_model.state_length
    param_length = mhe_model.param_length

    sim_x_est = np.zeros((horizon + 1, nx))
    sim_w_est = np.zeros((horizon, nx))
    sim_param_est = np.zeros(param_length,)

    # Fill data for nodes 0..N-1 (the first N nodes)
    for i in range(horizon):
        x_augmented = acados_solver_mhe.get(i, "x")
        sim_x_est[i, :] = x_augmented[:nx]
        if (use_noise):
            sim_w_est[i, :] = acados_solver_mhe.get(i, "u")

    # Get the state at the final node (index N)
    x_final = acados_solver_mhe.get(horizon, "x")
    sim_x_est[horizon, :] = x_final[:nx]
    sim_param_est = x_final[nx : nx + param_length]

    # Retrieve cost and iterations (available from the solver after solving)

    cost_value = acados_solver_mhe.get_cost()
    sqp_iter = acados_solver_mhe.get_stats('sqp_iter')

    return MheEstimationData(
        sim_x_est=sim_x_est,
        sim_w_est=sim_w_est,
        sim_param_est=sim_param_est,
        cost_value=cost_value,
        sqp_iter=sqp_iter
    )


def make_system_trajectory(mhe_model: MheModel,
               control_sequence: np.array,
               initial_x0: np.array,
               initial_theta: np.array,
               horizon: int, dt: float) -> tuple:
    assert (len(initial_x0) == mhe_model.state_length)
    assert (len(initial_theta) == mhe_model.param_length)
    assert control_sequence.shape[0] >= horizon, f"control_sequence должен содержать хотя бы {horizon} строк"

    x_sim = initial_x0.copy()
    trajectory = np.zeros((horizon + 1, mhe_model.state_length))
    integrate_f = mhe_model.create_integrate_function(dt, "integrate")
    trajectory[0] = x_sim
    for j in range(horizon):
        x_aug = np.hstack((x_sim, initial_theta))
        if j < horizon - 1:
            x_sim = np.array(integrate_f(x_sim, initial_theta, control_sequence[j, :])).T[0]
        trajectory[j + 1] = x_sim
    return trajectory


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider

# def plot_mhe_results(results, overlap=0, initial_params=None, initial_std=None,
#                      theta_true=None,
#                      plot_states=True, plot_params=True,
#                      plot_eigvals=True, plot_noise=True,
#                      plot_cost=True, plot_iter=True, plot_status=True,
#                      plot_cov_matrix=True,
#                      figsize=(15, 15), verbose=False):
#     """
#     Визуализация результатов MHE с интерактивным слайдером времени
#     и автоматическим масштабированием оси Y.

#     Параметры:
#     -----------
#     results : list of MheIterationResult
#         Список объектов с результатами каждого окна.
#     overlap : int
#         Число перекрывающихся точек между окнами.
#     initial_params : np.ndarray или None
#         Начальная оценка параметров (до первого окна) – размер (n_theta,).
#     initial_std : np.ndarray или None
#         Начальные стандартные отклонения параметров – размер (n_theta,).
#         Если None, используется std первого окна (с предупреждением).
#     theta_true : np.ndarray или None
#         Истинные значения параметров (для отрисовки горизонтальных линий).
#     plot_* : bool
#         Флаги включения отдельных графиков.
#     figsize : tuple
#         Размер фигуры.
#     verbose : bool
#         Печатать отладочную информацию.
#     """
#     if not results:
#         print("No results to plot.")
#         return

#     # -------- Определение активных панелей --------
#     active_plots = []
#     if plot_states:   active_plots.append('states')
#     if plot_params:   active_plots.append('params')
#     if plot_eigvals:  active_plots.append('eigvals')
#     if plot_noise:    active_plots.append('noise')
#     if plot_cost:     active_plots.append('cost')
#     if plot_iter:     active_plots.append('iter')
#     if plot_status:   active_plots.append('status')
#     if plot_cov_matrix: active_plots.append('cov_matrix')

#     n_plots = len(active_plots)
#     if n_plots == 0:
#         print("Nothing to plot.")
#         return

#     # Высоты: первая панель (если 'states') – 2, остальные – 1, слайдер – 0.2
#     ratios = [2 if p == 'states' else 1 for p in active_plots] + [0.2]
#     fig, axs = plt.subplots(n_plots + 1, 1, figsize=figsize,
#                             gridspec_kw={'height_ratios': ratios},
#                             squeeze=False)
#     axs = axs.flatten()
#     slider_ax = axs[-1]
#     plot_axes = axs[:-1]
#     axes_dict = dict(zip(active_plots, plot_axes))

#     # -------- Склейка данных с учётом перекрытия --------
#     t_full, measured_full, estimated_full = [], [], []
#     params_full, std_full = [], []

#     for idx, res in enumerate(results):
#         t = np.asarray(res.t_batch)
#         meas = np.asarray(res.state_sequence)
#         est = np.asarray(res.state_est)
#         params = np.asarray(res.param_est)

#         n_points = min(len(t), len(est), len(meas))
#         t = t[:n_points]; meas = meas[:n_points]; est = est[:n_points]

#         # Восстанавливаем std для этого окна
#         cov_flat = res.cov_matrix
#         n_theta = len(params) if params.ndim == 1 else params.shape[1]
#         cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
#         diag = np.diag(cov_mat)
#         std_window = np.sqrt(np.maximum(diag, 0.0))

#         # Размножаем параметры и std на все временные точки окна
#         if params.ndim == 1:
#             params_2d = np.tile(params, (n_points, 1))
#         else:
#             params_2d = params[:n_points]
#         std_2d = np.tile(std_window, (n_points, 1))

#         # Учёт перекрытия
#         start = 0 if idx == 0 else min(overlap, n_points) if overlap < n_points else n_points

#         t_full.extend(t[start:])
#         measured_full.extend(meas[start:])
#         estimated_full.extend(est[start:])
#         params_full.extend(params_2d[start:])
#         std_full.extend(std_2d[start:])

#         if verbose:
#             print(f"Window {idx}: n_points={n_points}, start={start}, added={n_points - start}")

#     # Приводим к массивам и одинаковой длине
#     t_full = np.array(t_full)
#     measured_full = np.array(measured_full)
#     estimated_full = np.array(estimated_full)
#     params_full = np.array(params_full)
#     std_full = np.array(std_full)

#     min_len = min(len(t_full), len(measured_full), len(estimated_full),
#                   len(params_full), len(std_full))
#     t_full = t_full[:min_len]
#     measured_full = measured_full[:min_len]
#     estimated_full = estimated_full[:min_len]
#     params_full = params_full[:min_len]
#     std_full = std_full[:min_len]

#     # -------- Отрисовка графиков --------
#     # 1. Состояния
#     if 'states' in axes_dict:
#         ax = axes_dict['states']
#         n_obs = measured_full.shape[1]
#         n_x = estimated_full.shape[1]
#         for i in range(n_obs):
#             ax.plot(t_full, measured_full[:, i], '--', label=f'Meas y_{i+1}')
#         for i in range(n_x):
#             ax.plot(t_full, estimated_full[:, i], '-', label=f'Est x_{i+1}')
#         ax.set_title("States: Measured (dashed) vs Estimated (solid)")
#         ax.set_xlabel("Time"); ax.set_ylabel("State")
#         ax.legend(); ax.grid(True)

#     # 2. Параметры с доверительными интервалами
#     if 'params' in axes_dict:
#         ax = axes_dict['params']
#         # Число параметров определяем по данным окон, а если их нет, то по initial_params или theta_true
#         if params_full.size > 0:
#             ntheta = params_full.shape[1]
#         elif initial_params is not None:
#             ntheta = len(initial_params)
#         elif theta_true is not None:
#             ntheta = len(theta_true)
#         else:
#             ntheta = 0

#         if ntheta == 0:
#             print("Warning: Could not determine number of parameters, skipping 'params' plot.")
#             axes_dict['params'].set_visible(False)
#         else:
#             t_plot = t_full.copy()
#             p_plot = params_full.copy()
#             s_plot = std_full.copy()

#             # Добавляем начальную точку, если заданы initial_params
#             if initial_params is not None:
#                 if len(t_plot) > 1:
#                     dt = t_plot[1] - t_plot[0]
#                 else:
#                     # Если всего одна точка окна, берём средний шаг из первого окна
#                     dt = np.mean(np.diff(results[0].t_batch)) if len(results[0].t_batch) > 1 else 1.0
#                 t_start = t_plot[0] - dt

#                 # Проверка/задание начального std
#                 if initial_std is not None:
#                     init_std = np.asarray(initial_std).flatten()
#                     if len(init_std) != ntheta:
#                         raise ValueError(f"initial_std length {len(init_std)} != ntheta {ntheta}")
#                 else:
#                     import warnings
#                     warnings.warn("initial_std not provided, using std of first window for initial point.")
#                     init_std = s_plot[0] if len(s_plot) > 0 else np.zeros(ntheta)

#                 # Вставляем начальные значения
#                 p_plot = np.vstack([initial_params, p_plot])
#                 s_plot = np.vstack([init_std, s_plot])
#                 t_plot = np.insert(t_plot, 0, t_start)

#             for i in range(ntheta):
#                 ax.plot(t_plot, p_plot[:, i], label=f'θ_{i+1} estimated')
#                 ax.fill_between(t_plot,
#                                 p_plot[:, i] - s_plot[:, i],
#                                 p_plot[:, i] + s_plot[:, i],
#                                 alpha=0.2, label=f'θ_{i+1} ±1σ' if i == 0 else "")
#             if theta_true is not None:
#                 for i, val in enumerate(theta_true):
#                     ax.axhline(y=val, linestyle=':', color=f'C{i}', alpha=0.8,
#                                label=f'θ_{i+1} true')
#             ax.set_title("Parameter estimates over time (shaded: ±1σ)")
#             ax.set_xlabel("Time"); ax.set_ylabel("Parameter value")
#             ax.legend(); ax.grid(True)

#     # 3. Собственные числа FIM
#     if 'eigvals' in axes_dict:
#         ax = axes_dict['eigvals']
#         n_theta = results[0].eigvals.shape[0]
#         eig_vals_matrix = np.array([res.eigvals for res in results])
#         eig_vals_sorted = np.sort(eig_vals_matrix, axis=1)[:, ::-1]
#         for i in range(n_theta):
#             ax.semilogy(eig_vals_sorted[:, i], marker='o', label=f'λ_{i+1}')
#         ax.set_title("FIM eigenvalues per window (log scale)")
#         ax.set_xlabel("Window index"); ax.set_ylabel("Eigenvalue magnitude")
#         ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.7)

#     # 4. Стандартные отклонения параметров (отдельный график)
#     if 'cov_matrix' in axes_dict:
#         ax = axes_dict['cov_matrix']
#         n_theta = results[0].eigvals.shape[0]
#         std_per_window = []
#         for res in results:
#             cov_flat = res.cov_matrix
#             cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
#             diag = np.diag(cov_mat)
#             std = np.sqrt(np.maximum(diag, 0.0))
#             std_per_window.append(std)
#         std_per_window = np.array(std_per_window)
#         for i in range(n_theta):
#             ax.plot(range(len(std_per_window)), std_per_window[:, i], marker='o',
#                     label=f'θ_{i+1} std')
#         ax.set_title("Parameter standard deviation (sqrt of diag(cov))")
#         ax.set_xlabel("Window index"); ax.set_ylabel("Standard deviation")
#         ax.legend(); ax.grid(True)

#     # 5. Распределение шума процесса
#     if 'noise' in axes_dict:
#         ax = axes_dict['noise']
#         all_noise = np.concatenate([res.noise_est.flatten() for res in results])
#         ax.hist(all_noise, bins=50, alpha=0.7, density=True)
#         ax.set_title("Process noise distribution")
#         ax.set_xlabel("Noise value"); ax.set_ylabel("Density")
#         ax.grid(True)

#     # 6. Стоимость
#     if 'cost' in axes_dict:
#         ax = axes_dict['cost']
#         cost_vals = [res.cost_value for res in results]
#         ax.plot(range(len(cost_vals)), cost_vals, marker='o')
#         ax.set_title("Cost value per window")
#         ax.set_xlabel("Window index"); ax.set_ylabel("Cost")
#         ax.grid(True); ax.set_yscale('log')

#     # 7. Итерации SQP
#     if 'iter' in axes_dict:
#         ax = axes_dict['iter']
#         iter_vals = [res.sqp_iter for res in results]
#         ax.plot(range(len(iter_vals)), iter_vals, marker='o')
#         ax.set_title("SQP iterations per window")
#         ax.set_xlabel("Window index"); ax.set_ylabel("Iterations")
#         ax.grid(True)

#     # 8. Статус солвера
#     if 'status' in axes_dict:
#         ax = axes_dict['status']
#         status_vals = [res.status for res in results]
#         ax.plot(range(len(status_vals)), status_vals, marker='o', linestyle='-')
#         ax.set_title("Solver status per window (0 = success)")
#         ax.set_xlabel("Window index"); ax.set_ylabel("Status")
#         ax.set_yticks(sorted(set(status_vals)))
#         ax.grid(True)

#     # -------- Интерактивный слайдер времени с авто-масштабированием Y --------
#     time_axes = []
#     if 'states' in axes_dict:
#         time_axes.append(axes_dict['states'])
#     if 'params' in axes_dict:
#         time_axes.append(axes_dict['params'])

#     if time_axes and len(t_full) > 1:
#         t_min, t_max = t_full.min(), t_full.max()
#         slider = RangeSlider(slider_ax, "Time", t_min, t_max, valinit=(t_min, t_max))

#         def update_ylims(ax, t_left, t_right):
#             """Пересчитывает пределы Y по видимым данным."""
#             lines = ax.get_lines()
#             if not lines:
#                 return
#             y_min = np.inf
#             y_max = -np.inf
#             for line in lines:
#                 xdata = np.asarray(line.get_xdata())
#                 ydata = np.asarray(line.get_ydata())
#                 mask = (xdata >= t_left) & (xdata <= t_right)
#                 if mask.any():
#                     y_min = min(y_min, np.nanmin(ydata[mask]))
#                     y_max = max(y_max, np.nanmax(ydata[mask]))
#             if np.isfinite(y_min) and np.isfinite(y_max):
#                 margin = 0.05 * (y_max - y_min) or 0.1 * max(abs(y_min), 1e-6)
#                 ax.set_ylim(y_min - margin, y_max + margin)

#         def update_xlim(val):
#             t_left, t_right = val
#             for ax in time_axes:
#                 ax.set_xlim(t_left, t_right)
#                 update_ylims(ax, t_left, t_right)
#             fig.canvas.draw_idle()

#         slider.on_changed(update_xlim)
#         # Инициализируем Y пределы для полного диапазона
#         for ax in time_axes:
#             update_ylims(ax, t_min, t_max)
#         slider_ax.set_xlabel('Time')
#         slider_ax.xaxis.set_label_coords(0.5, -0.5)
#     else:
#         slider_ax.set_visible(False)

#     plt.tight_layout()
#     plt.show()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import hex_to_rgb

# def plot_mhe_results(results, overlap=0, initial_params=None, initial_std=None,
#                      theta_true=None, plot_states=True, plot_params=True,
#                      plot_eigvals=True, plot_noise=True, plot_cost=True,
#                      plot_iter=True, plot_status=True, plot_cov_matrix=True,
#                      figsize=(1200, 900), verbose=False):
#     if not results:
#         print("No results to plot.")
#         return go.Figure()

#     active_plots = []
#     if plot_states:    active_plots.append("states")
#     if plot_params:    active_plots.append("params")
#     if plot_eigvals:   active_plots.append("eigvals")
#     if plot_noise:     active_plots.append("noise")
#     if plot_cost:      active_plots.append("cost")
#     if plot_iter:      active_plots.append("iter")
#     if plot_status:    active_plots.append("status")
#     if plot_cov_matrix:active_plots.append("cov_matrix")

#     if not active_plots:
#         print("All plots disabled.")
#         return go.Figure()

#     # ---------- склейка ----------
#     t_full, measured_full, estimated_full = [], [], []
#     params_full, std_full = [], []
#     for idx, res in enumerate(results):
#         t = np.asarray(res.t_batch)
#         meas = np.asarray(res.state_sequence)
#         est = np.asarray(res.state_est)
#         param = np.asarray(res.param_est)
#         n_points = min(len(t), len(est), len(meas))
#         t, meas, est = t[:n_points], meas[:n_points], est[:n_points]
#         param = param[:n_points] if param.ndim > 1 else param

#         cov_flat = res.cov_matrix
#         n_theta = len(param) if param.ndim == 1 else param.shape[1]
#         cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
#         std_window = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))

#         if param.ndim == 1:
#             param_2d = np.tile(param, (n_points, 1))
#         else:
#             param_2d = param[:n_points]
#         std_2d = np.tile(std_window, (n_points, 1))

#         start = 0 if idx == 0 else min(overlap, n_points) if overlap < n_points else n_points
#         t_full.extend(t[start:])
#         measured_full.extend(meas[start:])
#         estimated_full.extend(est[start:])
#         params_full.extend(param_2d[start:])
#         std_full.extend(std_2d[start:])

#         if verbose:
#             print(f"Window {idx}: n={n_points}, start={start}, added={n_points-start}, "
#                   f"param shape={param_2d.shape}")

#     t_full = np.array(t_full); measured_full = np.array(measured_full)
#     estimated_full = np.array(estimated_full); params_full = np.array(params_full)
#     std_full = np.array(std_full)

#     # Обрезаем до минимальной длины
#     min_len = min(len(t_full), len(measured_full), len(estimated_full),
#                   len(params_full), len(std_full))
#     t_full = t_full[:min_len]; measured_full = measured_full[:min_len]
#     estimated_full = estimated_full[:min_len]; params_full = params_full[:min_len]
#     std_full = std_full[:min_len]

#     if verbose:
#         print("\nFinal merged shapes:")
#         print(f"t_full: {t_full.shape}, measured: {measured_full.shape}, "
#               f"estimated: {estimated_full.shape}, params: {params_full.shape}, "
#               f"std: {std_full.shape}")

#     if min_len == 0:
#         print("ERROR: merged data is empty (overlap too large?)")
#         fig = go.Figure()
#         fig.add_annotation(text="No data after merging", showarrow=False)
#         return fig

#     # Добавление начальной точки для параметров
#     t_plot, p_plot, s_plot = t_full.copy(), params_full.copy(), std_full.copy()
#     if initial_params is not None and "params" in active_plots:
#         n_theta = p_plot.shape[1] if p_plot.size > 0 else len(initial_params)
#         dt = t_full[1]-t_full[0] if len(t_full)>1 else 1.0
#         t_start = t_full[0] - dt
#         init_std = np.asarray(initial_std) if initial_std is not None else np.zeros(n_theta)
#         p_plot = np.vstack([initial_params, p_plot])
#         s_plot = np.vstack([init_std, s_plot])
#         t_plot = np.insert(t_plot, 0, t_start)

#     # ---------- создание subplots ----------
#     row_heights = []
#     for p in active_plots:
#         if p == "states":
#             row_heights.append(3)
#         elif p == "params":
#             row_heights.append(2)   # параметры – второй по важности
#         else:
#             row_heights.append(1) 
#     fig = make_subplots(rows=len(active_plots), cols=1,
#                         shared_xaxes=False, row_heights=row_heights,
#                         vertical_spacing=0.05)

#     row_idx = 1
#     # 1. States
#     if "states" in active_plots:
#         n_obs = measured_full.shape[1] if measured_full.ndim>1 else 1
#         n_x = estimated_full.shape[1] if estimated_full.ndim>1 else 1
#         for i in range(n_obs):
#             y = measured_full[:, i] if measured_full.ndim>1 else measured_full
#             fig.add_trace(go.Scatter(x=t_full, y=y, mode='lines',
#                             name=f'Meas y_{i+1}', line=dict(dash='dash')),
#                           row=row_idx, col=1)
#         for i in range(n_x):
#             y = estimated_full[:, i] if estimated_full.ndim>1 else estimated_full
#             fig.add_trace(go.Scatter(x=t_full, y=y, mode='lines',
#                             name=f'Est x_{i+1}'),
#                           row=row_idx, col=1)
#         fig.update_xaxes(title_text="Time", row=row_idx, col=1)
#         fig.update_yaxes(title_text="State", row=row_idx, col=1)
#         row_idx += 1

#     # 2. Parameters
#     if "params" in active_plots:
#         n_theta = p_plot.shape[1] if p_plot.size>0 else 0
#         if n_theta == 0:
#             print("WARNING: n_theta=0 – skipping params plot.")
#             fig.add_annotation(text="No parameter data", row=row_idx, col=1)
#         else:
#             colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
#             for i in range(n_theta):
#                 color = colors[i % len(colors)]
#                 fig.add_trace(go.Scatter(x=t_plot, y=p_plot[:, i], mode='lines',
#                                 name=f'θ_{i+1} est', line=dict(color=color)),
#                               row=row_idx, col=1)
#                 try:
#                     r, g, b = hex_to_rgb(color)
#                     fillcolor = f'rgba({r},{g},{b},0.2)'
#                     fig.add_trace(go.Scatter(
#                         x=np.concatenate([t_plot, t_plot[::-1]]),
#                         y=np.concatenate([p_plot[:, i]-s_plot[:, i],
#                                           (p_plot[:, i]+s_plot[:, i])[::-1]]),
#                         fill='toself', fillcolor=fillcolor,
#                         line=dict(width=0), showlegend=False),
#                         row=row_idx, col=1)
#                 except Exception as e:
#                     print(f"Error adding CI for param {i}: {e}")
#                 if theta_true is not None and i < len(theta_true):
#                     fig.add_trace(go.Scatter(x=[t_plot[0], t_plot[-1]],
#                                     y=[theta_true[i], theta_true[i]],
#                                     mode='lines', line=dict(dash='dot', color=color),
#                                     name=f'θ_{i+1} true'),
#                                   row=row_idx, col=1)
#         fig.update_xaxes(title_text="Time", row=row_idx, col=1)
#         fig.update_yaxes(title_text="Parameter value", row=row_idx, col=1)
#         row_idx += 1

#     # Остальные графики (eigvals, cov_matrix, noise, cost, iter, status)
#     # вставьте их точно так же, как в предыдущей версии (без изменений)
#     # ... (для краткости опущу, они идентичны исправленному ранее коду)

#     fig.update_layout(height=figsize[1], width=figsize[0],
#                       title_text="MHE Results", showlegend=True,
#                       template="plotly_white")
#     print(f"Figure created with {len(fig.data)} traces.")
#     return fig
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import hex_to_rgb

def plot_mhe_slider_lines(
    results,
    overlap=0,
    initial_params=None,
    initial_std=None,
    theta_true=None,
    state_names=None,        # список имён оцениваемых состояний (длина = n_x)
    meas_names=None,          # список имён измерений (длина = n_obs)
    param_names=None,         # список имён параметров (длина = n_theta)
    fontsize = 10,
    figsize=(1200, 600),
):
    """
    Основные графики состояний и параметров на всём интервале.
    Вертикальные линии показывают границы выбранного окна.
    Слайдер переключает окна.

    Параметры:
    -----------
    results : list of MheIterationResult
        Список объектов с результатами каждого окна.
    overlap : int
        Число перекрывающихся точек между окнами.
    initial_params, initial_std : np.ndarray или None
        Начальные значения и стандартные отклонения параметров (добавляются одной точкой).
    theta_true : np.ndarray или None
        Истинные значения параметров (горизонтальные пунктирные линии).
    state_names : list of str или None
        Имена оцениваемых состояний (например, ['v_y', 'r']).
    meas_names : list of str или None
        Имена измерений (например, ['a_lat', 'r_meas']).
    param_names : list of str или None
        Имена параметров (например, ['C_f', 'C_r', 'a_rel', 'I_norm']).
    figsize : tuple
        Размер фигуры (ширина, высота) в пикселях.
    """
    if not results:
        return go.Figure()

    # --- Склейка данных ---
    t_full, meas_full, est_full, params_full, std_full = [], [], [], [], []
    window_bounds = [0]
    for idx, res in enumerate(results):
        t = np.asarray(res.t_batch)
        meas = np.asarray(res.state_sequence)
        est = np.asarray(res.state_est)
        param = np.asarray(res.param_est)
        n_points = min(len(t), len(est), len(meas))
        t, meas, est = t[:n_points], meas[:n_points], est[:n_points]

        cov_flat = res.cov_matrix
        n_theta_curr = len(param) if param.ndim == 1 else param.shape[1]
        cov_mat = np.array(cov_flat).reshape(n_theta_curr, n_theta_curr)
        std_window = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))

        if param.ndim == 1:
            param_2d = np.tile(param, (n_points, 1))
        else:
            param_2d = param[:n_points]
        std_2d = np.tile(std_window, (n_points, 1))

        start = 0 if idx == 0 else min(overlap, n_points) if overlap < n_points else n_points
        t_full.append(t[start:])
        meas_full.append(meas[start:])
        est_full.append(est[start:])
        params_full.append(param_2d[start:])
        std_full.append(std_2d[start:])
        window_bounds.append(window_bounds[-1] + len(t_full[-1]))

    t_full = np.concatenate(t_full)
    meas_full = np.concatenate(meas_full)
    est_full = np.concatenate(est_full)
    params_full = np.concatenate(params_full)
    std_full = np.concatenate(std_full)

    # Начальная точка параметров
    t_plot, p_plot, s_plot = t_full.copy(), params_full.copy(), std_full.copy()
    if initial_params is not None:
        dt = t_full[1] - t_full[0] if len(t_full) > 1 else 1.0
        t_start = t_full[0] - dt
        init_std_val = np.asarray(initial_std) if initial_std is not None else np.zeros(len(initial_params))
        p_plot = np.vstack([initial_params, p_plot])
        s_plot = np.vstack([init_std_val, s_plot])
        t_plot = np.insert(t_plot, 0, t_start)
        window_bounds = [b + 1 for b in window_bounds]

    n_windows = len(results)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    n_obs = meas_full.shape[1] if meas_full.ndim > 1 else 1
    n_x = est_full.shape[1] if est_full.ndim > 1 else 1
    n_theta = p_plot.shape[1] if p_plot.size > 0 else 0

    # --- Генерация имён по умолчанию, если не заданы ---
    if meas_names is None:
        meas_names = [f'Meas y_{i+1}' for i in range(n_obs)]
    else:
        if len(meas_names) != n_obs:
            raise ValueError(f"meas_names length {len(meas_names)} != n_obs {n_obs}")

    if state_names is None:
        state_names = [f'Est x_{i+1}' for i in range(n_x)]
    else:
        if len(state_names) != n_x:
            raise ValueError(f"state_names length {len(state_names)} != n_x {n_x}")

    if param_names is None:
        param_names = [f'θ_{i+1} est' for i in range(n_theta)]
    else:
        if len(param_names) != n_theta:
            raise ValueError(f"param_names length {len(param_names)} != n_theta {n_theta}")

    # --- Создание subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[3, 2],
        vertical_spacing=0.08,
        subplot_titles=["States & Measurements", "Parameters"]
    )

    # --- Основные трейсы (состояния) ---
    for i in range(n_obs):
        y = meas_full[:, i] if meas_full.ndim > 1 else meas_full
        fig.add_trace(go.Scatter(x=t_full, y=y, mode='lines',
                                 name=meas_names[i],                 # ← имя измерения
                                 line=dict(dash='dash')),
                      row=1, col=1)
    for i in range(n_x):
        y = est_full[:, i] if est_full.ndim > 1 else est_full
        fig.add_trace(go.Scatter(x=t_full, y=y, mode='lines',
                                 name=state_names[i]),               # ← имя состояния
                      row=1, col=1)

    # --- Основные трейсы (параметры) ---
    for i in range(n_theta):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=t_plot, y=p_plot[:, i], mode='lines',
                                 name=param_names[i],                 # ← имя параметра
                                 line=dict(color=color)),
                      row=2, col=1)
        try:
            r, g, b = hex_to_rgb(color)
            fig.add_trace(go.Scatter(
                x=np.concatenate([t_plot, t_plot[::-1]]),
                y=np.concatenate([p_plot[:, i] - s_plot[:, i],
                                  (p_plot[:, i] + s_plot[:, i])[::-1]]),
                fill="toself", fillcolor=f'rgba({r},{g},{b},0.2)',
                line=dict(width=0), showlegend=False),
                row=2, col=1)
        except:
            pass
        if theta_true is not None and i < len(theta_true):
            fig.add_trace(go.Scatter(x=[t_plot[0], t_plot[-1]], y=[theta_true[i], theta_true[i]],
                                     mode='lines', line=dict(dash='dot', color=color),
                                     name=f'True {param_names[i]}'),   # ← имя истинного значения
                          row=2, col=1)

    # Запоминаем, сколько сейчас трейсов – это «всегда видимые» основные данные
    num_main_traces = len(fig.data)

    # --- Вертикальные линии для каждого окна ---
    y_min_states = min(meas_full.min(), est_full.min())
    y_max_states = max(meas_full.max(), est_full.max())
    y_min_params = p_plot.min()
    y_max_params = p_plot.max()

    window_state_vlines_idx = []   # для каждого окна: [idx_start, idx_end]
    window_param_vlines_idx = []   # для каждого окна: idx_start

    for win in range(n_windows):
        start_idx = window_bounds[win]
        end_idx = window_bounds[win+1]
        t_start = t_full[start_idx]
        t_end = t_full[end_idx-1] if end_idx < len(t_full) else t_full[-1]

        # States: две линии
        l1 = go.Scatter(x=[t_start, t_start], y=[y_min_states, y_max_states],
                        mode='lines', line=dict(color='red', dash='dash', width=2),
                        showlegend=False, visible=(win==0))
        fig.add_trace(l1, row=1, col=1)
        idx_l1 = len(fig.data) - 1

        l2 = go.Scatter(x=[t_end, t_end], y=[y_min_states, y_max_states],
                        mode='lines', line=dict(color='red', dash='dash', width=2),
                        showlegend=False, visible=(win==0))
        fig.add_trace(l2, row=1, col=1)
        idx_l2 = len(fig.data) - 1
        window_state_vlines_idx.append([idx_l1, idx_l2])

        # Params: одна линия
        l3 = go.Scatter(x=[t_start, t_start], y=[y_min_params, y_max_params],
                        mode='lines', line=dict(color='red', dash='dash', width=2),
                        showlegend=False, visible=(win==0))
        fig.add_trace(l3, row=2, col=1)
        idx_l3 = len(fig.data) - 1
        window_param_vlines_idx.append(idx_l3)

    # --- Слайдер ---
    always_visible = list(range(num_main_traces))
    steps = []
    for win in range(n_windows):
        visibility = [False] * len(fig.data)
        for idx in always_visible:
            visibility[idx] = True
        for idx in window_state_vlines_idx[win]:
            visibility[idx] = True
        visibility[window_param_vlines_idx[win]] = True

        step = dict(
            method='update',
            args=[{'visible': visibility}],
            label=str(win)
        )
        steps.append(step)

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={'prefix': 'Window: '},
            steps=steps,
            pad=dict(t=50)
        )],
        height=figsize[1],
        width=figsize[0],
        template='plotly_white',
        font=dict(size=fontsize),
        showlegend=True
    )

    #fig.update_xaxes(title_text='Time', row=1, col=1)
    fig.update_yaxes(title_text='State', row=1, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Parameter value', row=2, col=1)
    fig.update_annotations(font=dict(size=fontsize))
    return fig