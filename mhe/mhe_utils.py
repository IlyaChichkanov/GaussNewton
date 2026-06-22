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
import casadi as ca
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

import numpy as np
from mhe.mhe_base_model_interface import MheModel

class ArrivalCostUpdater:
    def __init__(self, mhe_model: MheModel, dt: float, Q_state_diag=1e-4,
                 initial_x0=None, initial_theta=None, initial_Sigma=None):
        self.mhe_model = mhe_model
        self.nx = mhe_model.state_length
        self.np = mhe_model.param_length
        self.n_aug = self.nx + self.np
        self.dt = dt

        # Функция одного шага с якобианами (создаётся один раз)
        self.step_func = mhe_model.create_step_function(dt, "ekf_step")

        # Начальная ковариация
        if initial_Sigma is not None:
            self.Sigma = initial_Sigma
        else:
            # Большая начальная неопределённость
            self.Sigma = np.eye(self.n_aug)

        # Начальная оценка расширенного состояния
        if initial_x0 is not None and initial_theta is not None:
            self.x_aug_est = np.hstack([initial_x0, initial_theta])
        else:
            self.x_aug_est = np.zeros(self.n_aug)

        # Ковариация шума процесса
        self.Q = np.diag(np.concatenate([
            np.full(self.nx, Q_state_diag),
            np.zeros(self.np)   # параметры постоянны
        ]))

    def _compute_full_A(self, x, theta, u):
        """Вычисляет матрицу A для расширенного состояния в точке (x,theta,u)."""
        res = self.step_func(x=x, theta=theta, u=u)
        J_x = np.array(res['Jx'])
        J_theta = np.array(res['Jtheta'])
        A = np.zeros((self.n_aug, self.n_aug))
        A[:self.nx, :self.nx] = J_x
        A[:self.nx, self.nx:] = J_theta
        A[self.nx:, self.nx:] = np.eye(self.np)
        return A

    def _predict_single_step(self, x, theta, u):
        """Один шаг EKF‑предсказания. Обновляет внутреннюю Sigma и x_aug_est."""
        A = self._compute_full_A(x, theta, u)
        # Предсказание ковариации
        Sigma_pred = A @ self.Sigma @ A.T + self.Q
        # Предсказание состояния (для следующего шага)
        x_next = np.array(self.step_func(x=x, theta=theta, u=u)["x_next"]).flatten()
        # Обновляем внутренние переменные
        self.Sigma = Sigma_pred
        self.x_aug_est = np.hstack([x_next, theta])
        return x_next, theta

    def predict_multistep(self, L, U_list):
        """
        L шагов EKF‑предсказания.
        U_list: массив управлений (L, nu) для шагов от предыдущего начального состояния к новому.
        После вызова self.Sigma и self.x_aug_est соответствуют началу нового окна.
        Возвращает P_aug = inv(Sigma) для передачи в MHE.
        """
        x = self.x_aug_est[:self.nx]
        theta = self.x_aug_est[self.nx:]
        for k in range(L):
            u = U_list[k, :]
            x, theta = self._predict_single_step(x, theta, u)
        return np.linalg.inv(self.Sigma)

    def correct(self, F_aug, x_aug_opt):
        """
        Шаг коррекции после MHE.
        F_aug: полная информационная матрица Фишера (n_aug × n_aug) текущего окна.
        x_aug_opt: оптимальная оценка начального состояния окна (nx+np,).
        """
        Y_prior = np.linalg.inv(self.Sigma)
        Y_post = Y_prior + F_aug
        self.Sigma = np.linalg.inv(Y_post)
        self.x_aug_est = x_aug_opt
        


def reset_mhe_solver(mhe_model: MheModel,
                     acados_solver_mhe: AcadosOcpSolver,
                     control_sequence: np.array,
                     initial_x0: np.array,
                     initial_theta: np.array,
                     horizon: int,
                     dt: float) -> None:
    """
    Сбрасывает начальное приближение для всех узлов MHE солвера,
    интегрируя модель вдоль горизонта с помощью create_step_function.
    """
    assert len(initial_x0) == mhe_model.state_length
    assert len(initial_theta) == mhe_model.param_length
    assert control_sequence.shape[0] >= horizon

    # Функция одного шага (без якобианов нам достаточно)
    step_func = mhe_model.create_step_function(dt, "reset_traj")
    x_sim = initial_x0.copy()

    for j in range(horizon):
        x_aug = np.hstack((x_sim, initial_theta))
        acados_solver_mhe.set(j, "x", x_aug)

        if j < horizon - 1:
            # step_func возвращает [x_next, J_x, J_theta] – берём первый элемент
            res = step_func(x=x_sim, theta=initial_theta, u=control_sequence[j, :])
            x_sim = np.array(res['x_next']).flatten()   # или res['x_next']


@dataclass
class MheIterationResult:
    """Хранит результаты одного окна MHE."""
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
    Q_state_diag: float = 1e-4,
    initial_Sigma: Optional[np.ndarray] = None,
    ridge_reg: float = 1e-6,
    plot: bool = False,
    progress_bar: bool = True,
) -> List[MheIterationResult]:
    """
    Запуск MHE на последовательности окон с EKF-обновлением априорной ковариации.
    """
    N_measurement = mhe_params.mhe_horizont
    nx = mhe_model.state_length
    n_theta = mhe_model.param_length
    results = []

    # --- Инициализация ArrivalCostUpdater ---
    updater = ArrivalCostUpdater(
        mhe_model, dt, Q_state_diag=Q_state_diag,
        initial_x0=np.zeros(nx),           # будет переопределено на первом окне
        initial_theta=initial_theta,
        initial_Sigma=initial_Sigma
    )

    first_window = True
    # Переменная для хранения управлений предыдущего окна (нужна для предсказания)
    control_sequence_prev = None

    iterator = range(num_windows)
    if progress_bar:
        iterator = tqdm(iterator, desc="MHE windows", unit="window")

    for iter_idx in iterator:
        # 1. Получаем данные окна
        t_batch, control_sequence, state_sequence, _ = get_window_func(iter_idx)

        if first_window:
            initial_x0 = get_initial_state_func(
                state_sequence[0], control_sequence[0], initial_theta
            )
            updater.x_aug_est = np.hstack([initial_x0, initial_theta])
            first_window = False

        # 3. Предсказание EKF (кроме первого окна)
        if iter_idx > 0:
            # Число шагов от начала предыдущего окна до начала текущего
            L = N_measurement - overlap_points
            # Управления, которые переводят систему из предыдущего начального состояния в новое
            P_aug = updater.predict_multistep(L, control_sequence_prev[:L, :])
        else:
            P_aug = np.linalg.inv(updater.Sigma)

        # Сохраняем управления этого окна для следующего шага
        control_sequence_prev = control_sequence.copy()

        # 4. Установка параметров солвера и сброс траектории
        set_mhe_solver(
            mhe_model, acados_solver_factory,
            state_sequence, control_sequence,
            updater.x_aug_est[:nx], updater.x_aug_est[nx:],
            N_measurement, P_aug
        )
        reset_mhe_solver(
            mhe_model, acados_solver_factory,
            control_sequence,
            updater.x_aug_est[:nx], updater.x_aug_est[nx:],
            N_measurement, dt
        )

        # 5. Решение MHE
        status = acados_solver_factory.solve()
        if status != 0:
            msg = f"Window {iter_idx}: acados returned status {status}. Skipping."
            logger.warning(msg)
            continue

        # 6. Извлечение результатов
        mhe_output = get_mhe_estimated_data(mhe_model, acados_solver_factory, N_measurement)
        sim_x_est = mhe_output.sim_x_est
        sim_w_est = mhe_output.sim_w_est
        theta_opt = np.asarray(mhe_output.sim_param_est).flatten()

        # 7. Полная FIM для коррекции
        x0_opt = np.asarray(sim_x_est[0]).flatten() # начальное состояние окна (оптимальное)
        N_actual = control_sequence.shape[0] - overlap_points
        F_aug = mhe_model.compute_augmented_fim(
            N_actual, dt, control_sequence[:N_actual, :],
            x0_opt.flatten(), theta_opt.flatten(), r_inv
        )
        F_aug_reg, eigvals = regularize_fim(F_aug, ridge=ridge_reg)

        # 8. Коррекция ковариации
        x_aug_opt = np.hstack([x0_opt, theta_opt])
        updater.correct(F_aug_reg, x_aug_opt)

        # 9. Сохранение результатов
        result = MheIterationResult(
            t_batch=t_batch,
            state_sequence=state_sequence,
            control_sequence=control_sequence,
            state_est=sim_x_est,
            noise_est=sim_w_est,
            param_est=theta_opt,
            cov_matrix = updater.Sigma[nx:, nx:].flatten(),  # для графика апостериорная ковариация
            fim=F_aug_reg,
            eigvals=eigvals,
            status=status,
            cost_value=mhe_output.cost_value,
            sqp_iter=mhe_output.sqp_iter
        )
        results.append(result)

        # 10. Опциональный график
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(t_batch, state_sequence, 'g', label='measured')
            plt.plot(t_batch, sim_x_est[:-1], 'b', label='estimated')
            plt.title(f"Window {iter_idx}")
            plt.legend()
            plt.show()

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
                      fontsize=9)

    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('MHE data windows', fontsize=14)
    plt.tight_layout()
    plt.show()


@dataclass
class MheEstimationData:
    """Container for MHE estimation results."""
    sim_x_est: np.ndarray      # (N+1, nx) – estimated states at all nodes
    sim_w_est: np.ndarray      # (N, nx)   – estimated process noise at each step
    sim_param_est: np.ndarray # (N+1, param_length) – estimated parameters at all nodes
    cost_value: float        # final cost value
    sqp_iter: int            # number of SQP iterations

def regularize_fim(m, ridge=1.0):
    m = (m + m.T) / 2.0
    eigvals, eigvecs = la.eigh(m)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    m_reg = m + ridge * np.eye(m.shape[0])
    #F_reg = ridge * np.eye(F.shape[0])
    return m_reg, eigvals

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


def get_mhe_estimated_data(mhe_model: MheModel, acados_solver_mhe: AcadosOcpSolver, horizon: int):
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

def plot_mhe_results(results, overlap=0, initial_params=None, initial_std=None,
                     theta_true=None,
                     plot_states=True, plot_params=True,
                     plot_eigvals=True, plot_noise=True,
                     plot_cost=True, plot_iter=True, plot_status=True,
                     plot_cov_matrix=True,
                     figsize=(15, 15), verbose=False):
    """
    Визуализация результатов MHE с интерактивным слайдером времени
    и автоматическим масштабированием оси Y.

    Параметры:
    -----------
    results : list of MheIterationResult
        Список объектов с результатами каждого окна.
    overlap : int
        Число перекрывающихся точек между окнами.
    initial_params : np.ndarray или None
        Начальная оценка параметров (до первого окна) – размер (n_theta,).
    initial_std : np.ndarray или None
        Начальные стандартные отклонения параметров – размер (n_theta,).
        Если None, используется std первого окна (с предупреждением).
    theta_true : np.ndarray или None
        Истинные значения параметров (для отрисовки горизонтальных линий).
    plot_* : bool
        Флаги включения отдельных графиков.
    figsize : tuple
        Размер фигуры.
    verbose : bool
        Печатать отладочную информацию.
    """
    if not results:
        print("No results to plot.")
        return

    # -------- Определение активных панелей --------
    active_plots = []
    if plot_states:   active_plots.append('states')
    if plot_params:   active_plots.append('params')
    if plot_eigvals:  active_plots.append('eigvals')
    if plot_noise:    active_plots.append('noise')
    if plot_cost:     active_plots.append('cost')
    if plot_iter:     active_plots.append('iter')
    if plot_status:   active_plots.append('status')
    if plot_cov_matrix: active_plots.append('cov_matrix')

    n_plots = len(active_plots)
    if n_plots == 0:
        print("Nothing to plot.")
        return

    # Высоты: первая панель (если 'states') – 2, остальные – 1, слайдер – 0.2
    ratios = [2 if p == 'states' else 1 for p in active_plots] + [0.2]
    fig, axs = plt.subplots(n_plots + 1, 1, figsize=figsize,
                            gridspec_kw={'height_ratios': ratios},
                            squeeze=False)
    axs = axs.flatten()
    slider_ax = axs[-1]
    plot_axes = axs[:-1]
    axes_dict = dict(zip(active_plots, plot_axes))

    # -------- Склейка данных с учётом перекрытия --------
    t_full, measured_full, estimated_full = [], [], []
    params_full, std_full = [], []

    for idx, res in enumerate(results):
        t = np.asarray(res.t_batch)
        meas = np.asarray(res.state_sequence)
        est = np.asarray(res.state_est)
        params = np.asarray(res.param_est)

        n_points = min(len(t), len(est), len(meas))
        t = t[:n_points]; meas = meas[:n_points]; est = est[:n_points]

        # Восстанавливаем std для этого окна
        cov_flat = res.cov_matrix
        n_theta = len(params) if params.ndim == 1 else params.shape[1]
        cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
        diag = np.diag(cov_mat)
        std_window = np.sqrt(np.maximum(diag, 0.0))

        # Размножаем параметры и std на все временные точки окна
        if params.ndim == 1:
            params_2d = np.tile(params, (n_points, 1))
        else:
            params_2d = params[:n_points]
        std_2d = np.tile(std_window, (n_points, 1))

        # Учёт перекрытия
        start = 0 if idx == 0 else min(overlap, n_points) if overlap < n_points else n_points

        t_full.extend(t[start:])
        measured_full.extend(meas[start:])
        estimated_full.extend(est[start:])
        params_full.extend(params_2d[start:])
        std_full.extend(std_2d[start:])

        if verbose:
            print(f"Window {idx}: n_points={n_points}, start={start}, added={n_points - start}")

    # Приводим к массивам и одинаковой длине
    t_full = np.array(t_full)
    measured_full = np.array(measured_full)
    estimated_full = np.array(estimated_full)
    params_full = np.array(params_full)
    std_full = np.array(std_full)

    min_len = min(len(t_full), len(measured_full), len(estimated_full),
                  len(params_full), len(std_full))
    t_full = t_full[:min_len]
    measured_full = measured_full[:min_len]
    estimated_full = estimated_full[:min_len]
    params_full = params_full[:min_len]
    std_full = std_full[:min_len]

    # -------- Отрисовка графиков --------
    # 1. Состояния
    if 'states' in axes_dict:
        ax = axes_dict['states']
        n_obs = measured_full.shape[1]
        n_x = estimated_full.shape[1]
        for i in range(n_obs):
            ax.plot(t_full, measured_full[:, i], '--', label=f'Meas y_{i+1}')
        for i in range(n_x):
            ax.plot(t_full, estimated_full[:, i], '-', label=f'Est x_{i+1}')
        ax.set_title("States: Measured (dashed) vs Estimated (solid)")
        ax.set_xlabel("Time"); ax.set_ylabel("State")
        ax.legend(); ax.grid(True)

    # 2. Параметры с доверительными интервалами
    if 'params' in axes_dict:
        ax = axes_dict['params']
        # Число параметров определяем по данным окон, а если их нет, то по initial_params или theta_true
        if params_full.size > 0:
            ntheta = params_full.shape[1]
        elif initial_params is not None:
            ntheta = len(initial_params)
        elif theta_true is not None:
            ntheta = len(theta_true)
        else:
            ntheta = 0

        if ntheta == 0:
            print("Warning: Could not determine number of parameters, skipping 'params' plot.")
            axes_dict['params'].set_visible(False)
        else:
            t_plot = t_full.copy()
            p_plot = params_full.copy()
            s_plot = std_full.copy()

            # Добавляем начальную точку, если заданы initial_params
            if initial_params is not None:
                if len(t_plot) > 1:
                    dt = t_plot[1] - t_plot[0]
                else:
                    # Если всего одна точка окна, берём средний шаг из первого окна
                    dt = np.mean(np.diff(results[0].t_batch)) if len(results[0].t_batch) > 1 else 1.0
                t_start = t_plot[0] - dt

                # Проверка/задание начального std
                if initial_std is not None:
                    init_std = np.asarray(initial_std).flatten()
                    if len(init_std) != ntheta:
                        raise ValueError(f"initial_std length {len(init_std)} != ntheta {ntheta}")
                else:
                    import warnings
                    warnings.warn("initial_std not provided, using std of first window for initial point.")
                    init_std = s_plot[0] if len(s_plot) > 0 else np.zeros(ntheta)

                # Вставляем начальные значения
                p_plot = np.vstack([initial_params, p_plot])
                s_plot = np.vstack([init_std, s_plot])
                t_plot = np.insert(t_plot, 0, t_start)

            for i in range(ntheta):
                ax.plot(t_plot, p_plot[:, i], label=f'θ_{i+1} estimated')
                ax.fill_between(t_plot,
                                p_plot[:, i] - s_plot[:, i],
                                p_plot[:, i] + s_plot[:, i],
                                alpha=0.2, label=f'θ_{i+1} ±1σ' if i == 0 else "")
            if theta_true is not None:
                for i, val in enumerate(theta_true):
                    ax.axhline(y=val, linestyle=':', color=f'C{i}', alpha=0.8,
                               label=f'θ_{i+1} true')
            ax.set_title("Parameter estimates over time (shaded: ±1σ)")
            ax.set_xlabel("Time"); ax.set_ylabel("Parameter value")
            ax.legend(); ax.grid(True)

    # 3. Собственные числа FIM
    if 'eigvals' in axes_dict:
        ax = axes_dict['eigvals']
        n_theta = results[0].eigvals.shape[0]
        eig_vals_matrix = np.array([res.eigvals for res in results])
        eig_vals_sorted = np.sort(eig_vals_matrix, axis=1)[:, ::-1]
        for i in range(n_theta):
            ax.semilogy(eig_vals_sorted[:, i], marker='o', label=f'λ_{i+1}')
        ax.set_title("FIM eigenvalues per window (log scale)")
        ax.set_xlabel("Window index"); ax.set_ylabel("Eigenvalue magnitude")
        ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # 4. Стандартные отклонения параметров (отдельный график)
    if 'cov_matrix' in axes_dict:
        ax = axes_dict['cov_matrix']
        n_theta = results[0].eigvals.shape[0]
        std_per_window = []
        for res in results:
            cov_flat = res.cov_matrix
            cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
            diag = np.diag(cov_mat)
            std = np.sqrt(np.maximum(diag, 0.0))
            std_per_window.append(std)
        std_per_window = np.array(std_per_window)
        for i in range(n_theta):
            ax.plot(range(len(std_per_window)), std_per_window[:, i], marker='o',
                    label=f'θ_{i+1} std')
        ax.set_title("Parameter standard deviation (sqrt of diag(cov))")
        ax.set_xlabel("Window index"); ax.set_ylabel("Standard deviation")
        ax.legend(); ax.grid(True)

    # 5. Распределение шума процесса
    if 'noise' in axes_dict:
        ax = axes_dict['noise']
        all_noise = np.concatenate([res.noise_est.flatten() for res in results])
        ax.hist(all_noise, bins=50, alpha=0.7, density=True)
        ax.set_title("Process noise distribution")
        ax.set_xlabel("Noise value"); ax.set_ylabel("Density")
        ax.grid(True)

    # 6. Стоимость
    if 'cost' in axes_dict:
        ax = axes_dict['cost']
        cost_vals = [res.cost_value for res in results]
        ax.plot(range(len(cost_vals)), cost_vals, marker='o')
        ax.set_title("Cost value per window")
        ax.set_xlabel("Window index"); ax.set_ylabel("Cost")
        ax.grid(True); ax.set_yscale('log')

    # 7. Итерации SQP
    if 'iter' in axes_dict:
        ax = axes_dict['iter']
        iter_vals = [res.sqp_iter for res in results]
        ax.plot(range(len(iter_vals)), iter_vals, marker='o')
        ax.set_title("SQP iterations per window")
        ax.set_xlabel("Window index"); ax.set_ylabel("Iterations")
        ax.grid(True)

    # 8. Статус солвера
    if 'status' in axes_dict:
        ax = axes_dict['status']
        status_vals = [res.status for res in results]
        ax.plot(range(len(status_vals)), status_vals, marker='o', linestyle='-')
        ax.set_title("Solver status per window (0 = success)")
        ax.set_xlabel("Window index"); ax.set_ylabel("Status")
        ax.set_yticks(sorted(set(status_vals)))
        ax.grid(True)

    # -------- Интерактивный слайдер времени с авто-масштабированием Y --------
    time_axes = []
    if 'states' in axes_dict:
        time_axes.append(axes_dict['states'])
    if 'params' in axes_dict:
        time_axes.append(axes_dict['params'])

    if time_axes and len(t_full) > 1:
        t_min, t_max = t_full.min(), t_full.max()
        slider = RangeSlider(slider_ax, "Time", t_min, t_max, valinit=(t_min, t_max))

        def update_ylims(ax, t_left, t_right):
            """Пересчитывает пределы Y по видимым данным."""
            lines = ax.get_lines()
            if not lines:
                return
            y_min = np.inf
            y_max = -np.inf
            for line in lines:
                xdata = np.asarray(line.get_xdata())
                ydata = np.asarray(line.get_ydata())
                mask = (xdata >= t_left) & (xdata <= t_right)
                if mask.any():
                    y_min = min(y_min, np.nanmin(ydata[mask]))
                    y_max = max(y_max, np.nanmax(ydata[mask]))
            if np.isfinite(y_min) and np.isfinite(y_max):
                margin = 0.05 * (y_max - y_min) or 0.1 * max(abs(y_min), 1e-6)
                ax.set_ylim(y_min - margin, y_max + margin)

        def update_xlim(val):
            t_left, t_right = val
            for ax in time_axes:
                ax.set_xlim(t_left, t_right)
                update_ylims(ax, t_left, t_right)
            fig.canvas.draw_idle()

        slider.on_changed(update_xlim)
        # Инициализируем Y пределы для полного диапазона
        for ax in time_axes:
            update_ylims(ax, t_min, t_max)
        slider_ax.set_xlabel('Time')
        slider_ax.xaxis.set_label_coords(0.5, -0.5)
    else:
        slider_ax.set_visible(False)

    plt.tight_layout()
    plt.show()