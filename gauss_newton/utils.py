from matplotlib.gridspec import GridSpec
import numpy as np
from gauss_newton.gauss_newton_math import TimeIntervalManager, MultipleShooting
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Optional
# Предполагается, что TimeIntervalManager уже определён (например, в отдельном модуле)
# from your_module import TimeIntervalManager

def plot_solution(
    fig: plt.Figure,
    problem: 'MultipleShooting',
    theta_hist: List[np.ndarray],
    *,
    plot_xy: bool = False,          # рисовать временные ряды всех состояний
    plot_theta: bool = True,
    plot_true_solution: bool = False,
    plot_residuals: bool = False,
    r_meas_hist: Optional[np.ndarray] = None,
    r_cont_hist: Optional[np.ndarray] = None,
    index: int = -1,
    theta_true: Optional[np.ndarray] = None
) -> None:
    """
    Визуализация результатов multiple shooting.
    Фазовая траектория занимает первую (увеличенную) строку.
    При plot_xy=True временные ряды всех состояний рисуются в отдельных строках.
    """
    fontsize = 20
    n_batches = len(problem.t_eval_measurements_batches)
    n_state, n_theta, _ = problem.system.get_dimentions()

    # ---- 1. Определение количества строк и весов высоты ----
    total_rows = 1                     # фазовая траектория
    if plot_xy:
        total_rows += n_state          # по одной строке на каждое состояние
    if plot_theta:
        total_rows += 1
    if plot_residuals:
        total_rows += 1

    # Веса высот: первая строка в 3 раза выше остальных
    height_ratios = [3] + [1] * (total_rows - 1)

    fig.clf()
    gs = GridSpec(total_rows, n_batches, figure=fig,
                  height_ratios=height_ratios, hspace=0.3, wspace=0.1)

    # ---- 2. Фазовая траектория (первая строка, объединяет все колонки) ----
    use_3d = n_state >= 3
    if use_3d:
        ax_traj = fig.add_subplot(gs[0, :], projection='3d')
        ax_traj.set_zlabel('z', fontsize=fontsize)
        plt.setp(ax_traj.get_zticklabels(), fontsize=fontsize)
    else:
        ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.set_xlabel('x', fontsize=fontsize)
    ax_traj.set_ylabel('y', fontsize=fontsize)
    ax_traj.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_traj.grid(True)

    # ---- 3. Оси для временных рядов каждого состояния ----
    axes_states = []   # axes_states[state_idx][batch_idx]
    if plot_xy:
        for state_idx in range(n_state):
            row_axes = []
            for batch in range(n_batches):
                ax = fig.add_subplot(gs[1 + state_idx, batch])
                ax.set_xlabel('time, s', fontsize=fontsize)
                ax.set_ylabel(f'state_{state_idx}', fontsize=fontsize)
                ax.grid(True)
                plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                row_axes.append(ax)
            axes_states.append(row_axes)

    # ---- 4. Ось для сходимости параметров ----
    if plot_theta:
        # Строка параметров: после строк состояний, но до невязок (если есть)
        row_theta = 1 + (n_state if plot_xy else 0)
        if plot_residuals:
            # Если есть невязки, параметры располагаются над ними (предпоследняя строка)
            row_theta = total_rows - 2
        ax_theta = fig.add_subplot(gs[row_theta, :])
        ax_theta.set_xlabel('iter', fontsize=fontsize)
        ax_theta.set_ylabel('Estimated coefs', fontsize=fontsize)
        ax_theta.grid(True)
        plt.setp(ax_theta.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax_theta.get_yticklabels(), fontsize=fontsize)

    # ---- 5. Ось для невязок ----
    if plot_residuals:
        ax_res = fig.add_subplot(gs[-1, :])
        ax_res.set_xlabel('iter', fontsize=fontsize)
        ax_res.set_ylabel('Residual norm', fontsize=fontsize)
        ax_res.grid(True)
        ax_res.set_yscale('log')
        plt.setp(ax_res.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax_res.get_yticklabels(), fontsize=fontsize)
        if r_meas_hist is not None:
            ax_res.plot(r_meas_hist, 'b-o', label='Measurement', linewidth=2)
        if r_cont_hist is not None:
            ax_res.plot(r_cont_hist, 'r-s', label='Continuity', linewidth=2)
        ax_res.legend(fontsize=fontsize)

    # ---- Вспомогательная функция для рисования фазовой траектории ----
    def _plot_traj(ax, states, is_3d, **kwargs):
        if is_3d and states.shape[1] >= 3:
            ax.plot(states[:, 0], states[:, 1], states[:, 2], **kwargs)
        elif states.shape[1] >= 2:
            ax.plot(states[:, 0], states[:, 1], **kwargs)
        else:
            ax.plot(states[:, 0], np.zeros_like(states[:, 0]), **kwargs)

    # ---- Подготовка данных ----
    if index < 0:
        index = len(theta_hist) - 1
    if index >= len(theta_hist):
        raise IndexError(f"index {index} вне диапазона theta_hist")

    theta_full = theta_hist[index]
    theta = theta_full[:n_theta]

    def get_traj(c0, t_eval):
        """Возвращает траекторию формы (len(t_eval), n_state)."""
        if problem.use_jax:
            sol = problem.system.get_solution_jax(c0, theta, t_eval)
        else:
            sol = problem.system.get_solution(c0, theta, t_eval)
        return sol.T   # (n_t, n_state)

    # ---- Истинное решение (если требуется) ----
    if plot_true_solution:
        if not hasattr(problem, 'full_trajectory') or problem.full_trajectory is None:
            raise AttributeError("problem.full_trajectory is None")
        _, state_true_batches = problem.full_trajectory
        for state_true in state_true_batches:
            # state_true должен быть формы (n_t, n_state)
            _plot_traj(ax_traj, state_true, use_3d, color='grey', label='True solution')

    # ---- Основной цикл по батчам и шутам (отрисовка оценённых траекторий) ----
    for batch, (state_full, state_measured, t_meas) in enumerate(
        zip(problem.state_full_batches, problem.state_measured_batches, problem.t_eval_measurements_batches)
    ):
        print(f'batch {batch}')

        # Используем тот же TimeIntervalManager, что и в solve_batch
        tm = TimeIntervalManager(problem.N_shoot, t_meas)
        n_shoot = tm.N_shoot

        for shoot in range(n_shoot):
            # Начальное состояние для этого шута
            start_idx = n_theta + problem.N_shoot * batch * n_state + shoot * n_state
            c0 = theta_full[start_idx:start_idx + n_state]

            # Интервал без перекрытия
            t_interval, meas_idx = tm.get_time_interval(shoot)
            t_start = t_interval[0]
            t_end = t_interval[-1]

            # Плотная сетка для гладкой отрисовки
            t_dense = np.linspace(t_start, t_end, 500)
            traj = get_traj(c0, t_dense)   # (500, n_state)

            # Фазовая траектория
            _plot_traj(ax_traj, traj, use_3d, color='blue', alpha=0.7)

            # Временные ряды всех состояний
            if plot_xy:
                for state_idx in range(n_state):
                    axes_states[state_idx][batch].plot(t_dense, traj[:, state_idx],
                                                       'b-', alpha=0.7)

        # ---- Измерения (зелёные крестики) ----
        # Фазовая плоскость
        if state_full.shape[1] >= 2:
            if use_3d and state_full.shape[1] >= 3:
                ax_traj.scatter(state_full[:, 0], state_full[:, 1], state_full[:, 2],
                                color='green', marker='x', s=10)
            else:
                ax_traj.scatter(state_full[:, 0], state_full[:, 1],
                    color='green', marker='x', s=10,
                    label='Measurements' if batch == 0 else "")

        # Временные ряды измерений
        if plot_xy:
            for state_idx in range(n_state):
                axes_states[state_idx][batch].scatter(t_meas, state_full[:, state_idx],
                                                      color='green', marker='x', s=10)

    # Легенда для фазовой траектории
    ax_traj.legend(fontsize=fontsize)

    # ---- График сходимости параметров ----
    if plot_theta:
        theta_history = np.array(theta_hist)[:index+1, :n_theta]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(theta_history.shape[1]):
            color = colors[i % len(colors)]
            if theta_true is not None and i < len(theta_true):
                ax_theta.axhline(y=theta_true[i], linestyle='--', color=color, alpha=0.7)
            ax_theta.step(range(len(theta_history)), theta_history[:, i],
                          where='post', color=color, label=f'$\\theta_{i}$')
        ax_theta.legend(fontsize=fontsize)

    plt.tight_layout()