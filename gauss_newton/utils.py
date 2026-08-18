from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import to_rgb

def plot_solution(
    fig: Optional[go.Figure] = None,          # теперь fig – plotly Figure (может быть None)
    problem: 'MultipleShooting' = None,
    theta_hist: List[np.ndarray] = None,
    *,
    plot_xy: bool = False,
    plot_theta: bool = True,
    plot_true_solution: bool = False,
    plot_residuals: bool = False,
    plot_measurements: bool = False,
    plot_trajectory: bool = True,
    plot_param_indices: Optional[List[int]] = None,
    r_meas_hist: Optional[np.ndarray] = None,
    r_cont_hist: Optional[np.ndarray] = None,
    index: int = -1,
    theta_true: Optional[np.ndarray] = None,
    ci_low_hist: Optional[np.ndarray] = None,
    ci_high_hist: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,       # теперь используется
    state_names: Optional[List[str]] = None,       # новое: имена компонент состояния
    fontsize = 10
) -> go.Figure:
    """
    Построение графиков идентификации с помощью Plotly.
    Возвращает объект Figure, который можно отобразить или экспортировать.
    """
    if problem is None or theta_hist is None:
        raise ValueError("problem and theta_hist must be provided")

    n_batches = len(problem.t_eval_measurements_batches)
    n_state, n_theta, n_obs = problem.system.dims()

    # Имена переменных (если не заданы – используем индексы)
    if state_names is None:
        state_names = [f'state{i}' for i in range(n_state)]
    # else:
    #     if len(state_names) != n_state:
    #         raise ValueError(f"state_names must have length {n_state}, got {len(state_names)}")

    if param_names is None:
        param_names = [f'θ_{i}' for i in range(n_theta)]
    else:
        if len(param_names) != n_theta:
            raise ValueError(f"param_names must have length {n_theta}, got {len(param_names)}")

    # Режим одномерного состояния
    is_single = n_state == 1
    if is_single and plot_xy:
        import warnings
        warnings.warn("plot_xy is ignored when n_state == 1 (state vs time is already shown)")
        plot_xy = False

    # Определяем количество строк и их типы
    total_rows = 0
    row_types = []          # 'scene' для 3d, 'xy' для 2d
    if plot_trajectory:
        total_rows += 1
        use_3d = (not is_single) and (n_state >= 3)
        row_types.append('scene' if use_3d else 'xy')
    if plot_xy and not is_single:
        for _ in range(n_obs):
            total_rows += 1
            row_types.append('xy')
    if plot_theta:
        total_rows += 1
        row_types.append('xy')
    if plot_residuals:
        total_rows += 1
        row_types.append('xy')

    if total_rows == 0:
        raise ValueError("Не выбрано ни одного графика для отображения")

    # Высоты строк (первая строка в 3 раза выше, если траектория)
    row_heights = []
    if plot_trajectory:
        row_heights.append(3)
    row_heights.extend([1] * (total_rows - len(row_heights)))

    # Подготовка subplots
    # Если plot_xy, то для каждой строки plot_xy нужно n_batches колонок, а остальные строки занимают все колонки
    cols_per_row = []
    for i, rtype in enumerate(row_types):
        if plot_xy and not is_single and i > (0 if plot_trajectory else -1) and i < (total_rows - (1 if plot_residuals else 0) - (1 if plot_theta else 0)):
            # Это строка plot_xy – для неё колонок n_batches
            cols_per_row.append(n_batches)
        else:
            cols_per_row.append(1)

    # make_subplots принимает rows, cols как числа, если cols разное – нужно передать список cols? 
    # В Plotly make_subplots позволяет задать cols для всей сетки, но у нас разное число колонок в строках.
    # Решение: задать максимальное число колонок (n_batches, если plot_xy, иначе 1) и объединять ячейки в строках, где нужна одна колонка.
    max_cols = n_batches if (plot_xy and not is_single) else 1
    specs = []
    row_idx = 0
    if plot_trajectory:
        specs.append([{'type': row_types[row_idx], 'colspan': max_cols}] + [None]*(max_cols-1))
        row_idx += 1
    if plot_xy and not is_single:
        for _ in range(n_obs):
            specs.append([{'type': row_types[row_idx]}]*max_cols)
            row_idx += 1
    if plot_theta:
        specs.append([{'type': row_types[row_idx], 'colspan': max_cols}] + [None]*(max_cols-1))
        row_idx += 1
    if plot_residuals:
        specs.append([{'type': row_types[row_idx], 'colspan': max_cols}] + [None]*(max_cols-1))
        row_idx += 1

    fig = make_subplots(
        rows=total_rows,
        cols=max_cols,
        specs=specs,
        row_heights=row_heights,
        # column_widths можно не задавать
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    # Настройка стиля по умолчанию

    fig.update_layout(
        width=1400, height=1000,
        font=dict(size=fontsize),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60),
    )

    # Индексация строк в fig
    current_row = 1   # в plotly строки нумеруются с 1

    # ------------------------------------------------------------
    # Вспомогательная функция для фазовой траектории
    def _plot_traj(row, col, states, name=None, color='blue', is_3d=False):
        if is_3d and states.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=states[:, 0], y=states[:, 1], z=states[:, 2],
                    mode='lines', line=dict(color=color, width=2),
                    name=name,
                    showlegend=(name is not None),
                ),
                row=row, col=col
            )
        elif states.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=states[:, 0], y=states[:, 1],
                    mode='lines', line=dict(color=color, width=2),
                    name=name,
                ),
                row=row, col=col
            )
        # Если одна компонента – не вызываем, т.к. будет временной ряд

    # ------------------------------------------------------------
    # Данные для итераций
    if index < 0:
        index = len(theta_hist) - 1
    if index >= len(theta_hist):
        raise IndexError(f"index {index} out of range for theta_hist")

    theta_full = theta_hist[index]
    theta = theta_full[:n_theta]

    def get_traj(c0, t_eval):
        if problem.use_jax:
            sol = problem.system.get_solution_jax(c0, theta, t_eval)
        else:
            sol = problem.system.get_solution(c0, theta, t_eval)
        return sol.T  # (n_time, n_state)

    # ------------------------------------------------------------
    # Истинное решение (если требуется)
    if plot_true_solution and plot_trajectory and not is_single:
        if not hasattr(problem, 'full_trajectory') or problem.full_trajectory is None:
            raise AttributeError("problem.full_trajectory is None")
        _, state_true_batches = problem.full_trajectory
        for state_true in state_true_batches:
            _plot_traj(current_row, 1, state_true, name='True', color='grey', is_3d=use_3d)

    # ------------------------------------------------------------
    # Цикл по батчам и шутам – построение траекторий и измерений
    for batch_idx, (state_measured, t_meas) in enumerate(
        zip(problem.state_measured_batches, problem.t_eval_measurements_batches)
    ):

        # Измерения (точки)
        if plot_measurements:
            # На верхнюю панель, если траектория
            if plot_trajectory:
                if is_single:
                    fig.add_trace(
                        go.Scatter(
                            x=t_meas, y=state_measured[:, 0],
                            mode='markers', marker=dict(color='green', symbol='x', size=8),
                            name='Measurements' if batch_idx == 0 else None,
                            showlegend=(batch_idx == 0),
                        ),
                        row=current_row, col=1
                    )
                else:
                    if use_3d and state_measured.shape[1] >= 3:
                        fig.add_trace(
                            go.Scatter3d(
                                x=state_measured[:, 0], y=state_measured[:, 1], z=state_measured[:, 2],
                                mode='markers', marker=dict(color='green', symbol='x', size=5),
                                name='Meas',
                            ),
                            row=current_row, col=1
                        )
                    elif state_measured.shape[1] >= 2:
                        fig.add_trace(
                            go.Scatter(
                                x=state_measured[:, 0], y=state_measured[:, 1],
                                mode='markers', marker=dict(color='green', symbol='x', size=8),
                                name='Measurements' if batch_idx == 0 else None,
                                showlegend=(batch_idx == 0),
                            ),
                            row=current_row, col=1
                        )
            # Измерения на графиках plot_xy
            if plot_xy and not is_single:
                base_row = current_row + (1 if plot_trajectory else 0)
                for obs_idx in range(n_obs):
                    fig.add_trace(
                        go.Scatter(
                            x=t_meas, y=state_measured[:, obs_idx],
                            mode='markers', marker=dict(color='green', symbol='x', size=6),
                            name=f'meas',
                            showlegend=False,
                        ),
                        row=base_row + obs_idx, col=batch_idx + 1
                    )
        tm = problem.interval_managers[batch_idx]
        n_shoot = tm.N_shoot
        c_offset = n_theta + sum(im.N_shoot for im in problem.interval_managers[:batch_idx]) * n_state

        for shoot in range(n_shoot):
            start_idx = c_offset + shoot * n_state
            c0 = theta_full[start_idx:start_idx + n_state]

            t_interval, _ = tm.get_time_interval(shoot)
            t_start, t_end = t_interval[0], t_interval[-1]

            t_dense = np.linspace(t_start, t_end, 500)
            traj = get_traj(c0, t_dense)   # (500, n_state)

            # Вычисляем наблюдения вдоль траектории
            meas_traj = []
            for t_, state in zip(t_dense, traj):
                meas_traj.append(problem.system.h(state, t_, theta))
            meas_traj = np.array(meas_traj)   # (500, n_obs)

            # Рисуем траекторию на верхней панели
            if plot_trajectory:
                if is_single:
                    # Временной ряд: измерение от времени
                    fig.add_trace(
                        go.Scatter(
                            x=t_dense, y=meas_traj[:, 0],
                            mode='lines', line=dict(color='blue', width=2),
                            name=f'batch {batch_idx}' if shoot == 0 else None,
                            showlegend=(shoot == 0),
                        ),
                        row=current_row, col=1
                    )
                else:
                    # Фазовая траектория
                    _plot_traj(current_row, 1, traj, name=f'batch {batch_idx}' if shoot == 0 else None,
                               color='blue', is_3d=use_3d)

            # Временные ряды для plot_xy (n_state>1)
            if plot_xy and not is_single:
                # Определяем строку, соответствующую этому наблюдению
                # plot_xy строки идут после траектории (если она есть)
                base_row = current_row + (1 if plot_trajectory else 0)  # начало строк plot_xy
                for obs_idx in range(n_obs):
                    fig.add_trace(
                        go.Scatter(
                            x=t_dense, y=meas_traj[:, obs_idx],
                            mode='lines', line=dict(color='blue', width=1.5),
                            name=f'batch {batch_idx}' if shoot == 0 else None,
                            showlegend=False,
                        ),
                        row=base_row + obs_idx, col=batch_idx + 1   # колонки с 1
                    )

    # Переход к следующей строке после траектории и plot_xy
    current_row += (1 if plot_trajectory else 0) + (n_obs if (plot_xy and not is_single) else 0)

    # ------------------------------------------------------------
    # График сходимости параметров
    if plot_theta:
        theta_history_full = np.array(theta_hist)[:index+1, :n_theta]  # (n_iter, n_theta)
        if plot_param_indices is not None:
            indices = list(plot_param_indices)
            if max(indices) >= n_theta or min(indices) < 0:
                raise IndexError(f"Индексы параметров должны быть в диапазоне 0..{n_theta-1}")
            theta_history = theta_history_full[:, indices]
            theta_true_filtered = theta_true[indices] if theta_true is not None else None
            ci_low_filt = ci_low_hist[:index+1, indices] if ci_low_hist is not None else None
            ci_high_filt = ci_high_hist[:index+1, indices] if ci_high_hist is not None else None
            labels = [param_names[i] for i in indices]
        else:
            theta_history = theta_history_full
            theta_true_filtered = theta_true
            ci_low_filt = ci_low_hist[:index+1, :] if ci_low_hist is not None else None
            ci_high_filt = ci_high_hist[:index+1, :] if ci_high_hist is not None else None
            labels = param_names

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(theta_history.shape[1]):
            color = colors[i % len(colors)]
            # Доверительный интервал – добавляем полупрозрачную область
            if ci_low_filt is not None and ci_high_filt is not None:
                # верхняя граница (невидимая)
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(ci_low_filt)),
                        y=ci_high_filt[:, i],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=current_row, col=1
                )
                # нижняя граница с заливкой до предыдущей линии
                r, g, b = to_rgb(color)
                fillcolor = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.2)'
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(ci_low_filt)),
                        y=ci_low_filt[:, i],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=current_row, col=1
                )
            # График параметра (ступенчатый)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(theta_history)),
                    y=theta_history[:, i],
                    mode='lines+markers',
                    line=dict(color=color, shape='hv'),
                    name=labels[i],
                    legendgroup='parameters',                    # новая группа
                    legendgrouptitle=dict(text='Parameters'),    # заголовок группы
                ),
                row=current_row, col=1
            )
            # Истинное значение (горизонтальная линия) – без легенды
            if theta_true_filtered is not None and i < len(theta_true_filtered):
                fig.add_hline(
                    y=theta_true_filtered[i],
                    line_dash='dash',
                    line_color=color,
                    opacity=0.7,
                    row=current_row, col=1,
                )
        # Обновление меток осей
        fig.update_xaxes(title_text='Iteration', row=current_row, col=1)
        fig.update_yaxes(title_text='Parameter value', row=current_row, col=1)
        current_row += 1

    # ------------------------------------------------------------
    # График невязок
    if plot_residuals:
        if r_meas_hist is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(r_meas_hist)),
                    y=r_meas_hist,
                    mode='lines+markers',
                    name='Measurement residual',
                    legendgroup='residuals',                    # ← группа "residuals"
                    legendgrouptitle=dict(text='Residuals'),    # ← заголовок группы
                    line=dict(color='blue'),
                ),
                row=current_row, col=1
            )
        if r_cont_hist is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(r_cont_hist)),
                    y=r_cont_hist,
                    mode='lines+markers',
                    name='Continuity residual',
                    legendgroup='residuals',
                    line=dict(color='red'),
                ),
                row=current_row, col=1
            )
            fig.update_xaxes(title_text='Iteration', row=current_row, col=1)
            fig.update_yaxes(title_text='Residual norm (log)', type='log', row=current_row, col=1)
            current_row += 1

    # Настройка осей для верхней панели (траектории) с использованием state_names
    if plot_trajectory:
        if is_single:
            fig.update_xaxes(title_text='time, s', row=1, col=1)
            fig.update_yaxes(title_text=state_names[0], row=1, col=1)
        else:
            if use_3d:
                fig.update_scenes(
                    xaxis_title=state_names[0],
                    yaxis_title=state_names[1],
                    zaxis_title=state_names[2],
                )
            else:
                fig.update_xaxes(title_text=state_names[0], row=1, col=1)
                fig.update_yaxes(title_text=state_names[1], row=1, col=1)

    # Если plot_xy, подписываем оси
    if plot_xy and not is_single:
        base_row = 2 if plot_trajectory else 1  # plot_xy начинаются после траектории
        for obs_idx in range(n_obs):
            for batch_idx in range(n_batches):
                row = base_row + obs_idx
                col = batch_idx + 1
                fig.update_xaxes(title_text='time, s' if obs_idx == n_obs-1 else '', row=row, col=col)
                fig.update_yaxes(title_text=f'{state_names[obs_idx]} (obs)', row=row, col=col)
    fig.update_annotations(font=dict(size=fontsize))
    return fig