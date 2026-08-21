"""Plotly visualisation of an identification run.

plot_solution takes a problem and the `hist` dict returned by
run_optimization_adaptive; see docs/api-reference.md.
"""
import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import to_rgb

if TYPE_CHECKING:
    from gauss_newton.problem import MultipleShooting

PARAM_COLORS = ['blue', 'orange', 'green', 'red', 'purple',
                'brown', 'pink', 'gray', 'olive', 'cyan']


def _resolve_names(state_names, param_names, n_state, n_theta):
    if state_names is None:
        state_names = [f'state{i}' for i in range(n_state)]
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(n_theta)]
    elif len(param_names) != n_theta:
        raise ValueError(
            f"param_names must have length {n_theta}, got {len(param_names)}")
    return state_names, param_names


def _make_figure(n_batches, n_obs, use_3d, is_single, fontsize,
                 plot_trajectory, plot_xy, plot_theta, plot_residuals):
    """Lay out the subplot grid: one row per panel, n_batches columns for xy."""
    rows = []                      # plotly subplot type per row
    if plot_trajectory:
        rows.append('scene' if use_3d else 'xy')
    if plot_xy and not is_single:
        rows.extend(['xy'] * n_obs)
    if plot_theta:
        rows.append('xy')
    if plot_residuals:
        rows.append('xy')
    if not rows:
        raise ValueError("no panels selected for plotting")

    # The trajectory panel is three times as tall as the rest
    row_heights = ([3] if plot_trajectory else []) + [1] * (len(rows) - (1 if plot_trajectory else 0))

    # Only the xy rows use several columns; the others span the whole width
    max_cols = n_batches if (plot_xy and not is_single) else 1
    wide = [{'type': 'xy', 'colspan': max_cols}] + [None] * (max_cols - 1)
    specs = []
    for i, rtype in enumerate(rows):
        is_xy_row = plot_xy and not is_single and (
            (1 if plot_trajectory else 0) <= i < (1 if plot_trajectory else 0) + n_obs)
        if is_xy_row:
            specs.append([{'type': rtype}] * max_cols)
        else:
            specs.append([{'type': rtype, 'colspan': max_cols}] + [None] * (max_cols - 1)
                         if rtype != 'xy' else list(wide))

    fig = make_subplots(
        rows=len(rows), cols=max_cols, specs=specs, row_heights=row_heights,
        horizontal_spacing=0.05, vertical_spacing=0.08,
    )
    fig.update_layout(
        width=1400, height=1000,
        font=dict(size=fontsize),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def _add_phase_trace(fig, row, col, states, name=None, color='blue', is_3d=False):
    """Phase trajectory; a single-component state is drawn as a time series."""
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


def _add_measurements(fig, row, batch_idx, state_measured, t_meas, n_obs,
                      use_3d, is_single, plot_trajectory, plot_xy):
    if plot_trajectory:
        if is_single:
            fig.add_trace(
                go.Scatter(
                    x=t_meas, y=state_measured[:, 0],
                    mode='markers', marker=dict(color='green', symbol='x', size=8),
                    name='Measurements' if batch_idx == 0 else None,
                    showlegend=(batch_idx == 0),
                ),
                row=row, col=1
            )
        elif use_3d and state_measured.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=state_measured[:, 0], y=state_measured[:, 1],
                    z=state_measured[:, 2],
                    mode='markers', marker=dict(color='green', symbol='x', size=5),
                    name='Meas',
                ),
                row=row, col=1
            )
        elif state_measured.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=state_measured[:, 0], y=state_measured[:, 1],
                    mode='markers', marker=dict(color='green', symbol='x', size=8),
                    name='Measurements' if batch_idx == 0 else None,
                    showlegend=(batch_idx == 0),
                ),
                row=row, col=1
            )
    if plot_xy and not is_single:
        base_row = row + (1 if plot_trajectory else 0)
        for obs_idx in range(n_obs):
            fig.add_trace(
                go.Scatter(
                    x=t_meas, y=state_measured[:, obs_idx],
                    mode='markers', marker=dict(color='green', symbol='x', size=6),
                    name='meas',
                    showlegend=False,
                ),
                row=base_row + obs_idx, col=batch_idx + 1
            )


def _add_shot_traces(fig, row, problem, theta, theta_full, batch_idx, n_state,
                     n_theta, n_obs, use_3d, is_single, plot_trajectory, plot_xy):
    """Densely resampled trajectory of every shot of one batch."""
    def trajectory(c0, t_eval):
        if problem.use_jax:
            sol = problem.integrator.get_solution_jax(c0, theta, t_eval)
        else:
            sol = problem.integrator.get_solution(c0, theta, t_eval)
        return sol.T                                    # (n_time, n_state)

    tm = problem.interval_managers[batch_idx]
    c_offset = n_theta + n_state * sum(
        im.N_shoot for im in problem.interval_managers[:batch_idx])

    for shoot in range(tm.N_shoot):
        start_idx = c_offset + shoot * n_state
        c0 = theta_full[start_idx:start_idx + n_state]

        t_interval, _ = tm.get_time_interval(shoot)
        t_dense = np.linspace(t_interval[0], t_interval[-1], 500)
        traj = trajectory(c0, t_dense)                  # (500, n_state)
        meas_traj = np.array([problem.system.h(state, t_, theta)
                              for t_, state in zip(t_dense, traj)])

        if plot_trajectory:
            if is_single:
                fig.add_trace(
                    go.Scatter(
                        x=t_dense, y=meas_traj[:, 0],
                        mode='lines', line=dict(color='blue', width=2),
                        name=f'batch {batch_idx}' if shoot == 0 else None,
                        showlegend=(shoot == 0),
                    ),
                    row=row, col=1
                )
            else:
                _add_phase_trace(
                    fig, row, 1, traj,
                    name=f'batch {batch_idx}' if shoot == 0 else None,
                    color='blue', is_3d=use_3d)

        if plot_xy and not is_single:
            base_row = row + (1 if plot_trajectory else 0)
            for obs_idx in range(n_obs):
                fig.add_trace(
                    go.Scatter(
                        x=t_dense, y=meas_traj[:, obs_idx],
                        mode='lines', line=dict(color='blue', width=1.5),
                        name=f'batch {batch_idx}' if shoot == 0 else None,
                        showlegend=False,
                    ),
                    row=base_row + obs_idx, col=batch_idx + 1
                )


def _select_parameters(theta_hist, index, n_theta, plot_param_indices,
                       param_names, theta_true, ci_low_hist, ci_high_hist):
    """Parameter history, labels and intervals, restricted to the chosen indices."""
    history = np.array(theta_hist)[:index + 1, :n_theta]
    if plot_param_indices is None:
        return (history, param_names, theta_true,
                None if ci_low_hist is None else ci_low_hist[:index + 1, :],
                None if ci_high_hist is None else ci_high_hist[:index + 1, :])

    indices = list(plot_param_indices)
    if max(indices) >= n_theta or min(indices) < 0:
        raise IndexError(f"parameter indices must lie in 0..{n_theta - 1}")
    return (history[:, indices],
            [param_names[i] for i in indices],
            theta_true[indices] if theta_true is not None else None,
            None if ci_low_hist is None else ci_low_hist[:index + 1, indices],
            None if ci_high_hist is None else ci_high_hist[:index + 1, indices])


def _add_theta_panel(fig, row, history, labels, theta_true, ci_low, ci_high):
    """Parameter convergence with a shaded confidence band."""
    for i in range(history.shape[1]):
        color = PARAM_COLORS[i % len(PARAM_COLORS)]
        if ci_low is not None and ci_high is not None:
            # Invisible upper bound, then the lower bound filled up to it
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(ci_low)), y=ci_high[:, i],
                    mode='lines', line=dict(width=0), showlegend=False,
                ),
                row=row, col=1
            )
            r, g, b = to_rgb(color)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(ci_low)), y=ci_low[:, i],
                    mode='lines', fill='tonexty',
                    fillcolor=f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.2)',
                    line=dict(width=0), showlegend=False,
                ),
                row=row, col=1
            )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(history)), y=history[:, i],
                mode='lines+markers',
                line=dict(color=color, shape='hv'),
                name=labels[i],
                legendgroup='parameters',
                legendgrouptitle=dict(text='Parameters'),
            ),
            row=row, col=1
        )
        if theta_true is not None and i < len(theta_true):
            fig.add_hline(
                y=theta_true[i], line_dash='dash', line_color=color,
                opacity=0.7, row=row, col=1,
                # The panel is addressed explicitly and has data, so the empty
                # subplot filter is unnecessary — and it breaks here: it reads
                # trace.xaxis of every trace, which a Scatter3d does not have
                exclude_empty_subplots=False,
            )
    fig.update_xaxes(title_text='Iteration', row=row, col=1)
    fig.update_yaxes(title_text='Parameter value', row=row, col=1)


def _add_residual_panel(fig, row, r_meas_hist, r_cont_hist):
    if r_meas_hist is not None:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r_meas_hist)), y=r_meas_hist,
                mode='lines+markers', name='Measurement residual',
                legendgroup='residuals',
                legendgrouptitle=dict(text='Residuals'),
                line=dict(color='blue'),
            ),
            row=row, col=1
        )
    if r_cont_hist is not None:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r_cont_hist)), y=r_cont_hist,
                mode='lines+markers', name='Continuity residual',
                legendgroup='residuals', line=dict(color='red'),
            ),
            row=row, col=1
        )
        fig.update_xaxes(title_text='Iteration', row=row, col=1)
        fig.update_yaxes(title_text='Residual norm (log)', type='log',
                         row=row, col=1)


def _label_axes(fig, state_names, n_obs, n_batches, use_3d, is_single,
                plot_trajectory, plot_xy):
    if plot_trajectory:
        if is_single:
            fig.update_xaxes(title_text='time, s', row=1, col=1)
            fig.update_yaxes(title_text=state_names[0], row=1, col=1)
        elif use_3d:
            fig.update_scenes(xaxis_title=state_names[0],
                              yaxis_title=state_names[1],
                              zaxis_title=state_names[2])
        else:
            fig.update_xaxes(title_text=state_names[0], row=1, col=1)
            fig.update_yaxes(title_text=state_names[1], row=1, col=1)

    if plot_xy and not is_single:
        base_row = 2 if plot_trajectory else 1
        for obs_idx in range(n_obs):
            for batch_idx in range(n_batches):
                fig.update_xaxes(
                    title_text='time, s' if obs_idx == n_obs - 1 else '',
                    row=base_row + obs_idx, col=batch_idx + 1)
                fig.update_yaxes(title_text=f'{state_names[obs_idx]} (obs)',
                                 row=base_row + obs_idx, col=batch_idx + 1)


def plot_solution(
    fig: Optional[go.Figure] = None,
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
    param_names: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
    fontsize=10
) -> go.Figure:
    """Figure of an identification run: trajectories, parameters, residuals.

    The panels are selected by the plot_* flags; `fig` is accepted for
    backwards compatibility and ignored — a new figure is always built.
    """
    if problem is None or theta_hist is None:
        raise ValueError("problem and theta_hist must be provided")

    n_batches = len(problem.t_eval_measurements_batches)
    n_state, n_theta, n_obs = problem.system.dims()
    state_names, param_names = _resolve_names(state_names, param_names,
                                              n_state, n_theta)

    is_single = n_state == 1
    if is_single and plot_xy:
        warnings.warn("plot_xy is ignored when n_state == 1 "
                      "(state vs time is already shown)")
        plot_xy = False
    use_3d = (not is_single) and (n_state >= 3)

    if index < 0:
        index = len(theta_hist) - 1
    if index >= len(theta_hist):
        raise IndexError(f"index {index} out of range for theta_hist")
    theta_full = theta_hist[index]
    theta = theta_full[:n_theta]

    fig = _make_figure(n_batches, n_obs, use_3d, is_single, fontsize,
                       plot_trajectory, plot_xy, plot_theta, plot_residuals)

    current_row = 1                                    # plotly rows start at 1
    if plot_true_solution and plot_trajectory and not is_single:
        if not hasattr(problem, 'full_trajectory') or problem.full_trajectory is None:
            raise AttributeError("problem.full_trajectory is None")
        _, state_true_batches = problem.full_trajectory
        for state_true in state_true_batches:
            _add_phase_trace(fig, current_row, 1, state_true, name='True',
                             color='grey', is_3d=use_3d)

    for batch_idx, (state_measured, t_meas) in enumerate(
            zip(problem.state_measured_batches,
                problem.t_eval_measurements_batches)):
        if plot_measurements:
            _add_measurements(fig, current_row, batch_idx, state_measured,
                              t_meas, n_obs, use_3d, is_single,
                              plot_trajectory, plot_xy)
        _add_shot_traces(fig, current_row, problem, theta, theta_full,
                         batch_idx, n_state, n_theta, n_obs, use_3d, is_single,
                         plot_trajectory, plot_xy)

    current_row += ((1 if plot_trajectory else 0)
                    + (n_obs if (plot_xy and not is_single) else 0))

    if plot_theta:
        history, labels, true_sel, ci_low, ci_high = _select_parameters(
            theta_hist, index, n_theta, plot_param_indices, param_names,
            theta_true, ci_low_hist, ci_high_hist)
        _add_theta_panel(fig, current_row, history, labels, true_sel,
                         ci_low, ci_high)
        current_row += 1

    if plot_residuals:
        _add_residual_panel(fig, current_row, r_meas_hist, r_cont_hist)

    _label_axes(fig, state_names, n_obs, n_batches, use_3d, is_single,
                plot_trajectory, plot_xy)
    fig.update_annotations(font=dict(size=fontsize))
    return fig
