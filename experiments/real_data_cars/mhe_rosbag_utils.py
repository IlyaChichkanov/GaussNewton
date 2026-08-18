"""Прогон MHE-моделей (acados) на сигналах реальных бэгов (ноутбук mhe_test_rosbag.ipynb).

Механика поверх tests_utils/mhe_utils.py: равномерная сетка -> vx-маска ->
непрерывные сегменты -> окна -> общий шаг окна `solve_mhe_window`. Семантика
зеркалит рантайм C++ (MhePipeline::update): между сегментами и после неудачного
solve буфер «чистится» (re-warm-start солвера), Σ arrival cost и прайор
параметров живут. Гейтов рантайма (info_gain_ref / clamp / floor / rho) здесь
нет — это «чистый» MHE. LS-референсы живут рядом с данными:
scripts/control_scripts/bag_to_csv/check_dynamic_mhe_inputs.py.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from acados_template import AcadosOcpSolver
from mhe.mhe_base_model_interface import MheCogeGenerator, MheModel
from plotly.subplots import make_subplots
from tests_utils.mhe_utils import (
    ArrivalCostUpdater,
    MheIterationResult,
    make_obs_map,
    reset_mhe_solver,
    solve_mhe_window,
)
from tqdm import tqdm

# палитра (валидированные категориальные слоты; красный — статус фейла)
C_BLUE, C_ORANGE, C_AQUA, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
_MUTED_GRAY = "#52514e"


# ---------------------------------------------------------------- сетка/сегменты

def build_grid(t1: float, t2: float, dt: float) -> np.ndarray:
    """Равномерная сетка с шагом dt строго внутри [t1, t2]."""
    n = int(np.floor((t2 - t1) / dt)) + 1
    return t1 + np.arange(n) * dt


def find_segments(mask: np.ndarray, min_len: int) -> list:
    """Непрерывные куски True длиной >= min_len -> [(start, stop), ...] (полуинтервалы)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


@dataclass
class SegmentWindows:
    seg_id: int
    start: int                 # начало сегмента в сетке
    stop: int                  # конец сегмента (полуинтервал)
    window_starts: list        # глобальные индексы начал окон (окно = [i, i+N))


def make_window_starts(segments: list, n_horizon: int, shift: int) -> list:
    out = []
    for k, (s0, s1) in enumerate(segments):
        starts = list(range(s0, s1 - n_horizon + 1, shift))
        out.append(SegmentWindows(k, s0, s1, starts))
    return out


def gap_guard_mask(t_grid: np.ndarray, raw_times: list, max_gap_s: float) -> np.ndarray:
    """True там, где точка сетки попала в разрыв сырых данных любого источника."""
    gap = np.zeros(len(t_grid), dtype=bool)
    for t_src in raw_times:
        idx = np.searchsorted(t_src, t_grid).clip(1, len(t_src) - 1)
        gap |= (t_src[idx] - t_src[idx - 1]) > max_gap_s
    return gap


# ---------------------------------------------------------------- солвер + кэш

def get_or_build_solver(system, mhe_params, gen_dir, model_name: str,
                        max_sqp_iter: int = 150, force_rebuild: bool = False):
    gen_dir = Path(gen_dir).resolve()
    sub = gen_dir / f"{model_name}_N{mhe_params.mhe_horizont}_it{max_sqp_iter}"

    class _Generator(MheCogeGenerator):
        def modify_ocp_problem(self, ocp_mhe):
            ocp_mhe.solver_options.print_level = 0
            ocp_mhe.solver_options.nlp_solver_stats_level = 0
            ocp_mhe.solver_options.nlp_solver_max_iter = max_sqp_iter
            return ocp_mhe

    generator = _Generator(system, mhe_params, sub, model_name)
    json_file = sub / "mhe_config.json"
    lib_file = sub / f"libacados_ocp_solver_{model_name}.so"
    solver = None
    if not force_rebuild and json_file.exists() and lib_file.exists():
        gc.collect()  # добить прошлые солверы: dlopen по тому же пути вернул бы старый handle
        try:
            solver = AcadosOcpSolver(None, json_file=str(json_file), build=False,
                                     generate=False)
            print(f"[{model_name}] солвер загружен из кэша: {sub}")
        except (KeyError, TypeError, ValueError) as exc:
            # Кэш собран другой версией acados: в mhe_config.json нет полей,
            # которые ждёт текущий acados_template (типичное — KeyError
            # 'code_gen_opts'). Это не ошибка пользователя — просто пересобираем.
            print(f"[{model_name}] кэш несовместим с текущим acados "
                  f"({type(exc).__name__}: {exc}) — пересобираю")

    if solver is None:
        solver = generator.generate_code()
        print(f"[{model_name}] собран: {sub}")
    return generator.get_model(), solver


# ---------------------------------------------------------------- раннер

@dataclass
class WindowResult:
    seg_id: int
    i_start: int               # глобальный индекс начала окна в сетке
    t_end: float               # время последнего сэмпла окна
    status: int                # статус acados (0 ok)
    res: Optional[MheIterationResult]   # None, если окно не решилось


def run_mhe_on_segments(
    mhe_model: MheModel,
    solver: AcadosOcpSolver,
    mhe_params,
    t_grid: np.ndarray,
    u: np.ndarray,             # (Ng, nu)
    y: np.ndarray,             # (Ng, ny)
    windows: list,             # list[SegmentWindows]
    x0_func: Callable,         # (y0_row, u0_row, theta) -> state (nx,)
    theta0: np.ndarray,
    sigma0: np.ndarray,        # (n_aug, n_aug)
    shift: int,
    q_state,                   # скаляр или вектор (nx,) — как state_aug_noise в C++
    q_param,                   # скаляр или вектор (np,)
    r_inv,
    sigma_max_std=None,        # (n_aug,) std; включает C++ clamp/floor Σ в updater
    sigma_min_ratio: float = 0.0,
    reset_prior_between_segments: bool = False,
    accept_max_iter: bool = False,
    max_windows: Optional[int] = None,
    progress: bool = True,
) -> list:
    """Цикл окон по сегментам через общий `solve_mhe_window` с зеркалом машины:
    между сегментами и после неудачного solve — re-warm-start (аналог clear()),
    Σ updater'а и theta_prior живут."""
    N = mhe_params.mhe_horizont
    L = shift
    nx = mhe_model.state_length
    dt = mhe_params.dt
    assert u.ndim == 2 and y.ndim == 2, "U/Y обязаны быть 2D (set_mhe_solver индексирует [j,:])"

    updater = ArrivalCostUpdater(mhe_model, dt, r_inv,
                                 q_state_diag=q_state, q_param_diag=q_param,
                                 initial_sigma=np.array(sigma0, dtype=float),
                                 sigma_max_std=sigma_max_std,
                                 sigma_min_ratio=sigma_min_ratio)
    obs_map = make_obs_map(mhe_model, N)
    theta_prior = np.asarray(theta0, dtype=float).copy()
    ok_statuses = (0, 2) if accept_max_iter else (0,)

    total = sum(len(sw.window_starts) for sw in windows)
    if max_windows is not None:
        total = min(total, max_windows)
    pbar = tqdm(total=total, desc="MHE окна", unit="окно") if progress else None

    results = []
    n_done = 0
    try:
        for sw in windows:
            if reset_prior_between_segments:
                updater.Sigma = np.array(sigma0, dtype=float)
                theta_prior = np.asarray(theta0, dtype=float).copy()
            need_warm_start = True     # начало сегмента = clear() на машине
            prev_x_est = None
            for i0 in sw.window_starts:
                if max_windows is not None and n_done >= max_windows:
                    return results
                u_seq = u[i0:i0 + N]
                y_seq = y[i0:i0 + N]
                t_batch = t_grid[i0:i0 + N]

                if need_warm_start:
                    x_prior = np.asarray(x0_func(y_seq[0], u_seq[0], theta_prior), dtype=float)
                    reset_mhe_solver(mhe_model, solver, u_seq, x_prior, theta_prior, N, dt)
                    need_warm_start = False
                else:
                    x_prior = prev_x_est[L]

                status, est, obs_pred = solve_mhe_window(
                    mhe_model, solver, updater, y_seq, u_seq, x_prior, theta_prior,
                    N, L, obs_map=obs_map, ok_statuses=ok_statuses)
                if est is None:
                    results.append(WindowResult(sw.seg_id, i0, t_batch[-1], status, None))
                    need_warm_start = True    # зеркало clear() на SOLVED_ERROR
                    prev_x_est = None
                    n_done += 1
                    if pbar:
                        pbar.update(1)
                    continue

                theta_prior = est.sim_param_est
                prev_x_est = est.sim_x_est.copy()

                res = MheIterationResult(
                    t_batch=t_batch,
                    state_sequence=y_seq,
                    control_sequence=u_seq,
                    state_est=est.sim_x_est,
                    noise_est=est.sim_w_est,
                    param_est=est.sim_param_est,
                    cov_matrix=updater.Sigma[nx:, nx:].flatten(),
                    status=status,
                    cost_value=est.cost_value,
                    sqp_iter=est.sqp_iter,
                    obs_est=obs_pred,
                )
                results.append(WindowResult(sw.seg_id, i0, t_batch[-1], status, res))
                n_done += 1
                if pbar:
                    pbar.update(1)
    finally:
        if pbar:
            pbar.close()
    return results


def summarize_results(name: str, results: list, theta_names: list) -> None:
    solved = [w for w in results if w.res is not None]
    statuses = {}
    for w in results:
        statuses[w.status] = statuses.get(w.status, 0) + 1
    print(f"{name}: решено {len(solved)}/{len(results)} окон, статусы {statuses}")
    if solved:
        n_theta = len(theta_names)
        th = solved[-1].res.param_est
        std = np.sqrt(np.maximum(np.diag(solved[-1].res.cov_matrix.reshape(n_theta, n_theta)), 0))
        for i, nm in enumerate(theta_names):
            print(f"  {nm}: {th[i]:.5g} ± {std[i]:.3g}")


# ---------------------------------------------------------------- оценки σ

def _n_corr_of(e, cap):
    """1 + 2*Σρ_k по автокорреляции ряда до первого ρ<0.05 (кэп cap лагов)."""
    e = e - e.mean()
    v = float(e @ e)
    if v <= 0:
        return 1.0
    s = 0.0
    for k in range(1, min(cap, len(e) - 1)):
        rk = float(e[:-k] @ e[k:]) / v
        if rk < 0.05:
            break
        s += rk
    return 1.0 + 2.0 * s


def sigma_bands(results, r_inv, sigma0_std, rho_bounds=(1e-2, 10.0),
                sigma_min_ratio=1e-3, ema_alpha=0.1, scatter_window=30,
                corr_lag_cap=100):
    """Три оценки σ параметров поверх сырой Σ_θθ (по решённым окнам, по порядку).

    - sigma_machine: ρ̂_EMA (зажим rho_bounds) · clip(σ_raw, sigma_min_ratio·σ0, σ0)
      — воспроизводит C++: clamp/floor режут саму Σ (в update()), и только ПОТОМ
      диагностика умножает на ρ̂ (update_diagnostics), поэтому на полу машина
      публикует ρ̂·ratio·σ0, а не ratio·σ0. Покомпонентный clip — приближение
      спектрального clamp; если updater гонялся с sigma_max_std (machine-режим),
      σ_raw уже обрезана и clip здесь no-op. Сопоставимо со std_devs топика mhe3.
      ρ̂ — апостериорный масштаб единицы веса: ρ̂² = Σ rᵀWr / dof,
      dof = N·n_obs − n_aug;
    - sigma_eff: ρ̂_raw · σ_raw · √n_corr — честный локальный CI: HAC-поправка
      на автокорреляцию невязки (dof считает измерения независимыми, реально
      независимых N/n_corr);
    - scatter_std/mean: скользящий разброс самой оценки — маршрутный интервал
      (систематику не покрывает никакая ковариация).

    rho_bounds=(1,1) эквивалентно «bounds не заданы» в C++ (ρ̂ ≡ 1, delay).
    Возвращает dict массивов длины n_solved (σ — формы (n, n_theta)).
    """
    solved = [w for w in results if w.res is not None]
    if not solved:
        return {}
    n_theta = len(np.asarray(solved[0].res.param_est).ravel())
    nx = np.asarray(solved[0].res.state_est).shape[1]
    r_inv = np.asarray(r_inv, dtype=float)
    W = np.diag(r_inv) if r_inv.ndim == 1 else r_inv

    t = np.array([w.t_end for w in solved])
    P = np.array([w.res.param_est for w in solved])
    S_raw = np.array([np.sqrt(np.maximum(
        np.diag(w.res.cov_matrix.reshape(n_theta, n_theta)), 0)) for w in solved])

    rho_raw = np.empty(len(solved))
    n_corr = np.empty(len(solved))
    for j, w in enumerate(solved):
        r = np.asarray(w.res.obs_est) - np.asarray(w.res.state_sequence)
        dof = max(1, r.size - (nx + n_theta))
        rho_raw[j] = np.sqrt(float(np.einsum("ki,ij,kj->", r, W, r)) / dof)
        n_corr[j] = max(_n_corr_of(r[:, i] * np.sqrt(W[i, i]), corr_lag_cap)
                        for i in range(r.shape[1]))

    # EMA по ρ̂² (как C++), затем зажим
    rho2 = rho_raw ** 2
    ema = np.empty_like(rho2)
    ema[0] = rho2[0]
    for j in range(1, len(rho2)):
        ema[j] = (1 - ema_alpha) * ema[j - 1] + ema_alpha * rho2[j]
    rho_machine = np.clip(np.sqrt(ema), rho_bounds[0], rho_bounds[1])

    s0 = np.asarray(sigma0_std, dtype=float)[-n_theta:]
    # порядок как в C++: сначала clamp/floor по Σ, потом умножение на rho
    sigma_machine = rho_machine[:, None] * np.clip(S_raw, sigma_min_ratio * s0, s0)
    sigma_eff = rho_raw[:, None] * S_raw * np.sqrt(n_corr)[:, None]

    scatter_mean = np.full_like(P, np.nan)
    scatter_std = np.full_like(P, np.nan)
    wlen = scatter_window
    for j in range(len(P)):
        a = max(0, j - wlen + 1)
        if j - a + 1 >= max(5, wlen // 3):
            scatter_mean[j] = P[a:j + 1].mean(axis=0)
            scatter_std[j] = P[a:j + 1].std(axis=0, ddof=1)

    return {"t": t, "sigma_raw": S_raw, "sigma_machine": sigma_machine,
            "sigma_eff": sigma_eff, "scatter_mean": scatter_mean,
            "scatter_std": scatter_std, "rho_raw": rho_raw,
            "rho_machine": rho_machine, "n_corr": n_corr}


# ---------------------------------------------------------------- графики

def _hex_rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def plot_param_tracks(results, param_names, ls_ref=None, bounds=None, theta0=None,
                      segments_time=None, title=None, height_per_param=320,
                      bands=None):
    """Треки параметров по всем окнам (plotly): линия + лента ±σ_raw (Σ_θθ
    arrival cost), заливка сегментов, красные vlines — нерешённые окна,
    пунктиры — LS-референс / θ0 / границы yaml. `bands` (из `sigma_bands`)
    добавляет: жёлтую ленту «σ как на машине» (ρ̂+clamp/floor, сверка с mhe3),
    пунктир ±σ_eff (HAC) и точечный скользящий разброс оценки."""
    solved = [w for w in results if w.res is not None]
    failed = [w for w in results if w.res is None]
    n_theta = len(param_names)
    fig = make_subplots(rows=n_theta, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, subplot_titles=param_names)
    if solved:
        t = np.array([w.t_end for w in solved])
        P = np.array([w.res.param_est for w in solved])
        S = np.array([np.sqrt(np.maximum(
            np.diag(w.res.cov_matrix.reshape(n_theta, n_theta)), 0)) for w in solved])
    if bands:
        assert len(bands["t"]) == len(solved), "bands из другого results"
    for i in range(n_theta):
        row = i + 1
        first = i == 0
        if solved:
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([P[:, i] - S[:, i], (P[:, i] + S[:, i])[::-1]]),
                fill="toself", fillcolor=_hex_rgba(C_BLUE, 0.18),
                line=dict(width=0), name="±σ raw", legendgroup="band",
                showlegend=first, hoverinfo="skip"), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=P[:, i], mode="lines", name="оценка MHE", legendgroup="est",
                showlegend=first, line=dict(color=C_BLUE, width=2)), row=row, col=1)
        if bands:
            sm = bands["sigma_machine"][:, i]
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([P[:, i] - sm, (P[:, i] + sm)[::-1]]),
                fill="toself", fillcolor=_hex_rgba(C_YELLOW, 0.30),
                line=dict(width=0), name="±σ машины (≈mhe3)",
                legendgroup="machine", showlegend=first, hoverinfo="skip"),
                row=row, col=1)
            se = bands["sigma_eff"][:, i]
            for sign, show in ((1, first), (-1, False)):
                fig.add_trace(go.Scatter(
                    x=t, y=P[:, i] + sign * se, mode="lines",
                    name="±σ_eff (HAC)", legendgroup="eff",
                    showlegend=show, hoverinfo="skip",
                    line=dict(color=C_MAGENTA, width=1.2, dash="dash")),
                    row=row, col=1)
            mu, sc = bands["scatter_mean"][:, i], bands["scatter_std"][:, i]
            for sign, show in ((1, first), (-1, False)):
                fig.add_trace(go.Scatter(
                    x=t, y=mu + sign * sc, mode="lines",
                    name="скользящий разброс", legendgroup="scatter",
                    showlegend=show, hoverinfo="skip",
                    line=dict(color=_MUTED_GRAY, width=1, dash="dot")),
                    row=row, col=1)
        if theta0 is not None:
            fig.add_hline(y=theta0[i], line=dict(color=_MUTED_GRAY, width=1, dash="dot"),
                          row=row, col=1)
        if ls_ref is not None:
            fig.add_hline(y=ls_ref[i], line=dict(color=C_ORANGE, width=1.5, dash="dash"),
                          row=row, col=1)
        if bounds is not None:
            for b in (bounds[0][i], bounds[1][i]):
                fig.add_hline(y=b, line=dict(color="#0b0b0b", width=1, dash="dot"),
                              opacity=0.4, row=row, col=1)
    if segments_time:
        for a, b in segments_time:
            fig.add_vrect(x0=a, x1=b, fillcolor=_hex_rgba(C_AQUA, 0.08),
                          line_width=0, layer="below", row="all", col=1)
    for w in failed[:200]:
        fig.add_vline(x=w.t_end, line=dict(color=C_RED, width=1), opacity=0.35,
                      row="all", col=1)
    if ls_ref is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="LS-референс",
                                 line=dict(color=C_ORANGE, width=1.5, dash="dash")))
    if theta0 is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="θ0",
                                 line=dict(color=_MUTED_GRAY, width=1, dash="dot")))
    fig.update_layout(
        template="plotly_white", title=title,
        height=max(320, height_per_param * n_theta + 90), width=1200,
        font=dict(size=12, color="#0b0b0b"),
        # легенда вертикально справа: горизонтальная наезжала на заголовки
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top"),
        margin=dict(r=200),
        hovermode="x unified")
    fig.update_xaxes(title_text="t [с]", row=n_theta, col=1)
    return fig
