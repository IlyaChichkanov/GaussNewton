import scipy.linalg as la
import numpy as np
from mhe.mhe_base_model_interface import MheModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


from dataclasses import dataclass


import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MheIterationResult:
    """Store results from one MHE iteration."""
    t_batch: np.ndarray
    simY: np.ndarray
    simU: np.ndarray
    state_est: np.ndarray
    noise_est: np.ndarray
    param_est: np.ndarray
    cov_matrix: np.ndarray
    fim: np.ndarray
    eigvals: np.ndarray
    status: int
    cost_value: float
    sqp_iter: int


def reset_mhe_solver(mhe_model: MheModel, 
               acados_solver_mhe: AcadosOcpSolver, 
               simU: np.array, 
               initial_x0: np.array, 
               initial_theta: np.array, 
               N: int) -> tuple :
    assert(len(initial_x0) == mhe_model.state_length)
    assert(len(initial_theta) == mhe_model.param_length)
    assert simU.shape[0] >= N, f"simU должен содержать хотя бы {N} строк"

    x_sim = initial_x0.copy()

    integrate_f = mhe_model.create_intefrate_function(0.02, "integrate")
    for j in range(N):
        # Формируем расширенный вектор состояния + параметров
        x_aug = np.hstack((x_sim, initial_theta))
        acados_solver_mhe.set(j, "x", x_aug)

        # Делаем шаг вперёд по дискретной динамике
        if j < N - 1:
            x_sim = np.array(integrate_f(x_sim, initial_theta, simU[j, :])).T[0]




def run_mhe_estimation(
    mhe_model,
    acados_solver_factory,
    get_window_func,
    get_initial_state_func,
    overlap_points: int,
    initial_theta: np.ndarray,
    mhe_params,
    num_windows: int,
    R_inv: np.ndarray,
    ridge_reg: float  = 1.0,
    forgetting_factor: float = 0.95,   # λ
    initial_precision: np.ndarray = None,
    compute_advanced_fim = True,
    plot: bool = True,
    progress_bar: bool = True,
) -> Tuple[List[MheIterationResult], np.ndarray]:
    N_measurement = mhe_params.mhe_horizont
    n_theta = mhe_model.param_length
    n_obs = mhe_model.obs_length
    nx = mhe_model.state_length
    results = []
    # Инициализация
    if initial_precision is None:
        P_inv = 1e-0 * np.eye(n_theta)
    else:
        P_inv = initial_precision
    theta_prior = initial_theta

    iterator = range(num_windows)
    if progress_bar:
        iterator = tqdm(iterator, desc="MHE windows", unit="window")

    for iter_idx in iterator:
        t_batch, simU, simY, _ = get_window_func(iter_idx)
        unknown_state_length = 0
        simY = np.hstack((simY, np.zeros((simY.shape[0], unknown_state_length))))

        if iter_idx == 0:
            initial_x0 = get_initial_state_func(simY[0], simU[0], initial_theta) 

        # Вычисляем FIM для текущего окна
        F_orig = None
        if(compute_advanced_fim):
            F_orig = mhe_model.compute_fim(simU.shape[0], mhe_params.dt, simU, initial_x0, theta_prior, R_inv)
        else:
            F_orig = mhe_model.compute_observed_fim(simU.shape[0], mhe_params.dt, simU, simY, initial_x0, theta_prior, R_inv)

        # Обновляем накопленную точность (информационную матрицу)
        P_inv = forgetting_factor * P_inv + F_orig
        F_reg, eig_orig, eig_reg = regularize_fim(P_inv, tau_ratio=1e-4, add_ridge=ridge_reg)
        # if(len(F_reg) == 1):
        #     F_reg = np.array([[1.0]])
        # Настраиваем MHE с априорными параметрами и точностью

        set_mhe_solver(
            mhe_model, acados_solver_factory, simY, simU, initial_x0, theta_prior,
            N_measurement, F_reg   # P0 = накопленная точность
        )

        status = acados_solver_factory.solve()
        if status != 0:
            logger.warning(f"Window {iter_idx}: acados returned status {status}. Skipping this window.")
            # При неудаче можно оставить старые параметры (theta_prior не менять)
            # Но всё равно нужно сдвинуть окно? Пропускаем результат.
            # Для простоты пропустим добавление в results и продолжим.
            continue

        mhe_output = get_mhe_estimated_data(mhe_model, acados_solver_factory, N_measurement)
        simXest = mhe_output.simXest
        simWest = mhe_output.simWest
        theta_new = mhe_output.sim_param_est

        # Обновляем априорные параметры для следующего окна
        theta_prior = theta_new

        # Сдвиг окна (для следующего шага)
        initial_x0 = simXest[N_measurement - overlap_points]

        # Сохраняем результаты
        result = MheIterationResult(
            t_batch=t_batch,
            simY=simY,
            simU=simU,
            state_est=simXest,
            noise_est=simWest,
            param_est=theta_prior,
            cov_matrix=np.linalg.inv(F_reg).flatten(),   # ковариация для отображения
            fim=F_reg,
            eigvals=eig_orig,
            status=status,
            cost_value=mhe_output.cost_value,
            sqp_iter=mhe_output.sqp_iter
        )
        results.append(result)

        if plot:
            plt.plot(t_batch, simY, 'g', label='measured')
            plt.plot(t_batch, simXest[:-1], 'b', label='estimated')
            plt.title(f"Window {iter_idx}")
            plt.legend()
            plt.show()

    return results

def plot_mhe_results(results, overlap=0, initial_params=None, theta_true=None,
                     plot_states=True, plot_params=True,
                     plot_eigvals=True, plot_noise=True,
                     plot_cost=True, plot_iter=True, plot_status=True, plot_cov_matrix = True,
                     figsize=(15, 15), verbose=False):
    """
    Plot aggregated results from a list of MheIterationResult.
    """
    if not results:
        print("No results to plot.")
        return

    # Determine which subplots are active and in which order
    active_plots = []
    if plot_states:   active_plots.append('states')
    if plot_params:   active_plots.append('params')
    if plot_eigvals:  active_plots.append('eigvals')
    if plot_noise:    active_plots.append('noise')
    if plot_cost:     active_plots.append('cost')
    if plot_iter:     active_plots.append('iter')
    if plot_status:   active_plots.append('status')
    if plot_cov_matrix:   active_plots.append('cov_matrix')

    n_plots = len(active_plots)
    if n_plots == 0:
        print("Nothing to plot.")
        return

    # Height ratios: first plot (if states) gets double height, others get 1
    ratios = [2 if p == 'states' else 1 for p in active_plots]

    fig, axs = plt.subplots(n_plots, 1, figsize=figsize,
                            gridspec_kw={'height_ratios': ratios},
                            squeeze=False)
    axs = axs.flatten()
    plot_idx = 0

    # ----- Data concatenation (unchanged) -----
    t_full = []
    measured_full = []
    estimated_full = []
    params_full = []

    for idx, res in enumerate(results):
        t = np.asarray(res.t_batch)
        meas = np.asarray(res.simY)
        est = np.asarray(res.state_est)
        params = np.asarray(res.param_est)

        n_points = min(len(t), len(est), len(meas))
        t = t[:n_points]
        meas = meas[:n_points]
        est = est[:n_points]

        if params.ndim == 1:
            params_2d = np.tile(params, (n_points, 1))
        else:
            params_2d = params[:n_points]

        if idx == 0:
            start = 0
        else:
            start = min(overlap, n_points) if overlap < n_points else n_points

        t_full.extend(t[start:])
        measured_full.extend(meas[start:])
        estimated_full.extend(est[start:])
        params_full.extend(params_2d[start:])

        if verbose:
            print(f"Window {idx}: n_points={n_points}, start={start}, added={n_points-start}")

    t_full = np.array(t_full)
    measured_full = np.array(measured_full)
    estimated_full = np.array(estimated_full)
    params_full = np.array(params_full)

    min_len = min(len(t_full), len(measured_full), len(estimated_full), len(params_full))
    t_full = t_full[:min_len]
    print(estimated_full.shape)
    measured_full = measured_full[:min_len]
    print(estimated_full.shape)
    estimated_full = estimated_full[:min_len]
    params_full = params_full[:min_len]
   
    # ----- Plotting -----
    for p in active_plots:
        ax = axs[plot_idx]
        plot_idx += 1

        if p == 'states':
            ax.set_title("States: Measured (dashed) vs Estimated (solid)")
            n_obs = measured_full.shape[1]
            n_x = estimated_full.shape[1]
            for i in range(n_obs):
                ax.plot(t_full, measured_full[:, i], '--', label=f'Meas y_{i+1}')

            for i in range(n_x):
                ax.plot(t_full, estimated_full[:, i], '-', label=f'Est x_{i+1}')
            ax.set_xlabel("Time")
            ax.set_ylabel("State")
            ax.legend()
            ax.grid(True)

        elif p == 'params':
            ax.set_title("Parameter estimates over time")
            ntheta = params_full.shape[1]
            for i in range(ntheta):
                ax.plot(t_full, params_full[:, i], label=f'θ_{i+1} estimated')
            if initial_params is not None:
                for i, val in enumerate(initial_params):
                    ax.axhline(y=val, linestyle='--', color=f'C{i}', alpha=0.7,
                               label=f'θ_{i+1} initial')
            if theta_true is not None:
                for i, val in enumerate(theta_true):
                    ax.axhline(y=val, linestyle=':', color=f'C{i}', alpha=0.8,
                               label=f'θ_{i+1} true')
            ax.set_xlabel("Time")
            ax.set_ylabel("Parameter value")
            ax.legend()
            ax.grid(True)

        elif p == 'eigvals':
            ax.set_title("FIM eigenvalues per window (log scale)")
            n_theta = results[0].eigvals.shape[0]
            eig_vals_matrix = np.array([res.eigvals for res in results])
            eig_vals_sorted = np.sort(eig_vals_matrix, axis=1)[:, ::-1]
            for i in range(n_theta):
                ax.semilogy(eig_vals_sorted[:, i], marker='o', label=f'λ_{i+1}')
            ax.set_xlabel("Window index")
            ax.set_ylabel("Eigenvalue magnitude")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.7)

        elif p == 'cov_matrix':
            ax.set_title("Parameter standard deviation (sqrt of diag(cov))")
            # Determine number of parameters from the first result's covariance matrix
            n_theta = results[0].eigvals.shape[0]   # because fim is flattened
            # Extract diagonal standard deviations per window
            std_vals = []
            for res in results:
                cov_flat = res.cov_matrix
                cov_mat = np.array(cov_flat).reshape(n_theta, n_theta)
                diag = np.diag(cov_mat)
                std = np.sqrt(diag)
                std_vals.append(std)
            std_vals = np.array(std_vals)   # shape (n_windows, n_theta)
            for i in range(n_theta):
                ax.plot(range(len(std_vals)), std_vals[:, i], marker='o', label=f'θ_{i+1} std')
            ax.set_xlabel("Window index")
            ax.set_ylabel("Standard deviation")
            ax.legend()
            ax.grid(True)
        elif p == 'noise':
            ax.set_title("Process noise distribution")
            all_noise = np.concatenate([res.noise_est.flatten() for res in results])
            ax.hist(all_noise, bins=50, alpha=0.7, density=True)
            ax.set_xlabel("Noise value")
            ax.set_ylabel("Density")
            ax.grid(True)

        elif p == 'cost':
            ax.set_title("Cost value per window")
            cost_vals = [res.cost_value for res in results]
            ax.plot(range(len(cost_vals)), cost_vals, marker='o')
            ax.set_xlabel("Window index")
            ax.set_ylabel("Cost")
            ax.grid(True)
            ax.set_yscale('log')

        elif p == 'iter':
            ax.set_title("SQP iterations per window")
            iter_vals = [res.sqp_iter for res in results]
            ax.plot(range(len(iter_vals)), iter_vals, marker='o')
            ax.set_xlabel("Window index")
            ax.set_ylabel("Iterations")
            ax.grid(True)

        elif p == 'status':
            ax.set_title("Solver status per window (0 = success)")
            status_vals = [res.status for res in results]
            ax.plot(range(len(status_vals)), status_vals, marker='o', linestyle='-')
            ax.set_xlabel("Window index")
            ax.set_ylabel("Status")
            ax.set_yticks(sorted(set(status_vals)))
            ax.grid(True)

    plt.tight_layout()
    plt.show()

@dataclass
class MheEstimationData:
    """Container for MHE estimation results."""
    simXest: np.ndarray      # (N+1, nx) – estimated states at all nodes
    simWest: np.ndarray      # (N, nx)   – estimated process noise at each step
    sim_param_est: np.ndarray # (N+1, param_length) – estimated parameters at all nodes
    cost_value: float        # final cost value
    sqp_iter: int            # number of SQP iterations


def regularize_fim(F, tau_ratio=1e-4, zero_threshold=1e-3, large_penalty=1e6, add_ridge=1e-9):
    """
    Спектральная регуляризация FIM.
    
    Для собственных чисел < zero_threshold → заменяем на large_penalty (сильный приор).
    Для собственных чисел < tau_ratio * max_eig, но >= zero_threshold → заменяем на tau.
    """
    F = (F + F.T) / 2.0
    eigvals, eigvecs = la.eigh(F)
    # сортируем по убыванию
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    max_eig = eigvals[0]
    tau = max(tau_ratio * max_eig, 1e-12)
    
    new_eigvals = eigvals.copy()
    for i, val in enumerate(eigvals):
        if val < zero_threshold:
            new_eigvals[i] = large_penalty
        elif val < tau:
            new_eigvals[i] = tau
    
    F_reg = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    F_reg += add_ridge * np.eye(F.shape[0])
    return F_reg, eigvals, new_eigvals

def set_mhe_solver(mhe_model: MheModel, 
               acados_solver_mhe: AcadosOcpSolver, 
               simY: np.array,  
               simU: np.array, 
               initial_x0: np.array, 
               initial_theta: np.array, 
               N: int,
               P0 = np.array) -> tuple :
    assert(len(initial_x0) == mhe_model.state_length)
    assert(len(initial_theta) == mhe_model.param_length)
    #nx = len(initial_x0)
    x_prior = np.hstack((initial_x0, initial_theta))
    for j in range(N):   
        p_ext = np.hstack((simU[j, :], simY[j, :], x_prior, P0.flatten()))
        acados_solver_mhe.set(j, "p", p_ext)
        # if j == 0:
        #     # Для первого узла используем переданное initial_x0 (которое уже должно быть согласовано)
        #     x_aug = np.hstack((initial_x0, initial_theta))
        #     acados_solver_mhe.set(j, "x", x_aug)

def get_mhe_estimated_data(mhe_model: MheModel, acados_solver_mhe: AcadosOcpSolver, N: int):
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
    
    simXest = np.zeros((N + 1, nx))
    simWest = np.zeros((N, nx))
    sim_param_est = np.zeros(param_length,)

    # Fill data for nodes 0..N-1 (the first N nodes)
    for i in range(N):
        x_augmented = acados_solver_mhe.get(i, "x")
        simXest[i, :] = x_augmented[:nx]
        simWest[i, :] = acados_solver_mhe.get(i, "u")

    # Get the state at the final node (index N)
    x_final = acados_solver_mhe.get(N, "x")
    simXest[N, :] = x_final[:nx]
    sim_param_est = x_final[nx : nx + param_length]

    # Retrieve cost and iterations (available from the solver after solving)

    cost_value = acados_solver_mhe.get_cost()
    sqp_iter = acados_solver_mhe.get_stats('sqp_iter')

    return MheEstimationData(
        simXest=simXest,
        simWest=simWest,
        sim_param_est=sim_param_est,
        cost_value=cost_value,
        sqp_iter=sqp_iter
    )