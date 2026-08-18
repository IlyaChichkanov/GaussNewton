import pytest
import numpy as np
import os
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import SyntheticDataGenerator, SystemJacobian
from gauss_newton.problem import MultipleShooting
from gauss_newton.adaptive import run_optimization_adaptive
from gauss_newton.utils import plot_solution

# Фигуры plotly по умолчанию не открываем: fig.show() вешает headless-прогон
PLOT = os.environ.get("GN_TEST_PLOT", "0") not in ("0", "", "false", "False")
# Словарь доступных систем с их конфигурациями
SYSTEMS_CONFIG = {
    "LotkaVolterra": {
        "class": LotkaVoltera,
        "true_params": np.array([1.2, 0.4, 0.3, 0.1]),
        "initial_state": np.array([6.0, 5.0]),
        "time_interval": (0.0, 4.0),
        "n_measurements": 50,
        "noise_sigma": 1e-15,
        "N_shoot": 1,
        "mu": 0.0, 
        "theta_init": np.array([1.0, 0.5, 0.2, 0.05]),  # смещённое начальное приближение
    },
    "Attractor": {
        "class": Attractor,
        "true_params": np.array([10.0, 28.0, 8.0/3.0]),  # пример для Лоренца
        "initial_state": np.array([1.0, 1.0, 1.0]),
        "time_interval": (0.0, 5.0),
        "n_measurements": 100,
        "sigma": 0.05,
        "N_shoot": 20,
        "mu": 100.0, 
        "theta_init": np.array([0.0, 0.0, 0.0]),
    }
}

def pytest_generate_tests(metafunc):
    """Динамическая параметризация: для каждого теста, который использует fixture 'system_name',
       создаём отдельные вызовы для каждой системы из SYSTEMS_CONFIG."""
    if "system_name" in metafunc.fixturenames:
        metafunc.parametrize("system_name", list(SYSTEMS_CONFIG.keys()), scope="function")

@pytest.fixture
def system_config(system_name):
    return SYSTEMS_CONFIG[system_name]

@pytest.fixture
def system(system_config):
    """Создаёт экземпляр системы по имени."""
    return system_config["class"]()

@pytest.fixture
def true_params(system_config):
    return system_config["true_params"]

@pytest.fixture
def initial_state(system_config):
    return system_config["initial_state"]

@pytest.fixture
def time_interval(system_config):
    return system_config["time_interval"]

@pytest.fixture
def synthetic_data(system, true_params, initial_state, time_interval, system_config):
    gen = SyntheticDataGenerator(
        system,
        sigma=system_config.get("noise_sigma", 0.01),
        perturb_initial=True,
        perturbation_scale=0.0,
        use_jax=True
    )
    t_batches, meas_batches, state_batches, _ = gen.generate(
        c0=initial_state,
        theta=true_params,
        time_intervals=[time_interval],
        n_measurements=system_config["n_measurements"]
    )
    return t_batches[0], meas_batches[0], state_batches[0]

def test_identification(system, true_params, synthetic_data, system_config):

    t_meas, meas_batch, state_true_batch = synthetic_data

    N_shoot = system_config["N_shoot"]
    gamma = np.ones(system.n_obs)  # веса измерений
    c0_cost = 1.0
    use_jax = False

    ms = MultipleShooting(
        system=system,
        N_shoot=N_shoot,
        gamma=gamma,
        c0_cost=c0_cost,
        use_jax=use_jax
    )
    ms.add_batch(meas_batch, t_meas)

    theta_init = system_config["theta_init"]
    theta_full = ms.make_full_theta(theta_init)


    # mu и lambda подбираются автоматически: system_config["mu"] больше не нужен
    theta_full_opt, hist = run_optimization_adaptive(ms, theta_full, n_iter=20)

    # Оценённые параметры
    n_theta = len(true_params)
    theta_est = theta_full_opt[:n_theta]

    # Относительная ошибка
    rel_error = np.abs((theta_est - true_params) / true_params)
    print(f"\nSystem: {system_config['class'].__name__}")
    print("True parameters:     ", true_params)
    print("Estimated parameters:", theta_est)
    print("Relative error:      ", rel_error)

    # Проверка
    assert np.all(rel_error < 0.05), \
        f"Estimation error too high: {rel_error}"

    if PLOT:
        fig = plot_solution(
            problem = ms,
            theta_hist = hist["theta"],
            plot_xy=1,
            plot_theta=True,
            plot_trajectory=0,
            plot_true_solution=False,
            plot_residuals=True,
            plot_measurements = 1,
            r_meas_hist=hist["r_meas"],
            r_cont_hist=hist["r_cont"],
            index=-1,
            theta_true=None,
        )
        fig.show()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])