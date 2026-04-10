import pytest
import numpy as np
import types
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils.systems import LotkaVoltera, Attractor
from commom_utils.ode_system import SyntheticDataGenerator, SystemJacobian
from gauss_newton.gauss_newton_math import MultipleShooting, run_optimization

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
        "N_shoot": 10,
        "mu": 20.0, 
        "theta_init": np.array([0.0, 0.0, 0.0]),
    }
    # Можно добавить другие системы:
    # "AnotherSystem": {...}
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
    """Генерирует синтетические данные с шумом."""
    gen = SyntheticDataGenerator(
        system,
        sigma=system_config.get("noise_sigma", 0.01),
        perturb_initial=True,
        perturbation_scale=0.0,
        use_jax=True
    )
    t_batches, meas_batches, state_batches = gen.generate(
        c0=initial_state,
        theta=true_params,
        time_intervals=[time_interval],
        n_measurements=system_config["n_measurements"]
    )
    return t_batches[0], meas_batches[0], state_batches[0]

def test_identification(system, true_params, synthetic_data, system_config):
    """
    Основной тест: выполняет идентификацию параметров для любой системы,
    заданной в SYSTEMS_CONFIG.
    """
    t_meas, meas_batch, state_true_batch = synthetic_data

    # Параметры конфигурации
    N_shoot = system_config["N_shoot"]
    gamma = np.ones(system.n_obs)  # веса измерений
    c0_cost = 1.0
    use_jax = False

    # Создаём SystemJacobian и MultipleShooting
    ms = MultipleShooting(
        system=system,
        N_shoot=N_shoot,
        gamma=gamma,
        c0_cost=c0_cost,
        use_jax=use_jax
    )
    ms.add_batch(state_true_batch, meas_batch, t_meas)

    # Начальное приближение параметров
    theta_init = system_config["theta_init"]
    theta_full = ms.make_full_theta(theta_init)

    # Конфигурация оптимизации
    config = types.SimpleNamespace()
    config.mu = system_config["mu"]
    config.n_iter = 20
    config.lambda_ = 0.001
    config.lambda_reg = 0.0

    # Запуск оптимизации
    theta_hist, r_meas_hist, r_cont_hist, theta_full_opt, ci_low_hist, ci_high_hist = run_optimization(
        problem=ms,
        config=config,
        theta_full=theta_full,
        system=system
    )

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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])