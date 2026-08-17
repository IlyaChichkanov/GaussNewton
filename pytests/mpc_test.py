import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
import sys
import numpy as np
import pytest
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

import pytest

# acados ставится из исходников, а не из PyPI: без него модуль
# целиком пропускается, иначе падала СБОРКА всего pytests/
pytest.importorskip("acados_template",
                    reason="acados_template не установлен — см. README")

from acados_template import AcadosOcp
from commom_utils.systems import KinematicBycicleErrors
from mpc.mpc_base_interface import KinematicMpcCodegenerator
from mpc.params import CarParams, MpcParams
from mpc.mpc_control_utils import LateralMPCController, reset_solver_initial_guess
from mpc.mpc_sim_utils import Simulator, SinusoidalCurveScenario, SimulationPlotter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

max_steering_wheel_angle = 440.0 # deg
max_steering_wheel_angle_rate = 350.0 # deg/s
steering_gear_ratio  = 14

u_max = np.deg2rad(max_steering_wheel_angle / steering_gear_ratio)  
ts = 0.02
mpc_horizont = 50
du_max = np.deg2rad(max_steering_wheel_angle_rate / steering_gear_ratio/ ts)
r_dist = 100.0
r_ang = 1.0
n_delay = 0
r_u = 0.2
r_du = 20.0
r_ddu = 1
r_jerk = 2
a_comf = 1.5
jerk_max = 5.0
jerk_comf = 2.0
wheel_base = 2.65
ddu_max = 1
r_dist = (r_dist/ wheel_base)
r_ang = (np.rad2deg(r_ang) / wheel_base)
r_w = 3



car_params = CarParams(
    gear_ratio=steering_gear_ratio,
    wheelbase=wheel_base,
    length=None,
    rear_overhang=None,
    u_max=u_max,
    du_max=du_max,
    ddu_max=ddu_max
)

mpc_params = MpcParams(
    car_params=car_params,
    use_dynamic_model=False,
    use_ddu_control=True,
    mpc_horizont=mpc_horizont,
    ts=ts,
    r_dist=r_dist,
    r_ang=r_ang,
    r_w=r_w,
    r_u=r_u,
    r_du=r_du,
    r_ddu=r_ddu,
    r_jerk=r_jerk,
    jerk_comf=jerk_comf,
    jerk_max=jerk_max,
    a_comf_max=5,
    final_cost=10.0
)
# Параметры симуляции
SIMULATION_TIME = 15.0      # секунд
TRACKING_TOLERANCE = 0.5   # максимально допустимая латеральная ошибка (м)

# ---------------------------------------------------------------------
# Фикстура и тест
# ---------------------------------------------------------------------
def test_mpc_tracking():
    """
    Запускает MPC-контроллер на кинематической модели велосипеда
    и проверяет, что поперечная ошибка не превышает заданного порога.
    """

    scenario = SinusoidalCurveScenario(velocity=10.0, curv_amplitude=0.01, frequency=0.08)

    
    tmp_gen_dir = Path.cwd() / "tmp_generated"
    if tmp_gen_dir.exists():
        shutil.rmtree(tmp_gen_dir)

    try:

        # Генерация solver'а MPC (code generation)
        code_generator = KinematicMpcCodegenerator(mpc_params, tmp_gen_dir, "mpc_test")
        solver = code_generator.generate_code()
        reset_solver_initial_guess(solver)

        # Траектория по времени
        t_sim = np.arange(0.0, SIMULATION_TIME + ts, ts)
        trajectory = scenario.create_trajectory(t_sim)

        # Система и сценарий
        system = KinematicBycicleErrors(wheelbase=wheel_base)

        # Контроллер и симуляция
        controller = LateralMPCController(solver, trajectory)
        sim = Simulator(system, controller, trajectory, model_params = np.array([0.2, 0]), delay_cycles=11, use_jax=False)
        states, controls = sim.run(t_sim, x0=np.array([0.2, 0]))
        #mpc_params.print()
        plot = False
        if(plot):
            plotter = SimulationPlotter(t_sim, states, controls, trajectory=trajectory, model=system)
            fig, axs = plotter.plot_all(include_jerk=True, include_comfort=True)
            plt.show()

        # Оценка поперечной ошибки (первая переменная состояния)
        lateral_error = np.abs(states[:, 0])
        max_error = np.max(lateral_error)
        print(f"Максимальная поперечная ошибка: {max_error:.4f} м (допуск: {TRACKING_TOLERANCE} м)")

        # Проверка соблюдения ограничений управления
        u_limits = car_params.u_max
        control_inputs = np.array(controls).flatten()
        assert np.all(np.abs(control_inputs) <= u_limits + 1e-6), "Выход управления за пределы ограничений"

        # Основная проверка точности слежения
        assert max_error <= TRACKING_TOLERANCE, \
            f"Ошибка MPC {max_error:.4f} м превышает допустимую {TRACKING_TOLERANCE} м"
    finally:
        # 2. Всегда удаляем временную директорию после теста
        if tmp_gen_dir.exists():
            shutil.rmtree(tmp_gen_dir)

