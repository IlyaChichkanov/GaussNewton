import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# acados is built from source, not installed from PyPI; without it the whole
# module is skipped, otherwise collection of all of pytests/ would fail
pytest.importorskip("acados_template",
                    reason="acados_template is not installed, see README")

from commom_utils.systems import KinematicBycicleErrors
from mpc.mpc_base_interface import KinematicMpcCodegenerator
from mpc.params import CarParams, MpcParams
from mpc.mpc_control_utils import LateralMPCController, reset_solver_initial_guess
from mpc.mpc_sim_utils import Simulator, SinusoidalCurveScenario, SimulationPlotter


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
# Simulation parameters
SIMULATION_TIME = 15.0      # seconds
TRACKING_TOLERANCE = 0.5   # largest acceptable lateral error, m

# ---------------------------------------------------------------------
# Fixture and test
# ---------------------------------------------------------------------
def test_mpc_tracking():
    """
    Runs the MPC controller on a kinematic bicycle model and checks that the
    lateral error stays under the tolerance.
    """

    scenario = SinusoidalCurveScenario(velocity=10.0, curv_amplitude=0.01, frequency=0.08)

    
    tmp_gen_dir = Path.cwd() / "tmp_generated"
    if tmp_gen_dir.exists():
        shutil.rmtree(tmp_gen_dir)

    try:

        # MPC solver code generation
        code_generator = KinematicMpcCodegenerator(mpc_params, tmp_gen_dir, "mpc_test")
        solver = code_generator.generate_code()
        reset_solver_initial_guess(solver)

        # Reference trajectory over time
        t_sim = np.arange(0.0, SIMULATION_TIME + ts, ts)
        trajectory = scenario.create_trajectory(t_sim)

        # System and scenario
        system = KinematicBycicleErrors(wheelbase=wheel_base)

        # Controller and simulation
        controller = LateralMPCController(solver, trajectory)
        sim = Simulator(system, controller, trajectory, model_params = np.array([0.2, 0]), delay_cycles=11, use_jax=False)
        states, controls = sim.run(t_sim, x0=np.array([0.2, 0]))
        #mpc_params.print()
        plot = False
        if(plot):
            plotter = SimulationPlotter(t_sim, states, controls, trajectory=trajectory, model=system)
            fig, axs = plotter.plot_all(include_jerk=True, include_comfort=True)
            plt.show()

        # Lateral error (the first state variable)
        lateral_error = np.abs(states[:, 0])
        max_error = np.max(lateral_error)
        print(f"Largest lateral error: {max_error:.4f} m (tolerance: {TRACKING_TOLERANCE} m)")

        # Control bounds must hold
        u_limits = car_params.u_max
        control_inputs = np.array(controls).flatten()
        assert np.all(np.abs(control_inputs) <= u_limits + 1e-6), "control left its bounds"

        # The tracking accuracy check itself
        assert max_error <= TRACKING_TOLERANCE, \
            f"MPC error {max_error:.4f} m exceeds the tolerance {TRACKING_TOLERANCE} m"
    finally:
        # Always remove the temporary directory afterwards
        if tmp_gen_dir.exists():
            shutil.rmtree(tmp_gen_dir)

