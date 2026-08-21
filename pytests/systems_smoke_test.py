"""Smoke test over every model in commom_utils/systems.py.

A model inconsistent with its declared sizes used to surface only when someone
tried to use it. Here each one is built and checked for self-consistency:

    - the constructor runs (observation must return an SX, not a tuple);
    - get_derivative gives exactly nx rows;
    - observation gives exactly n_obs rows;
    - get_input_signals(t) gives exactly nu signals;
    - CompiledModel compiles f, h and their Jacobians.
"""

import casadi as ca
import numpy as np
import pytest
from commom_utils import systems as S
from commom_utils.ode_system import CompiledModel, ODESystem

# Constructor arguments for the models that need them
SYSTEM_ARGS = {
    "LateralCarDynamic": (2.65,),
    "KinematicBycicleErrors": (2.65,),
    "KinematicBycicleActuator": (2.65,),
    "KinematicModel": (True,),
    "KinematicModelDelay": (2.65, 2),
    "OffsetEstimator": (2.65, 20.0),
    "DelaySystem": (2,),
    "DelayOffset": (2,),
}


def _all_system_classes():
    out = []
    for name in dir(S):
        obj = getattr(S, name)
        if (isinstance(obj, type) and issubclass(obj, ODESystem)
                and obj is not ODESystem):
            out.append((name, obj))
    return sorted(out)


ALL_SYSTEMS = _all_system_classes()


def test_system_list_is_not_empty():
    assert len(ALL_SYSTEMS) >= 10, f"only {len(ALL_SYSTEMS)} models were found"


@pytest.mark.parametrize("name,cls", ALL_SYSTEMS, ids=[n for n, _ in ALL_SYSTEMS])
def test_system_dimensions_self_consistent(name, cls):
    system = cls(*SYSTEM_ARGS.get(name, ()))

    # Sizes are declared as positive integers
    assert system.nx >= 1, f"{name}: nx={system.nx}"
    assert system.n_theta >= 1, f"{name}: n_theta={system.n_theta}"
    assert system.nu >= 0, f"{name}: nu={system.nu}"
    assert system.n_obs >= 1, f"{name}: n_obs={system.n_obs}"

    x = ca.SX.sym("x", system.nx)
    theta = ca.SX.sym("theta", system.n_theta)
    u = ca.SX.sym("u", system.nu)

    # Right-hand side: exactly nx components. This is what catches a model
    # that unpacks more parameters than it declares.
    f = ca.vertcat(system.get_derivative(x, theta, u))
    assert f.shape[0] == system.nx, \
        f"{name}: get_derivative returned {f.shape[0]} rows, expected nx={system.nx}"

    # Observation: exactly n_obs components, and an SX rather than a tuple
    h = system.observation(x, theta, u)
    assert isinstance(h, (ca.SX, ca.MX)), \
        f"{name}: observation returned {type(h).__name__}, expected SX " \
        f"(a tuple breaks the n_obs computation in ODESystem.__init__)"
    assert h.shape[0] == system.n_obs, \
        f"{name}: observation returned {h.shape[0]} rows, expected n_obs={system.n_obs}"

    # Input signals. Some models leave them to create_system, and the base
    # method returns []; but a model that OVERRIDES the method must return
    # exactly nu signals.
    if type(system).get_input_signals is not ODESystem.get_input_signals:
        signals = system.get_input_signals(0.5)
        assert len(signals) == system.nu, \
            f"{name}: get_input_signals gave {len(signals)} signals, " \
            f"expected nu={system.nu}"


@pytest.mark.parametrize("name,cls", ALL_SYSTEMS, ids=[n for n, _ in ALL_SYSTEMS])
def test_system_jacobian_compiles(name, cls):
    """CompiledModel builds f, h and the Jacobians, and they evaluate."""
    base_system = cls(*SYSTEM_ARGS.get(name, ()))

    if (base_system.nu > 0
            and type(base_system).get_input_signals is ODESystem.get_input_signals):
        # The model expects inputs from outside (create_system supplies them);
        # ones will do here, otherwise CasADi gets 0 signals instead of nu
        nu = base_system.nu

        class _WithInputs(cls):
            def get_input_signals(self, t):
                return [1.0] * nu

        system = _WithInputs(*SYSTEM_ARGS.get(name, ()))
    else:
        system = base_system

    sj = CompiledModel(system)

    assert sj.dims() == (system.nx, system.n_theta, system.n_obs)

    # Evaluate at a non-singular point. Neither zeros nor ones will do: zeros
    # divide by vx/tau, and ones zero the denominator 1 - c*d of
    # KinematicBycicleErrors. 0.1 for states and 0.5 for parameters keep every
    # denominator away from zero.
    state = 0.1 * np.ones(system.nx)
    theta = 0.5 * np.ones(system.n_theta)
    t = 0.5

    f_val = sj.f(state, t, theta)
    assert f_val.shape == (system.nx,)
    assert np.all(np.isfinite(f_val)), f"{name}: f is not finite"

    h_val = sj.h(state, t, theta)
    assert h_val.shape == (system.n_obs,)

    assert sj.df_dx(state, t, theta).shape == (system.nx, system.nx)
    assert sj.df_dtheta(state, t, theta).shape == (system.nx, system.n_theta)
    assert sj.dh_dx(state, t, theta).shape == (system.n_obs, system.nx)
