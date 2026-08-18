# -*- coding: utf-8 -*-
"""Smoke-тест всех моделей из commom_utils/systems.py.

Раньше несогласованность модели с объявленными размерностями всплывала только
при попытке ею воспользоваться — в ноутбуке или в MHE. Здесь каждая модель
строится и проверяется на самосогласованность:

    - конструктор отрабатывает (observation должна возвращать SX, а не кортеж);
    - get_derivative даёт ровно nx строк;
    - observation даёт ровно n_obs строк;
    - get_input_signals(t) даёт ровно nu сигналов;
    - CompiledModel компилирует f, h и их якобианы.

Тест ловит ровно тот класс ошибок, что был найден в ревью: Pendulum возвращал
питоновский кортеж, RosenzweigMacArthur объявляла np=1 при шести
распаковываемых параметрах.
"""
from pathlib import Path
import sys

import casadi as ca
import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from commom_utils import systems as S
from commom_utils.ode_system import CompiledModel, ODESystem

# Аргументы конструктора для моделей, которым они нужны
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
    assert len(ALL_SYSTEMS) >= 10, f"нашлось всего {len(ALL_SYSTEMS)} моделей"


@pytest.mark.parametrize("name,cls", ALL_SYSTEMS, ids=[n for n, _ in ALL_SYSTEMS])
def test_system_dimensions_self_consistent(name, cls):
    system = cls(*SYSTEM_ARGS.get(name, ()))

    # Размерности объявлены положительными и целыми
    assert system.nx >= 1, f"{name}: nx={system.nx}"
    assert system.n_theta >= 1, f"{name}: n_theta={system.n_theta}"
    assert system.nu >= 0, f"{name}: nu={system.nu}"
    assert system.n_obs >= 1, f"{name}: n_obs={system.n_obs}"

    x = ca.SX.sym("x", system.nx)
    theta = ca.SX.sym("theta", system.n_theta)
    u = ca.SX.sym("u", system.nu)

    # Правая часть: ровно nx компонент.
    # Ловит RosenzweigMacArthur (np=1, а распаковывалось 6 параметров):
    # обращение к theta[5] при np=1 падает здесь.
    f = ca.vertcat(system.get_derivative(x, theta, u))
    assert f.shape[0] == system.nx, \
        f"{name}: get_derivative вернул {f.shape[0]} строк, ожидалось nx={system.nx}"

    # Наблюдение: ровно n_obs компонент и это SX, а не кортеж
    h = system.observation(x, theta, u)
    assert isinstance(h, (ca.SX, ca.MX)), \
        f"{name}: observation вернула {type(h).__name__}, ожидался SX " \
        f"(кортеж ломает вычисление n_obs в ODESystem.__init__)"
    assert h.shape[0] == system.n_obs, \
        f"{name}: observation вернула {h.shape[0]} строк, ожидалось n_obs={system.n_obs}"

    # Входные сигналы. Часть моделей их не задаёт — тогда сигнал подставляет
    # create_system из SYSTEM_CONFIGS, и базовый метод возвращает []. Но если
    # модель метод ПЕРЕОПРЕДЕЛИЛА, длина обязана совпадать с nu.
    if type(system).get_input_signals is not ODESystem.get_input_signals:
        signals = system.get_input_signals(0.5)
        assert len(signals) == system.nu, \
            f"{name}: get_input_signals дал {len(signals)} сигналов, " \
            f"ожидалось nu={system.nu}"


@pytest.mark.parametrize("name,cls", ALL_SYSTEMS, ids=[n for n, _ in ALL_SYSTEMS])
def test_system_jacobian_compiles(name, cls):
    """CompiledModel строит f, h и якобианы, и они считаются численно."""
    base_system = cls(*SYSTEM_ARGS.get(name, ()))

    if (base_system.nu > 0
            and type(base_system).get_input_signals is ODESystem.get_input_signals):
        # Модель ждёт вход извне (его подставляет create_system) — для smoke
        # подставляем единицы, иначе CasADi получит 0 сигналов вместо nu
        nu = base_system.nu

        class _WithInputs(cls):
            def get_input_signals(self, t):
                return [1.0] * nu

        system = _WithInputs(*SYSTEM_ARGS.get(name, ()))
    else:
        system = base_system

    sj = CompiledModel(system)

    assert sj.dims() == (system.nx, system.n_theta, system.n_obs)

    # Численный вызов в неособой точке. Ни нули, ни единицы не годятся:
    # нули делят на vx/tau, а единицы обнуляют знаменатель 1 - c*d
    # у KinematicBycicleErrors. 0.1 по состоянию и 0.5 по параметрам
    # держат все знаменатели моделей вдали от нуля.
    state = 0.1 * np.ones(system.nx)
    theta = 0.5 * np.ones(system.n_theta)
    t = 0.5

    f_val = sj.f(state, t, theta)
    assert f_val.shape == (system.nx,)
    assert np.all(np.isfinite(f_val)), f"{name}: f не конечна"

    h_val = sj.h(state, t, theta)
    assert h_val.shape == (system.n_obs,)

    assert sj.df_dx(state, t, theta).shape == (system.nx, system.nx)
    assert sj.df_dtheta(state, t, theta).shape == (system.nx, system.n_theta)
    assert sj.dh_dx(state, t, theta).shape == (system.n_obs, system.nx)
