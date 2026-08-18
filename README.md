# GaussNewton — идентификация параметров ОДУ

[![CI](https://github.com/IlyaChichkanov/GaussNewton/actions/workflows/ci.yml/badge.svg)](https://github.com/IlyaChichkanov/GaussNewton/actions/workflows/ci.yml)

Оценка параметров θ нелинейных динамических систем по зашумлённым измерениям:
**метод Гаусса–Ньютона + multiple shooting**, с совместной оценкой θ и начальных
состояний шутов. Дополнительно — **MHE** (скользящее окно, рекурсивный вариант
для реального времени) и **MPC** на acados.

![gauss newton visualisation](gauss_newton/demo.gif)

## Возможности

- Символьное описание модели на CasADi, якобианы генерируются автоматически.
- **Два интегратора чувствительностей:**
  - вариационные уравнения (`scipy solve_ivp` / `jax odeint`) — явные методы,
    для нежёстких систем;
  - ортогональные коллокации **Радо IIA** с точными производными дискретной
    схемы (IND) — L-устойчивы, применимы к жёстким системам.
- **Два способа собрать шаг:** через разреженную матрицу J или накоплением
  H = ΣJᵢᵀJᵢ и g = ΣJᵢᵀrᵢ, когда большую J строить не нужно.
- **Адаптивная регуляризация:** λ по gain ratio (Нильсен), μ по кривизне
  с ужесточением по Пауэллу — ручной подбор не требуется.
- Ковариация θ и доверительные интервалы (Шур-комплемент по полной системе
  [J; J_G], корреляция θ с начальными состояниями учитывается).
- Несколько батчей данных с общими параметрами; генератор синтетических данных.
- Визуализация на plotly: фазовые траектории 2D/3D, временные ряды, сходимость
  параметров с CI, невязки измерений и непрерывности.

## Установка

```bash
uv sync                 # зависимости
bash tools/setup_repo.sh   # git-фильтр, вычищающий вывод из ноутбуков
```

`tools/setup_repo.sh` обязателен при первом клонировании: `.gitattributes`
объявляет фильтр `nbstrip`, но сам фильтр — локальная настройка git и в
репозитории не хранится. Без него git молча закоммитит ноутбуки вместе с
картинками и встроенным `plotly.js` (сотни килобайт на файл).

## Быстрый старт

```python
import numpy as np
from commom_utils.systems import LotkaVoltera
from commom_utils.ode_system import SyntheticDataGenerator
from gauss_newton.normal_equations import MultipleShootingAccum
from gauss_newton.adaptive import run_optimization_adaptive

system = LotkaVoltera()

# синтетические данные
gen = SyntheticDataGenerator(system, sigma=0.01, use_jax=True)
t_batches, meas_batches, _, _ = gen.generate(
    c0=np.array([6.0, 5.0]), theta=np.array([1.2, 0.4, 0.3, 0.1]),
    time_intervals=[(0.0, 4.0)], n_measurements=200)

# задача: 5 шутов, единичные веса измерений
problem = MultipleShootingAccum(system, N_shoot=5, gamma=np.ones(system.n_obs))
problem.add_batch(meas_batches[0], t_batches[0])

theta_full = problem.make_full_theta(np.array([1.0, 0.5, 0.2, 0.05]))
theta_opt, hist = run_optimization_adaptive(problem, theta_full, verbose=True)

print("θ:", theta_opt[:system.n_theta])
print("95% CI:", hist["ci_low"][-1], hist["ci_high"][-1])
```

Для жёстких систем достаточно поменять класс задачи — интерфейс тот же:

```python
from gauss_newton.normal_equations import CollocationShootingAccum
problem = CollocationShootingAccum(system, N_shoot=5, gamma=np.ones(system.n_obs),
                                   K=3, n_sub=2)   # Радо IIA, 3 стадии
```

### Свои модели

Наследуйтесь от `ODESystem` (`commom_utils/ode_system.py`) и задайте
`get_derivative`, при необходимости `observation` и `get_input_signals`:

```python
class MyModel(ODESystem):
    def __init__(self):
        super().__init__(nx=2, n_theta=3, nu=1)   # порядок: nx, n_theta, nu

    def get_derivative(self, state, theta, u):
        return ca.vertcat(...)                   # CasADi-выражение

    def observation(self, state, theta, u):
        return ca.vertcat(state[0])              # что реально измеряется

    def get_input_signals(self, t):
        return [jnp.sin(t)]                      # см. предупреждение ниже
```

**`get_input_signals` вызывается изнутри правой части ОДУ**, в том числе с
трассируемым временем под `jax odeint` и с массивом времени в коллокациях.
Поэтому: только `jnp`, никаких питоновских `if t < ...` (используйте
`jnp.where`) и `math.*`. Разрывные по времени входы дают неконтролируемую
ошибку у адаптивного явного интегратора — см. «Известные ограничения».

## Цикл оптимизации

Цикл один — `run_optimization_adaptive` (`gauss_newton/adaptive.py`), и ручной
подбор μ₀, `mu_dec` и λ для него не нужен:

| | как подбирается |
|---|---|
| λ (демпфер Марквардта) | по gain ratio, схема Нильсена |
| μ (вес невязок стыковки) | старт по кривизне ‖J_G‖²_F/tr(H), ужесточение по Пауэллу — только когда ‖R_G‖² не падает сама |
| принятие шага | ρ > 0 по Φ_μ = ‖R‖² + (1/μ)‖R_G‖², то есть по той функции, для которой шаг и посчитан |
| остановка | автоматическая: серия отказов, стагнация или pred ≈ 0 |

Прежний цикл с ручным расписанием μ (`run_optimization` + `compute_delta_gn`)
удалён: он был второй копией той же математики и успел разойтись с первой в
регуляризации диагонали. Схема воспроизведена локально в
`adaptive_regularization.ipynb` — там она предмет сравнения, а не рабочий код.

## Тесты

```bash
uv run pytest pytests/ -v         # весь набор (68 passed, 2 skipped)
uv run pytest pytests/jacobian_fd_test.py -v   # якобиан против конечных разностей
GN_TEST_PLOT=1 uv run pytest pytests/collocation_accum_test.py   # с графиками
```

Тесты `pytests/mhe_test.py` и `pytests/mpc_test.py` пропускаются, если не
установлен acados.

Состав: сверка накопительного пути с плотным, коллокаций с `solve_ivp`,
таблиц Радо с аналитическими значениями, шага ГН с плотным
`numpy.linalg.solve` той же седловой системы и — сверка с внешним эталоном —
якобиана с конечными разностями.

Отдельно `pytests/regression_test.py` держит замороженные матрицы шага
(`J`, `R`, `J_G`, `R_G`, `H`, `g`, `delta`) для четырёх задач: он ловит
изменения ЧИСЕЛ, которые не приводят к падению — перестановку осей, другой
порядок вычислений. Эталон пересоздаётся `GN_REGEN_REFERENCE=1`, и делать это
стоит только когда изменение осознанное.

## MHE и MPC

Требуют **acados**, который ставится из исходников, а не из PyPI:

```bash
git clone https://github.com/acados/acados.git && cd acados
git submodule update --init --recursive
mkdir build && cd build && cmake -DACADOS_WITH_QPOASES=ON .. && make install -j4
pip install -e ../interfaces/acados_template
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<acados>/lib
export ACADOS_SOURCE_DIR=<acados>
```

## Структура

```
commom_utils/
  ode_system.py      ODESystem, SystemJacobian, SystemIntegrator, генераторы данных
  sensitivity.py     SensitivityTrajectory (x, S_theta, S_c), группировка шутов
  collocation.py     таблицы Радо IIA и коллокационный интегратор
  systems.py         конкретные модели; system_config.py — их конфигурации
gauss_newton/
  problem.py         сборка задачи: ShootRows, UnknownsLayout, MultipleShooting
  normal_equations.py  NormalEquations (H, g), накопление, ковариация, CI
  adaptive.py        gn_step и run_optimization_adaptive
  collocation_shooting.py, utils.py
mhe/ mpc/         MHE и MPC на acados (кода, ноутбуков здесь больше нет)
experiments/
  sintetic_data/  прогоны на синтетике: gauss_newton_test, mhe_test
  real_data_cars/ прогоны на реальных данных машин (Ceed, Voyah)
  datasets/       сырые CSV и CAN-логи — вне git, см. datasets/README.md
  data_utils.py   LogReaderV2, theta_to_physical
pytests/          тесты
tools/            nbstrip.py (git-фильтр), setup_repo.sh
*.ipynb           теория: theory_gauss_newton, collocation,
                  adaptive_regularization, theory_mhe
```

Ноутбуки в `experiments/` начинаются с bootstrap-ячейки: она поднимается от
текущего каталога до `pyproject.toml`, кладёт корень в `sys.path` и задаёт
`DATASETS` (переопределяется переменной `GN_DATASETS`). Поэтому ноутбук
одинаково работает и из Jupyter, и при запуске раннером из корня.

**`NOTATION.md` — таблица «теория ↔ код»**: как формула из ноутбука называется
в коде. Подробная карта модулей и математика — в `CLAUDE.md`.

## Известные ограничения

- **Разрывные по времени входные сигналы.** Адаптивный явный интегратор не
  знает про излом, перешагивает его, и на содержащем разрыв интервале
  чувствительности теряют несколько порядков точности (зафиксировано в
  `pytests/jacobian_fd_test.py::test_discontinuous_input_degrades_sensitivities`).
  Обход: ставить границу шута в точку разрыва или использовать коллокации.
- **Коллокации идут фиксированным шагом**: точность определяется `n_sub`,
  оценки ошибки нет — проверяйте сходимость прогоном с `n_sub` и `2*n_sub`.
- `SyntheticDataGenerator` добавляет шум **к состояниям** до вычисления h(x)
  (шум процесса, не измерения); в `MHESyntheticDataGenerator` шум на выходе.
- Границы на θ есть только в MHE; в GN-части параметры не ограничены.
