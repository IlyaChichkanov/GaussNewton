# GaussNewton — оценка параметров ОДУ

Идентификация параметров θ систем ОДУ по измерениям: метод Гаусса–Ньютона +
multiple shooting. Два интегратора чувствительностей: вариационные уравнения
(JAX/scipy, явные методы) и ортогональные коллокации Радо IIA (жёсткие системы).

## Структура и связи файлов

- `commom_utils/ode_system.py`
  - `ODESystem` — абстрактная символьная модель (CasADi): `get_derivative`,
    `observation`, `get_input_signals`; размерности `nx`, `n_theta`, `nu`,
    `n_obs` (атрибут `np` переименован — он затенял numpy).
    `get_input_signals` вызывается ИЗНУТРИ правой части ОДУ — с трассируемым t
    под jax odeint и с массивом t в коллокациях: только `jnp`, без `math.*`
    и питоновских `if t < ...` (нужен `jnp.where`).
  - `SystemJacobian` — компилирует CasADi-функции f, h и их якобианов; интеграторы
    расширенной системы (состояние + чувствительности) через scipy `solve_ivp` и
    jax `odeint`. Значения: `f` (правая часть) и `h` (наблюдение) — имена
    именно такие, `f_x_theta`/`h_x` удалены как вводящие в заблуждение.
    Якобианы: `df_dx`, `df_dtheta`, `dh_dx`, `dh_dtheta`. Размерности —
    `dims()` → `Dims(nx, n_theta, n_obs)` (NamedTuple). Интеграторы:
    `get_jacobian_solution`, `get_solution`, `get_jacobian_solution_jax_batch`
    (vmap+jit, группировка шутов по длине сетки), `observation_batch`,
    `inverse_h`. Вариационные уравнения записаны ровно дважды —
    `_variational_rhs` и `_variational_rhs_jax`, одинаковой схемой; входные
    сигналы в numpy-версии вычисляются один раз за вызов правой части.
  - `SystemIntegrator(SystemJacobian)` — интегрирование с УДЕРЖИВАЕМЫМ входом u
    от вызывающей стороны (симуляция MPC), а не из модели. `step`, `step_jax`,
    `integrate`, `get_lin_system_dynamics` → (A, B, D).
  - `SyntheticDataGenerator` — генерация тестовых данных.
- `commom_utils/sensitivity.py`
  - `SensitivityTrajectory` — `x (m,nx)`, `S_theta (m,nx,n_theta)`,
    `S_c (m,nx,nx)` плюс `unpack`/`pack`/`head`. Плоский layout
    `[x; S_theta.flatten(); S_c.flatten()]` навязан интеграторами и живёт
    ТОЛЬКО между ними и `unpack` — наружу идут именованные массивы.
  - `group_by_grid_length(t_grids)` — группировка шутов по длине сетки,
    общая для jax-батча и коллокационного марша.
- `commom_utils/collocation.py`
  - `RadauTables(K)` — узлы Радо IIA (K=1,2,3), матрица дифференцирования
    D̃=[d0|D1], таблица Бутчера `butcher_a = inv(D1)`.
  - `CollocationSystemJacobian(SystemJacobian)` — переопределяет ТОЛЬКО
    интеграторы: компилированный марш CasADi (`ca.rootfinder('newton')` +
    `mapaccum` + `map('thread')` по шутам). Python-эталон удалён (авг 2026,
    решение пользователя); внешние арбитры точности — вариационные уравнения
    (`collocation_test.py::test_integrator_matches_reference`) и конечные
    разности (`jacobian_fd_test`). Несходимость Ньютона НЕ бросается из C++
    (`error_on_fail=False`): каждый элемент возвращает масштабированную
    невязку `stage_res`, марш проверяет `max(stage_res) <= 10*newton_tol` сам
    и поднимает RuntimeError одной строкой — без дампов CasADi. Kwargs:
    `newton_tol` (это И `abstol`, И `abstolStep`; критерии работают как ИЛИ),
    `newton_maxiter=25`, `rootfinder_plugin`/`rootfinder_options` (переход на
    kinsol/fast_newton со своими опциями).
- `commom_utils/systems.py` — конкретные системы (LotkaVoltera, Attractor, ...).
- `gauss_newton/problem.py` (бывший `gauss_newton_math.py`) — ТОЛЬКО сборка задачи
  - `MultipleShooting` — `solve(theta_full)` → `(J, R, J_G, R_G)`;
    `make_full_theta`; ядро `shoot_rows`.
  - `UnknownsLayout` — раскладка `theta_full = [θ; c_1..c_T]` по батчам и шутам
    (`layout.theta`, `layout.c(batch, shoot)`), строится один раз в `add_batch`.
  - `ShootRows` — блоки одного шута. `J_theta`/`J_c` — строки якобиана НЕВЯЗОК;
    `S_theta_end`/`S_c_end` — ЧУВСТВИТЕЛЬНОСТИ состояния в конце шута (из них
    собираются строки непрерывности). Раньше и то и другое звалось `J_*`.
  - `solve()` и `NormalEquations.from_jacobian` оставлены как ЭТАЛОННАЯ сборка:
    в цикле оптимизации не участвуют, но дают `jacobian_fd_test` доступ к J и R
    — это единственная сверка с внешним эталоном.
- **Два слоя поверх этого (не смешивать):**
  - `gauss_newton/normal_equations.py` — КАК получить H и g. `NormalEquations`
    (H, g, J_G, R_G, rss, n_rows) + методы `merit(mu)`, `mu_curvature()`,
    `covariance_theta(n_theta)`. Два источника: `from_jacobian(J,R,J_G,R_G)`
    (H = JᵀJ) и `AccumulateMixin.normal_equations(theta_full)` — накопление по
    документу «MS and Orthogonal Collocations»: H = ΣJᵢᵀJᵢ копится einsum-ами по
    точкам шута (стрелочная структура: θθ-блок общий, θс_j/c_jc_j по шутам),
    большая J не строится и JᵀJ не перемножается. Классы
    `MultipleShootingAccum`, `CollocationShootingAccum`; адаптер
    `normal_equations_of(problem, theta_full)`. Индекс документа `s` — это наш
    `c_j`, второго имени нет: `J_theta`/`J_c`, `H_theta`/`H_theta_c`/`H_c`,
    `g_theta`/`g_c`. Здесь же `confidence_intervals` — рядом с ковариацией.
  - `gauss_newton/adaptive.py` — ЧТО с ними делать: `gn_step(ne, mu, lam)`
    (одна седловая система + pred) и `run_optimization_adaptive` — единственный
    цикл на оба пути. λ по gain ratio (Нильсен), μ стартует по кривизне
    ‖J_G‖²F/tr(H) и ужесточается по Пауэллу (только когда ‖R_G‖² не падает
    сама), принятие по ρ>0 для Φ_μ = ‖R‖² + (1/μ)‖R_G‖² (шаг — это ГН именно
    для Φ_μ: исключение множителей из седловой системы), остановка
    автоматическая. Ручной подбор μ0/mu_dec не нужен. `hist` содержит
    theta/cost/mu/lam/r_meas/r_cont/ci_low/ci_high — готово для `plot_solution`.
- Общее ядро обоих путей — `MultipleShooting.shoot_rows` → список `ShootRows`
  (интегрирование шутов + наблюдения + веса); `_solve_batch` собирает из них
  разреженную J, накопительный слой — сразу H и g. `continuity_rows` — строки
  непрерывности.
- `gauss_newton/collocation_shooting.py` — `CollocationShooting(MultipleShooting)`:
  подменяет `self.system` на `CollocationSystemJacobian`; передаёт `use_jax=True`,
  чтобы `_solve_batch` шёл через батчевый вход `get_jacobian_solution_jax_batch`
  (в коллокационном классе он реализован потоками, JAX не используется).
- `gauss_newton/utils.py` — `plot_solution`.
- Теория (ноутбуки — пользователь читает формулы ТОЛЬКО в ноутбуках, не в чате):
  `theory_gauss_newton.ipynb` (ГН + MS + ковариация),
  `collocation.ipynb` (OCFE, Радо IIA, рекурсии Ψ/Γ, верификация),
  `adaptive_regularization.ipynb` (смысл μ, Нильсен-λ, Пауэлл-μ, эксперименты;
  прежняя схема с ручным μ воспроизведена там ЛОКАЛЬНО — `run_optimization_manual`
  поверх библиотечного `gn_step`, — потому что сравнение с ней и есть содержание
  ноутбука, а второй копии математики в библиотеке быть не должно).
- `NOTATION.md` — таблица «теория ↔ код». Ноутбуки приведены к записи кода:
  где было `H` (матрица плана) и `G`, теперь `J` и `J_G`; `H` осталось только
  за нормальной матрицей `JᵀJ`.
- `experiments/` — реальные данные и ноутбуки (CSV в .gitignore);
  `experiments/data_utils.py` — `LogReaderV2`, `theta_to_physical`.
- `pytests/` — `uv run pytest pytests/` (68 passed, 2 skipped без acados):
  gauss_newton_test, collocation_test, adaptive_test, accumulated_test,
  collocation_accum_test, systems_smoke_test. Две сверки с ВНЕШНИМ эталоном:
  **jacobian_fd_test** (якобиан против конечных разностей) и шаг ГН против
  плотного `numpy.linalg.solve` в adaptive_test; остальные взаимные и делят
  ядро `shoot_rows`. **regression_test** — замороженные J/R/J_G/R_G/H/g/delta
  для четырёх задач: ловит изменения ЧИСЕЛ, которые не приводят к падению
  (перестановка осей, другой порядок вычислений).
  mhe_test/mpc_test пропускаются без acados (`pytest.importorskip`).
  Фигуры plotly по умолчанию не открываются — `GN_TEST_PLOT=1` включает.
- `tools/setup_repo.sh` — регистрирует git-фильтр `nbstrip` (`.gitattributes`
  объявляет `filter=nbstrip`, но сам фильтр локальный и в репозиторий не
  попадает; без запуска скрипта тяжёлый вывод ноутбуков вернётся в историю).

## Математическая суть

Неизвестные `theta_full = [θ; c_1..c_T]` (параметры + начальные состояния шутов).

- Измерительные невязки: r_i = W(y_i − h(x(t_i))), J — их якобиан по [θ; c].
- Непрерывность: G_j = x_j(t_{j+1}; c_j, θ) − c_{j+1} → (J_G, R_G).
- Шаг (`gauss_newton/adaptive.py::gn_step`, единственная реализация): решается
  седловая система
  `[[H + λ_reg·I + λ·diag(H), J_Gᵀ], [J_G, −μI]] · δ = [Jᵀ R; R_G]`, H = JᵀJ.
  μ — релаксация ограничений непрерывности; расписание μ и λ — автоматическое,
  см. `run_optimization_adaptive` выше.
- Чувствительности (классический путь): вариационные уравнения
  J̇_θ = f_x J_θ + f_θ, J̇_c = f_x J_c — интегрируются вместе с состоянием.
  Оба интегратора (scipy RK45, jax dopri) ЯВНЫЕ → на жёстких системах неприменимы.
- Коллокации (жёсткий путь): на элементе стадийные уравнения
  z = A x_prev + h·B·F(z, θ), где A = 1_K⊗I, B = a⊗I, a = D1⁻¹ — таблица Бутчера
  Радо IIA (K=3: порядок 5, L-устойчивость). Ньютон: M = I − h·B·blkdiag(f_x).
  Чувствительности по теореме о неявной функции (IND, точные производные
  дискретной схемы): Ψ = (e_Kᵀ⊗I)M⁻¹A, Γ = (e_Kᵀ⊗I)M⁻¹hB F_θ; рекурсии
  S^c ← Ψ S^c (S^c_0 = I), S^θ ← Ψ S^θ + Γ (S^θ_0 = 0).
  В коде Ψ/Γ получаются как `ca.jacobian` от выхода rootfinder —
  CasADi дифференцирует через него той же теоремой о неявной функции.
- Ковариация θ: σ²·(θ-блок (J_fullᵀJ_full)⁻¹), J_full = [J; J_G], через splu.

## Контракты (не ломать)

- `get_jacobian_solution(c0, θ, t_eval)` → матрица, строки
  `[x; S_θ.flatten(); S_c.flatten()]` (C-order), столбцы = точки t_eval.
  Разбирать её руками не надо: `SensitivityTrajectory.unpack(flat, nx, n_theta)`.
- `solve(theta_full)` → `(J, R, J_G, R_G)`; `normal_equations(theta_full)` →
  `NormalEquations`. Слой оптимизации знает только их, интегратор ему безразличен.
- Имена — по `NOTATION.md`. Одна величина = одно имя; если для величины
  появляется второе имя, что-то пошло не так.
- Существующий код не менять без явной просьбы — новые возможности через
  новые классы/файлы (так сделан CollocationShooting).
- `pytests/regression_test.py` — численный эталон шага ГН. Рефакторинг обязан
  проходить его без изменений; пересоздавать (`GN_REGEN_REFERENCE=1`) только
  когда изменение чисел осознанное и объяснимое.

## Грабли (проверено на практике)

- `jax.experimental.odeint`: допуски по умолчанию 1.4e-8 — главный тормоз;
  всегда передавать `rtol=self.RTOL, atol=self.ATOL`. Исключение —
  `SystemIntegrator.step_jax` (симуляция MPC): там допуски по умолчанию
  сохранены намеренно.
- Правая часть ОДУ зовётся десятки тысяч раз за solve, поэтому всё, что можно
  вычислить один раз за вызов, надо вычислять один раз. Входные сигналы
  считались трижды (через `f`, `df_dx`, `df_dtheta`) — устранение этого дало
  1.7× на numpy-пути (3982 → 2294 мс на Integrator, 16 точек, 2 шута).
- Python dict: `True == 1` → коллизия ключей кэша `(N, True)` и `(N, 1)`;
  либо строковые префиксы в ключах, либо раздельные словари
  (`_accum_cache`/`_map_cache` в коллокациях).
- `ca.rootfinder` не поддерживает codegen/JIT; `map('openmp')` медленнее
  `map('thread')` (CasADi отпускает GIL — потоков достаточно).
- `abstol` у Newton-rootfinder — АБСОЛЮТНЫЙ допуск на невязку Φ (масштаба
  состояния): при |x|~1e6 порог 1e-10 недостижим (floor округления ~1e-9),
  Ньютон «не сходится» только на больших данных. Лекарство — `abstolStep`
  (допуск по шагу; с abstol работает как ИЛИ, проверка шага ДО его применения).
- `error_on_fail=True` у rootfinder печатает многострочные C++-дампы входов,
  даже когда исключение перехвачено питоном (map('thread') усугубляет).
  Тихий путь: `error_on_fail=False` + свой выход `stage_res` (масштабированная
  невязка в решении) + проверка после марша. Несошедшийся элемент при
  error_on_fail=False возвращает последний итерат БЕЗ nan — без проверки
  stage_res это молча неверные числа.
- Первая сборка mapaccum-функций ~0.6 с (разово на длину сетки); JAX JIT ~5 с.
- Радо-базис только по коллокационным точкам (степень K−1, без τ_0=0) —
  ВЫРОЖДЕН (константы в ядре D); правильная постановка — степень K с τ_0=0.
- РАЗРЫВНЫЙ по времени вход ломает точность чувствительностей у явного
  адаптивного solve_ivp: он перешагивает излом, контроль ошибки там не
  работает, якобиан расходится с конечными разностями на 3+ порядка
  (`jacobian_fd_test::test_discontinuous_input_degrades_sensitivities`).
  Обход: граница шута в точке разрыва либо коллокации. У коллокаций этой
  проблемы нет — сетка элементов фиксирована.
- Сверка якобиана конечными разностями: шаг 1e-7 годится не всегда. Ошибка,
  падающая строго как 1/h, — это ШУМ ОКРУГЛЕНИЯ интегратора, а не баг
  якобиана (у Integrator состояние растёт как t², нужен шаг ~1e-4).
  Отличить от настоящей ошибки помогает прогон той же задачи на коллокациях:
  IND даёт точные производные схемы и согласуется до ~1e-10.
- `df_dx_jax` отдаёт (1, nx, nx), `df_dtheta_jax` — уже (nx, np). Матумножение
  на (1,·,·) молча даёт лишнюю ось; в `make_full_system_jax` форма приводится
  явным `.reshape`.

## Производительность (реальные данные, 8000 точек, 10 шутов)

- MultipleShooting (JAX): solve ~0.2–0.4 с; CollocationShooting: ~0.33–0.43 с
  (профиль: ~0.2 с марш в C++, ~0.05 с numpy-рекурсии S, ~0.03 с observation).
