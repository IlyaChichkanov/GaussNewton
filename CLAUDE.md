# GaussNewton — оценка параметров ОДУ

Идентификация параметров θ систем ОДУ по измерениям: метод Гаусса–Ньютона +
multiple shooting. Два интегратора чувствительностей: вариационные уравнения
(JAX/scipy, явные методы) и ортогональные коллокации Радо IIA (жёсткие системы).

## Структура и связи файлов

- `commom_utils/ode_system.py`
  - `ODESystem` — абстрактная символьная модель (CasADi): `get_derivative`,
    `observation`, `get_input_signals`; размерности `nx`, `np`, `nu`, `n_obs`.
    `get_input_signals` вызывается ИЗНУТРИ правой части ОДУ — с трассируемым t
    под jax odeint и с массивом t в коллокациях: только `jnp`, без `math.*`
    и питоновских `if t < ...` (нужен `jnp.where`).
  - `SystemJacobian` — компилирует CasADi-функции f, h и их якобианов; интеграторы
    расширенной системы (состояние + чувствительности) через scipy `solve_ivp` и
    jax `odeint`. Ключевые методы: `get_jacobian_solution`, `get_solution`,
    `get_jacobian_solution_jax_batch` (vmap+jit, группировка шутов по длине сетки),
    `observation_batch`, `inverse_h`.
  - `SyntheticDataGenerator` — генерация тестовых данных.
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
- `gauss_newton/gauss_newton_math.py`
  - `MultipleShooting` — сборка задачи: `solve(theta_full)` → `(J, R, J_G, R_G)`
    (измерительные невязки и невязки непрерывности); `make_full_theta`.
  - `compute_delta_gn` — μ-регуляризованный ККТ-шаг (см. математику ниже).
  - `run_optimization` — итерации ГН с принятием/откатом шага и μ-расписанием.
  - `compute_parameter_covariance` — маргинальная ковариация θ
    (Шур-комплемент θ-блока по полной системе [J; J_G]).
- **Два слоя поверх этого (не смешивать):**
  - `gauss_newton/normal_equations.py` — КАК получить H и g. `NormalEquations`
    (H, g, J_G, R_G, rss, n_rows) + методы `merit(mu)`, `mu_curvature()`,
    `covariance_theta(n_theta)`. Два источника: `from_jacobian(J,R,J_G,R_G)`
    (H = JᵀJ) и `AccumulateMixin.normal_equations(theta_full)` — накопление по
    документу «MS and Orthogonal Collocations»: H = ΣJᵢᵀJᵢ копится einsum-ами по
    точкам шута (стрелочная структура: θθ-блок общий, θс_j/c_jc_j по шутам),
    большая J не строится и JᵀJ не перемножается. Классы
    `MultipleShootingAccum`, `CollocationShootingAccum`; адаптер
    `normal_equations_of(problem, theta_full)`. Имена — как в документе:
    `J_theta`/`J_s`, `H_theta`/`H_theta_s`/`H_s`, `g_theta`/`g_s` (s ≡ c_j).
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
  `adaptive_regularization.ipynb` (смысл μ, Нильсен-λ, Пауэлл-μ, эксперименты).
- `experiments/` — реальные данные и ноутбуки (CSV в .gitignore);
  `experiments/data_utils.py` — `LogReaderV2`, `theta_to_physical`.
- `pytests/` — `uv run pytest pytests/` (68 тестов с acados; mhe/mpc-тесты
  пропускаются без него через `pytest.importorskip`): gauss_newton_test,
  collocation_test, adaptive_test, accumulated_test, collocation_accum_test,
  **jacobian_fd_test** (единственная сверка с ВНЕШНИМ эталоном — конечными
  разностями; остальные тесты взаимные и делят ядро `shoot_rows`).
  mhe_test/mpc_test пропускаются без acados (`pytest.importorskip`).
  Фигуры plotly по умолчанию не открываются — `GN_TEST_PLOT=1` включает.
- `tools/setup_repo.sh` — регистрирует git-фильтр `nbstrip` (`.gitattributes`
  объявляет `filter=nbstrip`, но сам фильтр локальный и в репозиторий не
  попадает; без запуска скрипта тяжёлый вывод ноутбуков вернётся в историю).

## Математическая суть

Неизвестные `theta_full = [θ; c_1..c_T]` (параметры + начальные состояния шутов).

- Измерительные невязки: r_i = W(y_i − h(x(t_i))), J — их якобиан по [θ; c].
- Непрерывность: G_j = x_j(t_{j+1}; c_j, θ) − c_{j+1} → (J_G, R_G).
- Шаг (`compute_delta_gn`): решается седловая система
  `[[H + λ_reg·I + λ·diag(H), J_Gᵀ], [J_G, −μI]] · δ = [Jᵀ R; R_G]`, H = JᵀJ.
  μ — релаксация ограничений непрерывности: после успешного шага μ ← μ·mu_dec
  (не ниже mu_min), после неудачного μ ← μ/mu_dec; шаг принимается при
  trial_cost ≤ best_cost·1.1, cost = ‖R‖² + ‖R_G‖².
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
  `[x; J_θ.flatten(); J_c.flatten()]` (C-order), столбцы = точки t_eval.
- `solve(theta_full)` → `(J, R, J_G, R_G)`; `run_optimization`/`compute_delta_gn`/
  `compute_parameter_covariance` знают только этот интерфейс, интегратор им безразличен.
- Существующий код не менять без явной просьбы — новые возможности через
  новые классы/файлы (так сделан CollocationShooting).

## Грабли (проверено на практике)

- `jax.experimental.odeint`: допуски по умолчанию 1.4e-8 — главный тормоз;
  всегда передавать `rtol=self.RTOL, atol=self.ATOL`.
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
