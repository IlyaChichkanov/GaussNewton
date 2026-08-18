# Обозначения: теория ↔ код

Одна величина — одно имя. Таблица нужна потому, что ноутбуки написаны в
математической записи, а код — в программной, и раньше они расходились:
буква `H` в `theory_gauss_newton.ipynb` означала матрицу плана, а в коде —
нормальную матрицу `JᵀJ`. Ноутбуки приведены к записи кода.

## Модель и данные

| Теория | Код | Что это |
|---|---|---|
| `x ∈ ℝ^{n_x}` | `x`, `nx` | состояние |
| `θ ∈ ℝ^{n_θ}` | `theta`, `n_theta` | оцениваемые параметры |
| `u(t)` | `u`, `nu` | входные сигналы, `ODESystem.get_input_signals` |
| `f(t, x, θ)` | `SystemJacobian.f` | правая часть ОДУ |
| `h(x, θ)` | `SystemJacobian.h` | функция измерений, `n_obs` компонент |
| `f_x`, `f_θ` | `df_dx`, `df_dtheta` | якобианы правой части |
| `h_x`, `h_θ` | `dh_dx`, `dh_dtheta` | якобианы наблюдения |
| `y_i` | `state_measured[i]` | измерение в точке `t_i` |
| `W_i = Σ_i^{-1}` | `gamma²` | веса; **`gamma` — это √W**, невязка домножается на неё, а стоимость берёт квадрат |

## Неизвестные и чувствительности

| Теория | Код | Что это |
|---|---|---|
| `p = [θ; c_1 … c_T]` | `theta_full`, `UnknownsLayout` | вектор неизвестных |
| `c_j` | `c0` шута, `layout.c(batch, shoot)` | начальное состояние шута `j` |
| `J_θ(t) = ∂x/∂θ` | `S_theta`, `(m, nx, n_theta)` | чувствительность по параметрам |
| `J_{c_0}(t) = ∂x/∂c_0` | `S_c`, `(m, nx, nx)` | чувствительность по начальному состоянию |
| — | `SensitivityTrajectory` | `x`, `S_theta`, `S_c` на сетке |
| `Ψ`, `Γ` | `Psi`, `Gamma` | переход и вклад параметров за элемент (коллокации) |

В документе «MS and Orthogonal Collocations» начальное состояние обозначено
`s`; в multiple shooting это `c_j` конкретного шута, и второго имени в коде
нет: `ShootRows.J_c`, `H_theta_c`, `H_c`, `g_c`.

## Система Гаусса–Ньютона

| Теория (ноутбуки **до** правки) | Код и ноутбуки **сейчас** | Что это |
|---|---|---|
| `H(p)` | `J` | якобиан измерительных невязок |
| `G(p)` | `J_G` | якобиан невязок непрерывности |
| `r`, `h_cont` | `R`, `R_G` | сами невязки |
| `HᵀWH` | `H = JᵀJ` | **нормальная** матрица (веса уже внутри `J`) |
| `HᵀWr` | `g = JᵀR` | градиент (антиградиент цели) |
| `ρ` | `1/mu` | вес квадратичного штрафа |
| `diag(λ)` | `lambda_reg·I + lam·diag(H)` | демпфирование шага |
| `λ` (множители) | `nu` в `gn_step` | множители Лагранжа |

Шаг (`gauss_newton/adaptive.py::gn_step`):

```
[[H + lambda_reg·I + lam·diag(H),  J_Gᵀ],   [delta]   [g  ]
 [J_G,                            -mu·I]] · [nu   ] = [R_G]
```

Знаковое соглашение, проверенное `pytests/jacobian_fd_test.py`:

```
[J; J_G] = −∂[R; R_G]/∂theta_full
```

`J` — якобиан **предсказаний**, а невязка `R = W(y − h)`, отсюда минус.

## Где что лежит

| Слой | Файл | Отвечает за |
|---|---|---|
| модель | `commom_utils/ode_system.py` | `ODESystem`, `SystemJacobian`, `SystemIntegrator` |
| чувствительности | `commom_utils/sensitivity.py` | `SensitivityTrajectory`, `group_by_grid_length` |
| коллокации | `commom_utils/collocation.py` | `RadauTables`, `CollocationSystemJacobian` |
| сборка задачи | `gauss_newton/problem.py` | `ShootRows`, `UnknownsLayout`, `MultipleShooting` |
| нормальные уравнения | `gauss_newton/normal_equations.py` | `NormalEquations`, накопление `H`/`g`, ковариация, CI |
| оптимизация | `gauss_newton/adaptive.py` | `gn_step`, `run_optimization_adaptive` |
