# -*- coding: utf-8 -*-
"""Слой оптимизации: шаг ГН и цикл с адаптивной регуляризацией.

Знает только про NormalEquations (H, g, J_G, R_G) из
gauss_newton/normal_equations.py — как они получены (через большую J или
накоплением по документу «MS and Orthogonal Collocations»), циклу безразлично.
Теория и эксперименты: adaptive_regularization.ipynb.

Единственный цикл оптимизации в проекте. Чем он отличается от прежней
схемы с ручным расписанием mu (удалена вместе с compute_delta_gn):
- lambda (демпфер Марквардта) адаптируется по gain ratio (схема Нильсена);
- mu (обратный вес невязок стыковки: шаг — это ГН для
  Phi_mu = ||R||^2 + (1/mu)||R_G||^2) стартует по кривизне
  ||J_G||_F^2 / tr(H) и ужесточается по Пауэллу — только когда невязка
  стыковки не падает сама И измерительная невязка застопорилась
  (гейт rss_stall_tol);
- шаг принимается по той же Phi_mu, для которой посчитан (rho > 0),
  а не по '<= 1.1*best';
- есть остановка (серия отказов / стагнация / pred ~ 0), поэтому n_iter —
  верхняя граница, а не обязательный бюджет.
"""
import numpy as np
from scipy.sparse import diags, hstack, vstack, eye as speye
from scipy.sparse.linalg import spsolve

from gauss_newton.normal_equations import (confidence_intervals,
                                           correlation_matrix,
                                           normal_equations_of)


def gn_step(ne, mu, lam, lambda_reg=0.0, lam_dual=None):
    """Шаг из mu-регуляризованной седловой системы + pred для gain ratio.

    Решается седловая система (mu-регуляризованная ККТ), плюс pred:
        [[H + D, J_G^T], [J_G, -mu I]] [delta; nu] = [g; R_G],
    D = lambda_reg*I + lam*diag(H). Седловая форма вместо исключённой
    (H + (1/mu) J_G^T J_G): при mu -> 0 исключённая теряет обусловленность
    как O(1/mu), а здесь mu входит линейно.

    pred = delta^T (g_eff + D delta) >= 0 — предсказанное моделью уменьшение
    Phi_mu (вывод в adaptive_regularization.ipynb, §3); pred <= 0 возможен
    только при численном сбое и трактуется циклом как отказ шага.

    lam_dual — оценка множителей Лагранжа для augmented-Lagrangian-режима
    (эксперименты: adaptive_regularization.ipynb, раздел про AL). При
    заданном lam_dual RHS блока ограничений сдвигается на -mu*lam_dual —
    это шаг ГН для L_A = ||R||^2 - 2 lam_dual^T R_G + (1/mu)||R_G||^2
    (у нас R_G = -G), в g_eff добавляется -J_G^T lam_dual, формула pred
    не меняется. Возврат расширяется до (delta, pred, nu), где nu —
    двойственное решение седловой системы: nu = lam_dual + (J_G delta -
    R_G)/mu, то есть готовое первопорядковое обновление множителей.
    При lam_dual=None возврат прежний (delta, pred) и числа побитово те же.
    """
    n = ne.H.shape[0]
    # floor на диагонали: столбец, к которому невязки локально нечувствительны,
    # иначе получил бы нулевое демпфирование и произвольно большой шаг
    D = lambda_reg * speye(n) + lam * diags(np.maximum(ne.H.diagonal(), 1e-10))
    nu = None
    if ne.n_cont > 0:
        K = vstack([hstack([ne.H + D, ne.J_G.T]),
                    hstack([ne.J_G, -mu * speye(ne.n_cont)])]).tocsr()
        rhs_G = ne.R_G if lam_dual is None else ne.R_G - mu * lam_dual
        sol = spsolve(K, np.concatenate([ne.g, rhs_G]))
        delta, nu = sol[:n], sol[n:]
        g_eff = ne.g + (1.0 / mu) * (ne.J_G.T @ ne.R_G)
        if lam_dual is not None:
            g_eff = g_eff - ne.J_G.T @ lam_dual
    else:
        g_eff = ne.g
        delta = spsolve((ne.H + D).tocsr(), ne.g)
    pred = float(delta @ g_eff + delta @ (D @ delta))
    if lam_dual is not None:
        return delta, pred, nu
    return delta, pred


def run_optimization_adaptive(problem, theta_full, n_iter=40,
                              lam0=1e-3, lambda_reg=0.0,
                              mu_rule='curvature', mu_dec=0.5,
                              viol_target=0.25, rss_stall_tol=0.99, kappa=0.1,
                              mu_min=1e-8, mu_max=1e8,
                              rho_accept=0.0, max_rejects=8,
                              track_covariance=True, verbose=False):
    """Итерации ГН с адаптивными lambda (Нильсен) и mu (кривизна + Пауэлл).

    problem — любая задача со `solve` (MultipleShooting, CollocationShooting)
    или с `normal_equations` (…Accum-классы: H и g копятся, J не строится).

    Параметры безразмерны, дефолты проверены на ЛВ/Аттракторе/реальных данных:
    - lam0: стартовый демпфер, дальше управляется gain ratio;
    - mu_rule: 'curvature' (рекомендуется) — старт ||J_G||_F^2/tr(H),
      ужесточение mu*mu_dec, когда ||R_G||^2 > viol_target*||R_G||^2_prev
      И измерительная невязка застопорилась (см. rss_stall_tol);
      'ratio' — mu = ||R_G||^2/(kappa*||R||^2) (хуже на реальных данных,
      см. ноутбук §8);
    - rss_stall_tol: гейт Пауэлла — mu ужимается только когда
      ||R||^2 > rss_stall_tol*||R||^2_prev, т.е. измерения уже выжаты.
      Без гейта mu утаптывается на ранних итерациях (rss падает на порядки,
      стыковка колеблется), ограничения начинают доминировать и решение
      запирается на консистентной траектории вдали от измерений — Attractor
      при N_shoot=5 сходится с гейтом и не сходится без него;
    - rho_accept: порог принятия шага по gain ratio;
    - track_covariance: считать доверительные интервалы theta по ходу
      (для plot_solution), стоит один splu на итерацию.

    Возвращает (theta_full, hist); hist по итерациям (включая отклонённые —
    там значения не меняются): theta, cost, mu, lam, r_meas, r_cont,
    ci_low, ci_high; плюс accepted (номера принятых) и n_solves.
    """
    theta_full = theta_full.copy()
    n_theta = problem.system.dims()[1]
    try:
        ne = normal_equations_of(problem, theta_full)
    except RuntimeError as exc:
        raise RuntimeError(
            "Интегратор не справился в НАЧАЛЬНОЙ точке theta0 (до первого шага "
            "ГН). Для коллокаций: увеличьте n_sub (мельче элементы), ослабьте "
            "newton_tol или поднимите newton_maxiter, либо выберите более "
            f"правдоподобное theta0. Исходная ошибка: {exc}") from exc

    def mu_ratio(ne_):
        return float(np.clip(ne_.cont_sq() / (kappa * max(ne_.rss, 1e-300)),
                             mu_min, mu_max))

    lam, nu_esc = lam0, 2.0
    if ne.n_cont == 0:
        mu = 1.0  # не используется: merit = ||R||^2
    elif mu_rule == 'curvature':
        mu = float(np.clip(ne.mu_curvature(), mu_min, mu_max))
    else:
        mu = mu_ratio(ne)
    prev_cont, prev_rss = ne.cont_sq(), ne.rss

    hist = dict(theta=[], cost=[], mu=[], lam=[], r_meas=[], r_cont=[],
                ci_low=[], ci_high=[], corr_cond=[], accepted=[], n_solves=1)

    def record():
        hist['theta'].append(theta_full.copy())
        hist['cost'].append(ne.cost())
        hist['mu'].append(mu)
        hist['lam'].append(lam)
        hist['r_meas'].append(ne.rss / max(ne.n_rows, 1))
        hist['r_cont'].append(ne.cont_sq() / max(ne.n_cont, 1))
        if track_covariance:
            cov, _, dof = ne.covariance_theta(n_theta)
            ci_low, ci_high = confidence_intervals(theta_full[:n_theta], cov, dof)
            hist['ci_low'].append(ci_low)
            hist['ci_high'].append(ci_high)
            # диагностика идентифицируемости из уже посчитанной ковариации:
            # рост corr_cond = параметры уходят в плоскую долину
            hist['corr_cond'].append(correlation_matrix(cov)[1])

    record()
    rejects = stalls = 0
    for it in range(n_iter):
        delta, pred = gn_step(ne, mu, lam, lambda_reg)
        ok = np.all(np.isfinite(delta)) and pred > 0
        if ok:
            theta_trial = theta_full + delta
            try:
                ne_trial = normal_equations_of(problem, theta_trial)
                hist['n_solves'] += 1
                phi0 = ne.merit(mu)
                phi1 = ne_trial.merit(mu)
                rho = (phi0 - phi1) / pred if np.isfinite(phi1) else -np.inf
            except RuntimeError:
                # интегратор не справился с пробной точкой (например, Ньютон
                # коллокаций не сошёлся) — это отказ шага, а не фатальная ошибка
                rho = -np.inf
        else:
            rho = -np.inf

        if np.isfinite(rho) and rho > rho_accept:
            theta_full, ne = theta_trial, ne_trial
            lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-12)
            nu_esc, rejects = 2.0, 0
            if ne.n_cont > 0:
                cont, rss = ne.cont_sq(), ne.rss
                if mu_rule == 'curvature':
                    # Пауэлл с гейтом: ужесточаем штраф, только когда стыковка
                    # сама не падает И измерения уже выжаты — пока rss падает,
                    # mu не трогаем (иначе преждевременное запирание, docstring)
                    if cont > viol_target * prev_cont and rss > rss_stall_tol * prev_rss:
                        mu = max(mu * mu_dec, mu_min)
                    prev_cont, prev_rss = cont, rss
                else:
                    mu = mu_ratio(ne)
            hist['accepted'].append(it)
            stalls = stalls + 1 if (phi0 - phi1) <= 1e-10 * max(phi0, 1e-300) else 0
            if verbose:
                print(f'Iter {it:3d} | accept rho={rho:6.3f} | '
                      f'cost {hist["cost"][-1]:.3e} -> {ne.cost():.3e} | '
                      f'mu {mu:.2e} | lam {lam:.2e}')
        else:
            lam = min(lam * nu_esc, 1e10)
            nu_esc *= 2.0
            rejects += 1
            if verbose:
                print(f'Iter {it:3d} | reject rho={rho:6.3f} | lam {lam:.2e}')

        record()
        if rejects >= max_rejects or stalls >= 2:
            break
        if ok and pred < 1e-12 * max(ne.merit(mu), 1e-300):
            break

    for key in ('theta', 'ci_low', 'ci_high'):
        hist[key] = np.array(hist[key]) if hist[key] else np.zeros((0, n_theta))
    return theta_full, hist
