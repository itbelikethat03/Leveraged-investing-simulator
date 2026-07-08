import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any, Optional
from numba import jit, prange
from arch import arch_model
from scipy import stats
import warnings

# ----------------------------------------------------------------------
# Data Processing Module
# ----------------------------------------------------------------------
def load_default_data():
    """Load default Fama-French data with proper error handling."""
    file_name = "F-F_Research_Data_Factors_daily.csv"
    if not os.path.exists(file_name):
        return None
    df = pd.read_csv(file_name, skiprows=4, encoding='utf-8-sig')
    df.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date'])
    for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Mkt-RF', 'RF']).reset_index(drop=True)
    return df

@st.cache_data
def load_default():
    return load_default_data()

@st.cache_data
def load_data_from_file(file):
    """Load user-uploaded data with flexible column handling."""
    df = pd.read_csv(file, skiprows=4)
    if 'date' not in df.columns:
        first_col = df.columns[0]
        df.rename(columns={first_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    return df

def prepare_returns_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert percentage returns to decimal form and calculate total market return.
    IMPORTANT: This function should only be called ONCE per dataset.
    """
    data = df.copy()
    for col in ['Mkt-RF', 'RF']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce') / 100.0
    data['Mkt_Total'] = data['Mkt-RF'] + data['RF']
    return data.dropna().reset_index(drop=True)

# ----------------------------------------------------------------------
# Financial Modeling Module
# ----------------------------------------------------------------------
def compute_periodic_risk_ratios(
    dates: pd.Series, returns: np.ndarray, rf: np.ndarray, freq: str = 'ME', periods_per_year: float = 12.0
) -> dict:
    """
    Computes Sharpe and Sortino from PERIODIC (default: monthly) compounded
    returns rather than daily returns -- the convention used by Portfolio
    Visualizer, testfol.io, and most published fund factsheets. A single
    day's return is too small for within-day compounding to matter, so a
    daily arithmetic-return ratio barely reflects volatility drag; a month
    of daily compounding does, especially at 2-3x leverage, which is why
    these measures (unlike daily-return versions) actually decline with
    leverage the way a real leveraged ETF's reported ratios do.
    """
    factors = np.maximum(1.0 + np.asarray(returns), 0.0)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    period_nav = pd.Series(factors, index=idx).resample(freq).prod()
    period_returns = period_nav - 1.0

    rf_factors = 1.0 + np.asarray(rf)
    period_rf = pd.Series(rf_factors, index=idx).resample(freq).prod() - 1.0

    excess = (period_returns - period_rf).to_numpy()
    if len(excess) < 2 or excess.std() == 0:
        return {'sharpe': np.nan, 'sortino': np.nan}

    sharpe = (excess.mean() * periods_per_year) / (excess.std() * np.sqrt(periods_per_year))

    # Sortino: same numerator, but the denominator only penalizes downside
    # (below-RF) periods -- squared over ALL periods (Sortino's original
    # convention), not just the count of losing ones.
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
    sortino = (excess.mean() * periods_per_year) / downside_dev if downside_dev > 0 else np.nan

    return {'sharpe': sharpe, 'sortino': sortino}

def compute_block_sharpe_sortino(
    daily_returns: np.ndarray, daily_rf: np.ndarray, block_size: int = 21, periods_per_year: float = 12.0
) -> dict:
    """
    Vectorized analogue of compute_periodic_risk_ratios for MONTE CARLO
    simulated paths, which have no calendar dates to resample by. Chunks each
    simulation's daily return series into fixed-size blocks (21 trading days
    = 1 month at 252 trading days/year) and compounds within each block,
    mirroring the monthly-aggregation convention used for the historical
    Sharpe/Sortino so simulated and historical figures are computed the same
    way and are directly comparable -- without this, simulated Sharpe would
    silently fall back to daily-return arithmetic and (as with the historical
    calc before it was fixed) fail to decline with leverage the way a real
    leveraged ETF's reported Sharpe does.

    daily_returns, daily_rf: shape (n_simulations, n_days).
    Returns dict of 'sharpe' and 'sortino', each shape (n_simulations,) --
    one ratio per simulated path, ready to be summarized (e.g. median) like
    every other per-path statistic in this app.
    """
    n_simulations, n_days = daily_returns.shape
    n_blocks = n_days // block_size
    if n_blocks < 2:
        nan_arr = np.full(n_simulations, np.nan)
        return {'sharpe': nan_arr, 'sortino': nan_arr}

    trimmed_days = n_blocks * block_size
    ret = daily_returns[:, :trimmed_days].reshape(n_simulations, n_blocks, block_size)
    rf = daily_rf[:, :trimmed_days].reshape(n_simulations, n_blocks, block_size)

    block_factors = np.maximum(1.0 + ret, 0.0)
    block_return = np.prod(block_factors, axis=2) - 1.0
    block_rf = np.prod(1.0 + rf, axis=2) - 1.0

    excess = block_return - block_rf
    mean_excess = excess.mean(axis=1)
    std_excess = excess.std(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = (mean_excess * periods_per_year) / (std_excess * np.sqrt(periods_per_year))
    sharpe = np.where(std_excess > 0, sharpe, np.nan)

    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2, axis=1)) * np.sqrt(periods_per_year)
    with np.errstate(divide='ignore', invalid='ignore'):
        sortino = (mean_excess * periods_per_year) / downside_dev
    sortino = np.where(downside_dev > 0, sortino, np.nan)

    return {'sharpe': sharpe, 'sortino': sortino}

def simulate_leveraged_etf(
    df: pd.DataFrame,
    leverage: float = 2.0,
    spread_annual: float = 0.004,
    expense_ratio_annual: float = 0.0095,
    trading_days: int = 252
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    data = df.copy()
    spread_daily = spread_annual / trading_days
    expense_daily = expense_ratio_annual / trading_days
    
    data['Gross_Lev'] = leverage * data['Mkt_Total']
    data['Financing_Cost'] = (leverage - 1) * (data['RF'] + spread_daily)
    data['Expense_Cost'] = expense_daily
    data['Daily_Cost'] = data['Financing_Cost'] + data['Expense_Cost']
    data['Net_Lev_Total'] = data['Gross_Lev'] - data['Daily_Cost']
    
    data['Lev_NAV'] = (1 + data['Net_Lev_Total']).cumprod()
    data['Mkt_NAV'] = (1 + data['Mkt_Total']).cumprod()
    
    def calc_stats(nav_series, returns_series, rf_series, name):
        total_return = nav_series.iloc[-1] - 1
        ann_return = (1 + total_return) ** (trading_days / len(returns_series)) - 1
        ann_vol = returns_series.std() * np.sqrt(trading_days)
        risk_ratios = compute_periodic_risk_ratios(data['date'], returns_series.to_numpy(), rf_series.to_numpy())
        sharpe = risk_ratios['sharpe']
        sortino = risk_ratios['sortino']
        peak = nav_series.cummax()
        dd = (nav_series / peak - 1)
        max_dd = dd.min()
        sorted_returns = np.sort(returns_series)
        var_95 = np.percentile(sorted_returns, 5)
        cvar_95 = sorted_returns[sorted_returns <= var_95].mean()

        return {
            'name': name, 'ann_return': ann_return, 'ann_vol': ann_vol, 'sharpe': sharpe, 'sortino': sortino,
            'max_dd': max_dd, 'cvar_95': cvar_95, 'final_nav': nav_series.iloc[-1],
            'total_return': total_return, 'start_date': data['date'].min(),
            'end_date': data['date'].max(), 'num_days': len(data)
        }
    
    leveraged_stats = calc_stats(data['Lev_NAV'], data['Net_Lev_Total'], data['RF'], f'{leverage}x Leveraged')
    unleveraged_stats = calc_stats(data['Mkt_NAV'], data['Mkt_Total'], data['RF'], '1x Market')
    return data, leveraged_stats, unleveraged_stats

# ----------------------------------------------------------------------
# Monte Carlo Simulation Engine
# ----------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def simulate_paths_jit(
    sampled_pairs: np.ndarray, leverage: float, spread_daily: float, 
    expense_daily: float, n_simulations: int, n_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    lev_paths = np.zeros((n_simulations, n_days + 1))
    mkt_paths = np.zeros((n_simulations, n_days + 1))
    lev_paths[:, 0] = 1.0
    mkt_paths[:, 0] = 1.0
    
    for sim in prange(n_simulations):
        for day in range(n_days):
            mkt_rf = sampled_pairs[sim, day, 0]
            rf = sampled_pairs[sim, day, 1]
            mkt_total = mkt_rf + rf
            daily_cost = (leverage - 1) * (rf + spread_daily) + expense_daily
            lev_daily_ret = leverage * mkt_total - daily_cost
            lev_paths[sim, day + 1] = max(0.0, lev_paths[sim, day] * (1 + lev_daily_ret))
            mkt_paths[sim, day + 1] = max(0.0, mkt_paths[sim, day] * (1 + mkt_total))
    return lev_paths, mkt_paths

@jit(nopython=True, parallel=True)
def simulate_garch_returns_jit(
    z_samples: np.ndarray, mu: float, omega: float, alpha: float, beta: float,
    init_sigma2: float, n_simulations: int, n_days: int,
    z_clip: float = 5.0, sigma2_cap: float = 1e18
) -> np.ndarray:
    """
    Generates the GARCH(1,1) Mkt-RF return path only (leverage-independent).
    The sigma2 recursion depends only on past shocks, never on leverage, so this
    can be computed once and reused for any leverage via simulate_paths_jit.

    z_clip / sigma2_cap are numerical stability safeguards, not a fudge to hit a
    target statistic. When the fitted (alpha, beta) combined with the actual
    innovation kurtosis violates GARCH(1,1)'s finite-fourth-moment condition
    (kurt_z*alpha^2 + 2*alpha*beta + beta^2 >= 1 -- common with "Resampled"
    fat-tailed innovations), simulating thousands of independent 10-60 year
    paths lets the (sigma2, z) pairing wander into combinations that never
    co-occurred in the real historical sample, and sigma2's own recursion can
    compound without bound. z_clip caps the resampled/parametric shock at a
    generous +/-5 standard deviations (removes well under 0.1% of real
    historical mass); sigma2_cap is a much looser sanity ceiling on the
    variance recursion itself. Together they prevent runaway compounding
    without materially altering short/medium-horizon dynamics.
    """
    mkt_rf_paths = np.zeros((n_simulations, n_days))
    for sim in prange(n_simulations):
        sigma2 = init_sigma2
        for day in range(n_days):
            z = z_samples[sim, day]
            if z > z_clip:
                z = z_clip
            elif z < -z_clip:
                z = -z_clip
            eps = np.sqrt(sigma2) * z
            mkt_rf_paths[sim, day] = mu + eps
            sigma2 = omega + alpha * (eps ** 2) + beta * sigma2
            if sigma2 > sigma2_cap:
                sigma2 = sigma2_cap
    return mkt_rf_paths

@jit(nopython=True, parallel=True)
def simulate_regime_returns_jit(
    rand_regime: np.ndarray, rand_shock: np.ndarray, P: np.ndarray,
    mu: np.ndarray, sigma: np.ndarray, init_regime: np.ndarray,
    n_simulations: int, n_days: int
) -> np.ndarray:
    """
    Generates the Markov regime-switching Mkt-RF return path only
    (leverage-independent; regimes: 0 = Bull, 1 = Neutral, 2 = Crisis).
    """
    mkt_rf_paths = np.zeros((n_simulations, n_days))
    for sim in prange(n_simulations):
        regime = init_regime[sim]
        for day in range(n_days):
            r_mu = mu[regime]
            r_sigma = sigma[regime]
            z = rand_shock[sim, day]
            mkt_rf_paths[sim, day] = r_mu + r_sigma * z

            u = rand_regime[sim, day]
            if u < P[regime, 0]:
                regime = 0
            elif u < P[regime, 0] + P[regime, 1]:
                regime = 1
            else:
                regime = 2
    return mkt_rf_paths

def fit_garch_model(returns_series: pd.Series) -> dict:
    returns_pct = returns_series * 100.0

    model = arch_model(returns_pct, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp='off')

    omega = res.params['omega'] / 10000.0

    alpha_key = [k for k in res.params.keys() if k.startswith('alpha')][0]
    beta_key = [k for k in res.params.keys() if k.startswith('beta')][0]
    alpha = res.params[alpha_key]
    beta = res.params[beta_key]

    mu = res.params['mu'] / 100.0

    std_resid = (res.resid / res.conditional_volatility).dropna().values

    # Seed forward simulation with the model's actual end-of-sample conditional
    # variance, not the long-run unconditional variance, so the near-term vol
    # regime at the end of history carries into the projection.
    last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2) / 10000.0

    # Sanity ceiling for the simulation's sigma2 recursion: a generous multiple
    # of the worst variance actually observed historically, not a tight leash.
    max_hist_sigma2 = float(res.conditional_volatility.max() ** 2) / 10000.0

    return {
        'mu': mu, 'omega': omega, 'alpha': alpha, 'beta': beta, 'std_resid': std_resid,
        'last_sigma2': last_sigma2, 'max_hist_sigma2': max_hist_sigma2,
    }

def garch_fourth_moment_stability(alpha: float, beta: float, kurt_z: float) -> Tuple[float, bool]:
    """
    GARCH(1,1)'s unconditional 4th moment (hence long-run kurtosis) is finite
    only if kurt_z*alpha^2 + 2*alpha*beta + beta^2 < 1, where kurt_z is the
    actual (non-excess) kurtosis of the innovation distribution used in
    SIMULATION -- which depends on which of Gaussian / Student-t / Resampled
    the user picked, not just the fitted (alpha, beta). Violating this means
    simulated tail risk has no natural ceiling and grows with simulation
    length/count rather than converging.
    """
    stability = kurt_z * alpha ** 2 + 2 * alpha * beta + beta ** 2
    return stability, stability < 1.0

def estimate_regime_heuristic(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn is required for regime estimation.")

    window = 21
    s = pd.Series(returns)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    
    valid_idx = ~np.isnan(roll_mean)
    features = np.column_stack([roll_mean[valid_idx], roll_std[valid_idx]])
    
    if len(features) < 10:
        raise ValueError("Not enough data for regime estimation")
        
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
    gmm.fit(features)
    labels = gmm.predict(features)

    # Label regimes by the RAW daily-return mean per cluster (what actually
    # gets plugged into mu[regime] for simulation), not by the smoothed
    # rolling-window feature used only to help GMM find cluster boundaries.
    # These two disagree in practice: a cluster can sit within the worst
    # *trailing* 21-day trend (correctly flagged by the rolling feature) while
    # its individual days average a POSITIVE return, because violent rebound
    # rallies cluster together with the crashes that precede them. Sorting by
    # the rolling feature (an earlier attempt at this fix) still picked a
    # rebound-heavy cluster as "Crisis"; sorting by actual per-day returns
    # guarantees Crisis is genuinely the worst-return regime.
    valid_returns = returns[valid_idx]
    raw_cluster_means = [np.mean(valid_returns[labels == i]) for i in range(3)]
    sorted_by_mean = np.argsort(raw_cluster_means)  # ascending: worst -> best

    bull_idx = sorted_by_mean[-1]     # highest mean return
    crisis_idx = sorted_by_mean[0]    # lowest (most negative) mean return
    neutral_idx = sorted_by_mean[1]   # remaining, middle-mean cluster
    
    label_map = {bull_idx: 0, neutral_idx: 1, crisis_idx: 2}
    mapped_labels = np.array([label_map[l] for l in labels])
    
    full_labels = np.zeros(len(returns), dtype=int)
    full_labels[:window-1] = mapped_labels[0]
    full_labels[window-1:] = mapped_labels
    
    mu = np.zeros(3)
    sigma = np.zeros(3)
    for i in range(3):
        mask = full_labels == i
        if np.sum(mask) > 1:
            mu[i] = np.mean(returns[mask])
            sigma[i] = np.std(returns[mask])
        else:
            mu[i] = 0.0
            sigma[i] = 0.01
            
    P = np.zeros((3, 3))
    for t in range(len(full_labels) - 1):
        P[full_labels[t], full_labels[t+1]] += 1
        
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = P / row_sums
    
    eigvals, eigvecs = np.linalg.eig(P.T)
    stat_dist = np.abs(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    stat_dist = stat_dist / stat_dist.sum()
    
    stat_dist = np.clip(stat_dist, 1e-5, 1.0)
    stat_dist /= stat_dist.sum()
    
    return mu, sigma, P, stat_dist

def block_bootstrap_indices(
    n_observations: int, n_simulations: int, n_days: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Vectorized moving-block bootstrap index generator. Returns an
    (n_simulations, n_days) array of indices into a length-n_observations series.
    """
    n_possible_blocks = n_observations - block_size + 1
    if n_possible_blocks < 1:
        raise ValueError(
            f"block_size ({block_size}) exceeds available history ({n_observations} rows)"
        )
    n_blocks_needed = int(np.ceil(n_days / block_size))
    starts = rng.integers(0, n_possible_blocks, size=(n_simulations, n_blocks_needed))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_simulations, -1)[:, :n_days]
    return idx

@st.cache_data(show_spinner=False)
def generate_scenario_pairs(
    df_historical: pd.DataFrame, n_simulations: int, n_years: int, trading_days: int,
    block_size: int, model_type: str, garch_dist: str, random_seed: Optional[int]
) -> Tuple[np.ndarray, str, Optional[int], Optional[str]]:
    """
    Draws the (Mkt-RF, RF) scenario pairs for one Monte Carlo model. This is the
    expensive, leverage-INDEPENDENT part of the simulation (GARCH/regime dynamics,
    bootstrap resampling). Leverage and costs are only applied afterwards via
    simulate_paths_jit, so every caller -- the single-leverage MC panel and the
    Optimal Leverage Explorer's grid sweep -- can share one cached `pairs` array
    across many leverage values instead of redrawing scenarios per leverage.
    Cached by Streamlit so identical (data, model, seed) calls are instant.

    Returns (pairs, method_text, block_size_used, warning_msg) -- warning_msg is
    non-None only when the fitted GARCH model is numerically unstable for the
    chosen innovation distribution (see fit_garch_model).
    """
    pairs_hist = df_historical[['Mkt-RF', 'RF']].to_numpy()
    n_days = n_years * trading_days
    n_observations = len(pairs_hist)
    rng = np.random.default_rng(random_seed)
    # Risk-free rates persist in multi-year regimes (ZIRP, hiking cycles); bootstrap
    # them in ~1-year blocks (not i.i.d. day-by-day) so financing-cost drag
    # realistically stays "high" or "low" for extended stretches.
    rf_block_size = min(trading_days, n_observations)

    method_text = ""
    block_size_used = None
    warning_msg = None

    if model_type == "garch":
        garch_fit = fit_garch_model(df_historical['Mkt-RF'])
        mu, omega, alpha, beta = garch_fit['mu'], garch_fit['omega'], garch_fit['alpha'], garch_fit['beta']
        std_resid = garch_fit['std_resid']
        last_sigma2 = garch_fit['last_sigma2']
        # Seed with the model's actual end-of-sample conditional variance so the
        # simulation carries forward the current vol regime instead of jumping
        # straight to the long-run average.
        init_sigma2 = last_sigma2 if np.isfinite(last_sigma2) and last_sigma2 > 0 else omega

        if garch_dist == "Resampled (Fat Tails)":
            idx = rng.integers(0, len(std_resid), size=(n_simulations, n_days))
            z_samples = std_resid[idx]
            method_text = "GARCH(1,1) - Resampled"
            # Empirical kurtosis of the actual pool being resampled.
            kurt_z = float(pd.Series(std_resid).kurtosis()) + 3.0
        elif garch_dist == "Gaussian":
            z_samples = rng.normal(0, 1, size=(n_simulations, n_days))
            method_text = "GARCH(1,1) - Gaussian"
            kurt_z = 3.0
        else:
            # Fix loc=0 (residuals are demeaned) and rescale draws to exactly
            # unit variance so the GARCH recursion's unit-variance-innovation
            # assumption holds regardless of MLE fit noise in df/scale.
            df_t, _, scale_t = stats.t.fit(std_resid, floc=0.0)
            raw = stats.t.rvs(df_t, loc=0.0, scale=scale_t, size=(n_simulations, n_days), random_state=rng)
            z_samples = raw / np.sqrt(df_t / (df_t - 2.0))
            method_text = "GARCH(1,1) - Student-t"
            # Theoretical kurtosis of a standardized Student-t with df_t degrees of freedom.
            kurt_z = 3.0 + 6.0 / (df_t - 4.0) if df_t > 4 else float('inf')

        stability, is_stable = garch_fourth_moment_stability(alpha, beta, kurt_z)
        if not is_stable:
            warning_msg = (
                f"GARCH({garch_dist}) is outside its theoretically stable region for this fit "
                f"(4th-moment stability statistic = {stability:.3f}, must be < 1 for finite long-run kurtosis). "
                "Without a safeguard, simulated tail risk would grow without bound as more/longer paths are "
                "simulated. A numerical ceiling (shock size capped at 5 std. devs., variance capped at 10x the "
                "worst historically observed level) is active to prevent runaway compounding, but tail-risk "
                "figures from this model+distribution combination should still be read with more caution."
            )

        rf_hist = df_historical['RF'].to_numpy()
        rf_idx = block_bootstrap_indices(len(rf_hist), n_simulations, n_days, rf_block_size, rng)
        rf_samples = rf_hist[rf_idx]

        mkt_rf_paths = simulate_garch_returns_jit(
            z_samples, mu, omega, alpha, beta, init_sigma2, n_simulations, n_days,
            z_clip=5.0, sigma2_cap=10.0 * garch_fit['max_hist_sigma2']
        )
        pairs = np.dstack([mkt_rf_paths, rf_samples])

    elif model_type == "regime_switching":
        try:
            mu, sigma, P, stat_dist = estimate_regime_heuristic(df_historical['Mkt-RF'].to_numpy())
            method_text = "Markov Regime-Switching (Heuristic GMM)"
        except Exception:
            mu = np.array([0.0004, 0.0002, -0.0005])
            sigma = np.array([0.007, 0.012, 0.025])
            P = np.array([[0.98, 0.015, 0.005],
                          [0.05, 0.90, 0.05],
                          [0.05, 0.10, 0.85]])
            stat_dist = np.array([0.6, 0.3, 0.1])
            method_text = "Markov Regime-Switching (Calibrated Fallback)"

        rand_regime = rng.uniform(0, 1, size=(n_simulations, n_days))
        rand_shock = rng.normal(0, 1, size=(n_simulations, n_days))
        init_regime = rng.choice(3, size=n_simulations, p=stat_dist)

        rf_hist = df_historical['RF'].to_numpy()
        rf_idx = block_bootstrap_indices(len(rf_hist), n_simulations, n_days, rf_block_size, rng)
        rf_samples = rf_hist[rf_idx]

        mkt_rf_paths = simulate_regime_returns_jit(
            rand_regime, rand_shock, P, mu, sigma, init_regime, n_simulations, n_days
        )
        pairs = np.dstack([mkt_rf_paths, rf_samples])

    else:
        if model_type == "block_bootstrap":
            if n_observations - block_size + 1 < 1:
                raise ValueError(
                    f"Block size ({block_size}) exceeds available history "
                    f"({n_observations} rows). Reduce block size or widen the date filter."
                )
            idx = block_bootstrap_indices(n_observations, n_simulations, n_days, block_size, rng)
            pairs = pairs_hist[idx]
            method_text = f"Block Bootstrap (block size: {block_size} days)"
            block_size_used = block_size
        else:
            idx = rng.integers(0, n_observations, size=(n_simulations, n_days))
            pairs = pairs_hist[idx]
            method_text = "IID Bootstrap"

    return pairs, method_text, block_size_used, warning_msg

# ----------------------------------------------------------------------
# Dominance Analysis Diagnostic Layer (Post-Processing)
# ----------------------------------------------------------------------
def compute_dominance_analysis(lev_paths: np.ndarray, mkt_paths: np.ndarray, trading_days: int = 252) -> dict:
    n_simulations, n_days = mkt_paths.shape
    
    final_mkt_return = mkt_paths[:, -1]
    
    p25 = np.percentile(final_mkt_return, 25)
    p75 = np.percentile(final_mkt_return, 75)
    
    regimes = {
        'crash': final_mkt_return <= p25,
        'neutral': (final_mkt_return > p25) & (final_mkt_return <= p75),
        'bull': final_mkt_return > p75
    }
    
    conditional = {}
    for regime_name, mask in regimes.items():
        if np.sum(mask) == 0:
            conditional[regime_name] = {
                'prob_lev_wins': 0.0, 'mean_outperformance': 0.0,
                'prob_ruin': 0.0, 'median_ratio': 0.0
            }
            continue
            
        lev_final = lev_paths[mask, -1]
        mkt_final = mkt_paths[mask, -1]
        
        prob_lev_wins = np.mean(lev_final > mkt_final)
        mean_outperformance = np.mean(lev_final - mkt_final)
        prob_ruin = np.mean(lev_final < 0.1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(mkt_final > 1e-8, lev_final / mkt_final, np.nan)
            median_ratio = np.nanmedian(ratios)
            if np.isnan(median_ratio):
                median_ratio = 0.0
            
        conditional[regime_name] = {
            'prob_lev_wins': prob_lev_wins,
            'mean_outperformance': mean_outperformance,
            'prob_ruin': prob_ruin,
            'median_ratio': median_ratio
        }
        
    years = [1, 5, 10, 20, 30]
    time_weighted_probs = {}
    for y in years:
        t_days = y * trading_days
        if t_days >= n_days:
            t_days = n_days - 1
        prob_ahead = np.mean(lev_paths[:, t_days] > mkt_paths[:, t_days])
        time_weighted_probs[f"{y}y"] = prob_ahead
        
    lead_matrix = lev_paths > mkt_paths
    avg_time_in_lead = np.mean(lead_matrix[:, 1:]) 
    
    time_weighted = {
        'horizons': time_weighted_probs,
        'avg_time_in_lead': avg_time_in_lead
    }
    
    fraction_ahead = np.mean(lead_matrix[:, 1:], axis=1)
    median_fraction_ahead = np.median(fraction_ahead)
    
    sustained = {}
    thresholds = [0.50, 0.60, 0.70, 0.80]
    for x in thresholds:
        prob_sustained = np.mean(fraction_ahead >= x)
        sustained[f"{int(x*100)}%"] = prob_sustained
        
    sustained['median_fraction_ahead'] = median_fraction_ahead
    
    return {
        'conditional': conditional,
        'time_weighted': time_weighted,
        'sustained': sustained
    }

# ----------------------------------------------------------------------
# Main Monte Carlo Orchestrator
# ----------------------------------------------------------------------
def run_monte_carlo(
    df_historical: pd.DataFrame, leverage: float, spread_annual: float,
    expense_ratio_annual: float, n_simulations: int = 1000, n_years: int = 40,
    trading_days: int = 252, block_size: int = 20,
    model_type: str = "block_bootstrap", garch_dist: str = "Resampled (Fat Tails)",
    random_seed: Optional[int] = None
):
    n_days = n_years * trading_days
    spread_daily = spread_annual / trading_days
    expense_daily = expense_ratio_annual / trading_days

    with st.spinner("Preparing market scenario paths..."):
        try:
            pairs, method_text, block_size_used, garch_warning = generate_scenario_pairs(
                df_historical, n_simulations, n_years, trading_days,
                block_size, model_type, garch_dist, random_seed
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    with st.spinner("Applying leverage & compounding paths..."):
        lev_paths, mkt_paths = simulate_paths_jit(
            pairs, leverage, spread_daily, expense_daily, n_simulations, n_days
        )

    rf_samples = pairs[:, :, 1]

    def compute_stats(paths):
        percentiles = np.percentile(paths, [1, 5, 50, 95, 99], axis=0)
        final_navs = paths[:, -1]
        peaks = np.maximum.accumulate(paths, axis=1)
        drawdowns = paths / peaks - 1
        max_drawdowns = drawdowns.min(axis=1)

        # Calculate annualized returns per path for CVaR
        ann_returns = final_navs ** (trading_days / n_days) - 1.0

        n_tail = max(1, int(0.05 * len(final_navs)))
        sorted_ann_rets = np.sort(ann_returns)
        sorted_max_dds = np.sort(max_drawdowns)
        sorted_final_navs = np.sort(final_navs)

        cvar_terminal = np.mean(sorted_final_navs[:n_tail])
        cvar_ann_ret = np.mean(sorted_ann_rets[:n_tail])
        cvar_max_dd = np.mean(sorted_max_dds[:n_tail])

        # Sharpe/Sortino from block-compounded (~monthly) returns per simulated
        # path -- same convention as the historical calc, so simulated and
        # historical figures are comparable and both correctly decline with
        # leverage rather than silently using raw daily returns.
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_ret = np.where(paths[:, :-1] > 0, paths[:, 1:] / paths[:, :-1] - 1.0, -1.0)
        risk_ratios = compute_block_sharpe_sortino(daily_ret, rf_samples)
        median_sharpe = float(np.nanmedian(risk_ratios['sharpe']))
        median_sortino = float(np.nanmedian(risk_ratios['sortino']))

        # The percentile bands above are cross-sectional (a different subset of
        # paths at every day), so no single line in the fan chart is an actual
        # simulated trajectory. Surface one real path (closest to the median
        # terminal NAV) so users can see a genuine compounding experience.
        median_path_idx = np.argsort(final_navs)[len(final_navs) // 2]
        representative_path = paths[median_path_idx]

        return {
            'paths': paths, 'final_navs': final_navs, 'percentiles': percentiles,
            'median_final': np.median(final_navs), 'p1_final': np.percentile(final_navs, 1),
            'p5_final': np.percentile(final_navs, 5), 'p95_final': np.percentile(final_navs, 95),
            'p99_final': np.percentile(final_navs, 99), 'prob_loss': np.mean(final_navs < 1.0),
            'prob_double': np.mean(final_navs >= 2.0), 'prob_ruin': np.mean(final_navs < 0.1),
            'prob_70pct_dd': np.mean(max_drawdowns < -0.7), 'mean_final': np.mean(final_navs),
            'std_final': np.std(final_navs), 'max_dd_dist': max_drawdowns,
            'cvar_terminal': cvar_terminal,
            'cvar_ann_ret': cvar_ann_ret,
            'cvar_max_dd': cvar_max_dd,
            'median_sharpe': median_sharpe,
            'median_sortino': median_sortino,
            'representative_path': representative_path
        }
    
    lev_stats = compute_stats(lev_paths)
    mkt_stats = compute_stats(mkt_paths)
    
    prob_lev_beats_mkt = np.mean(lev_paths[:, -1] > mkt_paths[:, -1])
    
    dominance = compute_dominance_analysis(lev_paths, mkt_paths, trading_days)
    
    raw_excess_returns = df_historical['Mkt-RF'].to_numpy()
    underlying_kelly = compute_kelly_metrics(raw_excess_returns, trading_days=trading_days)
    
    return {
        'lev': lev_stats,
        'mkt': mkt_stats,
        'kelly': {'underlying': underlying_kelly},
        'method': method_text,
        'block_size': block_size_used,
        'garch_warning': garch_warning,
        'prob_lev_beats_mkt': prob_lev_beats_mkt,
        'dominance_analysis': dominance
    }

# ----------------------------------------------------------------------
# ✅ CORRECTED KELLY CRITERION MODULE
# ----------------------------------------------------------------------
def compute_kelly_metrics(R: np.ndarray, trading_days: int = 252) -> dict:
    mu = np.mean(R)
    sigma2 = np.var(R)

    if sigma2 < 1e-12:
        f_kelly = 0.0
    else:
        f_kelly = mu / sigma2

    f_kelly = np.clip(f_kelly, -1.0, 10.0)

    f_half = 0.5 * f_kelly
    f_quarter = 0.25 * f_kelly

    sorted_R = np.sort(R)
    var_5 = np.percentile(sorted_R, 5)
    cvar_5 = sorted_R[sorted_R <= var_5].mean() if np.any(sorted_R <= var_5) else var_5

    # cvar_5 is a DAILY tail loss (typically 1-3%), so penalizing by
    # (1 - |cvar_5|) barely moves f_kelly regardless of how fat the tail is.
    # Annualize the tail measure so the penalty is commensurate with f_kelly's
    # own (annualization-invariant) scale.
    cvar_5_annualized = cvar_5 * np.sqrt(trading_days)
    f_dd = f_kelly * (1.0 - min(1.0, abs(cvar_5_annualized)))
    f_dd = np.clip(f_dd, -1.0, 10.0)
    
    f_grid = np.linspace(-1.0, 2.0, 300)
    log_growth = np.zeros_like(f_grid)
    
    kelly_progress = st.progress(0)
    kelly_status = st.empty()
    
    for i, f in enumerate(f_grid):
        growth_factors = 1.0 + f * R
        log_growth[i] = np.mean(np.log(np.maximum(growth_factors, 1e-15)))
        
        if i % 30 == 0 or i == len(f_grid) - 1:
            kelly_progress.progress((i + 1) / len(f_grid))
            kelly_status.text(f"Optimizing Log-Optimal Kelly... {int((i + 1) / len(f_grid) * 100)}%")
            
    kelly_progress.empty()
    kelly_status.empty()
    
    f_log_opt = f_grid[np.argmax(log_growth)]
    
    return {
        'full_kelly': f_kelly, 'half_kelly': f_half, 'quarter_kelly': f_quarter,
        'dd_adjusted_kelly': f_dd, 'log_optimal_kelly': f_log_opt
    }

# ----------------------------------------------------------------------
# Optimal Leverage Explorer Module
# ----------------------------------------------------------------------
def get_leverage_grid(min_leverage: float = 1.0, max_leverage: float = 3.0, step: float = 0.1) -> np.ndarray:
    n_steps = int(round((max_leverage - min_leverage) / step)) + 1
    return np.round(np.linspace(min_leverage, max_leverage, n_steps), 10)

@st.cache_data(show_spinner=False)
def compute_leverage_grid_historical(
    df_filtered: pd.DataFrame, leverage_grid: np.ndarray, spread_annual: float,
    expense_ratio_annual: float, trading_days: int = 252
) -> pd.DataFrame:
    """
    Sweeps leverage over `leverage_grid` on the SAME historical daily return
    series for every leverage level, applying the identical daily-compounding +
    cost formula as simulate_leveraged_etf, and returns one metrics row per level.
    """
    mkt_total = df_filtered['Mkt_Total'].to_numpy()
    rf = df_filtered['RF'].to_numpy()
    n = len(mkt_total)
    spread_daily = spread_annual / trading_days
    expense_daily = expense_ratio_annual / trading_days

    rows = []
    for lev in leverage_grid:
        daily_cost = (lev - 1.0) * (rf + spread_daily) + expense_daily
        lev_ret = lev * mkt_total - daily_cost
        factors = np.maximum(1.0 + lev_ret, 0.0)
        nav = np.cumprod(factors)

        final_nav = nav[-1]
        total_return = final_nav - 1.0
        cagr = final_nav ** (trading_days / n) - 1.0
        ann_vol = lev_ret.std() * np.sqrt(trading_days)

        risk_ratios = compute_periodic_risk_ratios(df_filtered['date'], lev_ret, rf)
        sharpe = risk_ratios['sharpe']
        sortino = risk_ratios['sortino']

        peak = np.maximum.accumulate(nav)
        dd = nav / peak - 1.0
        max_dd = dd.min()

        sorted_ret = np.sort(lev_ret)
        var_95 = np.percentile(sorted_ret, 5)
        cvar_95 = sorted_ret[sorted_ret <= var_95].mean()

        log_growth = np.log(np.maximum(factors, 1e-12))
        mean_log_growth = log_growth.mean()
        geo_mean_daily = np.exp(mean_log_growth) - 1.0

        ulcer_index = np.sqrt(np.mean((dd * 100.0) ** 2))
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

        rows.append({
            'leverage': float(lev), 'cagr': cagr, 'ann_vol': ann_vol, 'sharpe': sharpe, 'sortino': sortino,
            'max_dd': max_dd, 'cvar_95': cvar_95, 'total_return': total_return,
            'geo_mean_daily': geo_mean_daily, 'ulcer_index': ulcer_index,
            'calmar': calmar, 'mean_log_growth': mean_log_growth
        })

    return pd.DataFrame(rows)

def find_historical_optimal_leverage(df: pd.DataFrame) -> dict:
    calmar = df['calmar'].replace([np.inf, -np.inf], np.nan)
    sortino = df['sortino'].replace([np.inf, -np.inf], np.nan)
    return {
        'max_cagr': float(df.loc[df['cagr'].idxmax(), 'leverage']),
        'max_sharpe': float(df.loc[df['sharpe'].idxmax(), 'leverage']),
        'max_sortino': float(df.loc[sortino.idxmax(), 'leverage']) if sortino.notna().any() else float('nan'),
        'max_calmar': float(df.loc[calmar.idxmax(), 'leverage']) if calmar.notna().any() else float('nan'),
        'log_optimal': float(df.loc[df['mean_log_growth'].idxmax(), 'leverage']),
    }

def compute_leverage_grid_mc(
    pairs: np.ndarray, leverage_grid: np.ndarray, spread_annual: float,
    expense_ratio_annual: float, trading_days: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Sweeps leverage using the SAME simulated (Mkt-RF, RF) scenario paths for
    every leverage value, so "probability of outperforming the next-lower
    leverage" is a genuine paired, path-by-path comparison rather than an
    independent re-draw. Only scalar/vector summary stats are retained per
    leverage for the main grid table (never a full NAV-path matrix per
    leverage), keeping memory bounded regardless of grid size.

    Also returns the (n_simulations, n_grid) TERMINAL nav matrix -- one
    column per leverage_grid entry -- captured as a free byproduct of this
    same per-leverage compounding pass. Downstream leverage-vs-leverage
    analyses (Leverage Comparison Analysis) derive every pairwise/ranking/
    regret statistic from this one stored matrix instead of re-simulating
    or re-compounding anything.
    """
    n_simulations, n_days, _ = pairs.shape
    spread_daily = spread_annual / trading_days
    expense_daily = expense_ratio_annual / trading_days

    mkt_total = pairs[:, :, 0] + pairs[:, :, 1]
    rf = pairs[:, :, 1]

    mkt_factors = np.maximum(1.0 + mkt_total, 0.0)
    mkt_nav = np.cumprod(mkt_factors, axis=1)
    mkt_final = mkt_nav[:, -1]

    n_grid = len(leverage_grid)
    terminal_navs = np.empty((n_simulations, n_grid))

    rows = []
    prev_final = None
    for i, lev in enumerate(leverage_grid):
        daily_cost = (lev - 1.0) * (rf + spread_daily) + expense_daily
        lev_ret = lev * mkt_total - daily_cost
        factors = np.maximum(1.0 + lev_ret, 0.0)
        nav = np.cumprod(factors, axis=1)
        final_navs = nav[:, -1]
        terminal_navs[:, i] = final_navs
        peak = np.maximum.accumulate(nav, axis=1)
        dd = nav / peak - 1.0
        max_dd = dd.min(axis=1)

        cagr = final_navs ** (trading_days / n_days) - 1.0
        sorted_final = np.sort(final_navs)
        n_tail = max(1, int(0.05 * len(final_navs)))

        # Sharpe/Sortino from block-compounded (~monthly) returns per simulated
        # path, matching the historical calc's convention (see
        # compute_block_sharpe_sortino) so simulated and historical figures
        # are comparable and both correctly decline with leverage.
        risk_ratios = compute_block_sharpe_sortino(factors - 1.0, rf)
        median_sharpe = float(np.nanmedian(risk_ratios['sharpe']))
        median_sortino = float(np.nanmedian(risk_ratios['sortino']))

        rows.append({
            'leverage': float(lev),
            'median_final_nav': float(np.median(final_navs)),
            'mean_final_nav': float(np.mean(final_navs)),
            'median_cagr': float(np.median(cagr)),
            'prob_loss': float(np.mean(final_navs < 1.0)),
            'prob_ruin': float(np.mean(final_navs < 0.1)),
            'prob_beat_mkt': float(np.mean(final_navs > mkt_final)),
            'prob_beat_prev_leverage': float(np.mean(final_navs > prev_final)) if prev_final is not None else np.nan,
            'cvar_terminal': float(sorted_final[:n_tail].mean()),
            'median_max_dd': float(np.median(max_dd)),
            'median_sharpe': median_sharpe,
            'median_sortino': median_sortino,
        })
        prev_final = final_navs

    return pd.DataFrame(rows), terminal_navs

def compute_leverage_comparison_tables(
    terminal_navs: np.ndarray, leverage_grid: np.ndarray, regret_threshold_pct: float = 10.0
) -> dict:
    """
    Derives every leverage-vs-leverage comparison from the ALREADY-SIMULATED
    (n_simulations, n_grid) terminal NAV matrix -- no re-simulation, pure
    vectorized NumPy over the stored matrix. Column i of `terminal_navs`
    corresponds to leverage_grid[i]; every simulation (row) shares the same
    underlying market/RF path across all leverage columns, so these are
    genuine paired comparisons, not independent re-draws.
    """
    n_simulations, n_grid = terminal_navs.shape
    leverage_grid = np.asarray(leverage_grid, dtype=float)

    baseline_idx = int(np.argmin(np.abs(leverage_grid - 1.0)))
    baseline_leverage = float(leverage_grid[baseline_idx])
    baseline_navs = terminal_navs[:, baseline_idx]

    # Table 1: Probability of Beating 1x (baseline = grid level closest to 1.0x)
    with np.errstate(divide='ignore', invalid='ignore'):
        outperf = terminal_navs / baseline_navs[:, None] - 1.0
    outperf[~np.isfinite(outperf)] = np.nan
    beat_baseline_df = pd.DataFrame({
        'leverage': leverage_grid,
        'prob_beat_baseline': np.mean(terminal_navs > baseline_navs[:, None], axis=0),
        'median_outperformance': np.nanmedian(outperf, axis=0),
        'mean_outperformance': np.nanmean(outperf, axis=0),
    })

    # Table 2: Probability of Beating the Previous (adjacent, next-lower) Leverage
    beat_prev_df = pd.DataFrame({
        'leverage': leverage_grid[1:],
        'compared_against': leverage_grid[:-1],
        'prob_win': np.mean(terminal_navs[:, 1:] > terminal_navs[:, :-1], axis=0) if n_grid > 1 else [],
    })

    # Table 3: Optimal Leverage Frequency (argmax per simulation)
    winner_idx = np.argmax(terminal_navs, axis=1)
    counts = np.bincount(winner_idx, minlength=n_grid)
    optimal_freq_df = pd.DataFrame({
        'leverage': leverage_grid,
        'pct_optimal': counts / n_simulations,
    })

    # Table 4: Average Rank (1 = best) per leverage, averaged across all simulations
    order = np.argsort(-terminal_navs, axis=1)
    ranks = np.empty((n_simulations, n_grid), dtype=np.float64)
    rank_values = np.tile(np.arange(1, n_grid + 1, dtype=np.float64), (n_simulations, 1))
    np.put_along_axis(ranks, order, rank_values, axis=1)
    avg_rank_df = pd.DataFrame({
        'leverage': leverage_grid,
        'avg_rank': ranks.mean(axis=0),
    })

    # Optional: Pairwise win matrix, cell[A, B] = P(NAV(A) > NAV(B))
    win_matrix = np.empty((n_grid, n_grid))
    for i in range(n_grid):
        win_matrix[i, :] = np.mean(terminal_navs[:, i:i + 1] > terminal_navs, axis=0)
    np.fill_diagonal(win_matrix, np.nan)
    labels = [f"{l:.1f}x" for l in leverage_grid]
    pairwise_df = pd.DataFrame(win_matrix, index=labels, columns=labels)

    # Additional: Probability of Regret -- P(finishes within X% of the best leverage, per sim)
    best_per_sim = terminal_navs.max(axis=1)
    threshold = regret_threshold_pct / 100.0
    regret_df = pd.DataFrame({
        'leverage': leverage_grid,
        'prob_within_threshold': np.mean(terminal_navs >= best_per_sim[:, None] * (1.0 - threshold), axis=0),
    })

    return {
        'baseline_leverage': baseline_leverage,
        'beat_baseline': beat_baseline_df,
        'beat_prev': beat_prev_df,
        'optimal_freq': optimal_freq_df,
        'avg_rank': avg_rank_df,
        'pairwise_win_matrix': pairwise_df,
        'regret': regret_df,
        'regret_threshold_pct': regret_threshold_pct,
    }

def find_mc_optimal_leverage(df: pd.DataFrame, ruin_threshold_pct: float) -> dict:
    calmar_like = (df['median_cagr'] / df['median_max_dd'].abs()).replace([np.inf, -np.inf], np.nan)
    survivors = df[df['prob_ruin'] * 100.0 < ruin_threshold_pct]
    survival_optimal = float(survivors['leverage'].max()) if len(survivors) > 0 else float(df['leverage'].min())
    sharpe = df['median_sharpe'].replace([np.inf, -np.inf], np.nan)
    return {
        'expected_return_optimal': float(df.loc[df['median_final_nav'].idxmax(), 'leverage']),
        'utility_optimal': float(df.loc[df['median_cagr'].idxmax(), 'leverage']),
        'risk_adjusted_optimal': float(df.loc[calmar_like.idxmax(), 'leverage']) if calmar_like.notna().any() else float('nan'),
        'survival_optimal': survival_optimal,
        'lowest_prob_loss': float(df.loc[df['prob_loss'].idxmin(), 'leverage']),
        'max_sharpe': float(df.loc[sharpe.idxmax(), 'leverage']) if sharpe.notna().any() else float('nan'),
    }

def generate_leverage_interpretation(
    hist_opt: dict, mc_opt: Optional[dict], mc_method: Optional[str], ruin_threshold_pct: float
) -> str:
    text = (
        f"Historical backtest data suggests maximum compounded growth (log-optimal leverage) occurred "
        f"around **{hist_opt['log_optimal']:.1f}x**, while the highest Sharpe ratio occurred near "
        f"**{hist_opt['max_sharpe']:.1f}x** and the highest Calmar ratio near **{hist_opt['max_calmar']:.1f}x**."
    )
    if mc_opt is not None:
        text += (
            f" Monte Carlo simulation ({mc_method}) found the highest median terminal wealth at "
            f"**{mc_opt['expected_return_optimal']:.1f}x** and the highest median CAGR at "
            f"**{mc_opt['utility_optimal']:.1f}x**, with the best risk-adjusted leverage (median CAGR / "
            f"median max drawdown) at **{mc_opt['risk_adjusted_optimal']:.1f}x**. Leverage can be pushed as "
            f"high as **{mc_opt['survival_optimal']:.1f}x** while keeping the probability of ruin below "
            f"{ruin_threshold_pct:.1f}%."
        )
        spread = mc_opt['expected_return_optimal'] - mc_opt['risk_adjusted_optimal']
        if abs(spread) >= 0.3:
            text += (
                " Leverage beyond the risk-adjusted optimum substantially increases drawdown risk for only "
                "modest gains in expected terminal wealth, suggesting diminishing marginal benefit."
            )
        else:
            text += (
                " The expected-return-optimal and risk-adjusted-optimal leverage levels are close, suggesting "
                "leverage in this range offers a reasonably efficient risk/reward trade-off."
            )
    else:
        text += " Run the Monte Carlo leverage sweep to compare against forward-looking simulated paths."
    return text

def plot_leverage_line(leverage: np.ndarray, y: np.ndarray, title: str, ylabel: str, highlights: Optional[list] = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(leverage, y, marker='o', markersize=3, color='royalblue', linewidth=1.5)
    if highlights:
        colors = ['darkorange', 'crimson', 'purple', 'darkgreen']
        for i, (lev_val, label) in enumerate(highlights):
            if lev_val is None or np.isnan(lev_val):
                continue
            idx = int(np.argmin(np.abs(leverage - lev_val)))
            ax.scatter([leverage[idx]], [y[idx]], color=colors[i % len(colors)], s=70, zorder=5, label=label)
        ax.legend(fontsize=8)
    ax.set_xlabel('Leverage (x)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_leverage_heatmap_strip(leverage: np.ndarray, values: np.ndarray, title: str, cmap: str = 'RdYlGn', as_pct: bool = True):
    fig, ax = plt.subplots(figsize=(12, 1.3))
    data = np.asarray(values, dtype=float).reshape(1, -1)
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks(range(len(leverage)))
    ax.set_xticklabels([f"{l:.1f}x" for l in leverage], rotation=90, fontsize=7)
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(values):
        disp = v * 100.0 if as_pct else v
        ax.text(i, 0, f"{disp:.0f}" if as_pct else f"{disp:.2f}", ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.55, fraction=0.25)
    fig.tight_layout()
    return fig

# ----------------------------------------------------------------------
# Starting-Date Sensitivity Module (Rolling Historical Windows)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_rolling_window_analysis(
    df_filtered: pd.DataFrame, leverage_grid: np.ndarray, window_years: int,
    spread_annual: float, expense_ratio_annual: float, trading_days: int = 252, step_years: float = 1.0
) -> pd.DataFrame:
    """
    Slices the SAME historical daily series into many overlapping windows of
    `window_years` length, starting in different years, and applies the
    identical daily-compounding + cost formula used everywhere else in this
    app to every (start date, leverage) combination. This answers "how much
    does the historically-preferred leverage depend on when you happened to
    start investing", as opposed to the single fixed-range backtest above.

    Windows overlap (by design, to get enough starting points out of one
    ~100-year history) so they are NOT independent draws -- treat the spread
    across rows as a description of historical sensitivity, not a confidence
    interval.
    """
    mkt_total = df_filtered['Mkt_Total'].to_numpy()
    rf = df_filtered['RF'].to_numpy()
    dates = df_filtered['date'].to_numpy()
    n = len(mkt_total)
    window_days = int(round(window_years * trading_days))
    step_days = max(1, int(round(step_years * trading_days)))
    spread_daily = spread_annual / trading_days
    expense_daily = expense_ratio_annual / trading_days

    if window_days >= n:
        return pd.DataFrame(columns=['start_date', 'leverage', 'cagr', 'max_dd', 'calmar'])

    rows = []
    for start in range(0, n - window_days + 1, step_days):
        end = start + window_days
        w_mkt = mkt_total[start:end]
        w_rf = rf[start:end]
        start_date = pd.Timestamp(dates[start])
        for lev in leverage_grid:
            daily_cost = (lev - 1.0) * (w_rf + spread_daily) + expense_daily
            lev_ret = lev * w_mkt - daily_cost
            factors = np.maximum(1.0 + lev_ret, 0.0)
            nav = np.cumprod(factors)
            final_nav = nav[-1]
            cagr = final_nav ** (trading_days / window_days) - 1.0
            peak = np.maximum.accumulate(nav)
            dd = nav / peak - 1.0
            max_dd = dd.min()
            calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
            rows.append({
                'start_date': start_date, 'leverage': float(lev),
                'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar
            })

    return pd.DataFrame(rows)

def summarize_rolling_window_analysis(rolling_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lev, grp in rolling_df.groupby('leverage'):
        cagr = grp['cagr'].to_numpy()
        max_dd = grp['max_dd'].to_numpy()
        calmar = grp['calmar'].to_numpy()
        rows.append({
            'leverage': float(lev),
            'median_cagr': float(np.median(cagr)),
            'p5_cagr': float(np.percentile(cagr, 5)),
            'p95_cagr': float(np.percentile(cagr, 95)),
            'median_max_dd': float(np.median(max_dd)),
            'p5_max_dd': float(np.percentile(max_dd, 5)),
            'p95_max_dd': float(np.percentile(max_dd, 95)),
            'worst_max_dd': float(np.min(max_dd)),
            'median_calmar': float(np.nanmedian(calmar)),
            'n_windows': int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values('leverage').reset_index(drop=True)

def find_rolling_optimal_leverage(summary_df: pd.DataFrame) -> dict:
    cagr_range = summary_df['p95_cagr'] - summary_df['p5_cagr']
    return {
        'best_median_cagr': float(summary_df.loc[summary_df['median_cagr'].idxmax(), 'leverage']),
        'best_worst_case_dd': float(summary_df.loc[summary_df['worst_max_dd'].idxmax(), 'leverage']),
        'narrowest_cagr_range': float(summary_df.loc[cagr_range.idxmin(), 'leverage']),
    }

def generate_rolling_interpretation(rolling_opt: dict, window_years: int, n_windows: int) -> str:
    return (
        f"Across {n_windows} overlapping {window_years}-year windows starting in different years, the leverage "
        f"with the highest **median** CAGR was **{rolling_opt['best_median_cagr']:.1f}x**, while "
        f"**{rolling_opt['best_worst_case_dd']:.1f}x** had the mildest worst-case drawdown of any starting date, "
        f"and **{rolling_opt['narrowest_cagr_range']:.1f}x** had the narrowest spread between good and bad starting "
        f"dates -- i.e. the most consistent outcome regardless of when you began. These windows overlap heavily, "
        f"so treat this as a description of historical sensitivity to starting date, not as {n_windows} independent trials."
    )

def plot_rolling_band(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    lev = summary_df['leverage'].to_numpy()
    median = summary_df[f'median_{metric}'].to_numpy() * 100.0
    p5 = summary_df[f'p5_{metric}'].to_numpy() * 100.0
    p95 = summary_df[f'p95_{metric}'].to_numpy() * 100.0
    ax.fill_between(lev, p5, p95, color='royalblue', alpha=0.2, label='P5-P95 across starting dates')
    ax.plot(lev, median, marker='o', markersize=3, color='royalblue', linewidth=1.5, label='Median')
    ax.set_xlabel('Leverage (x)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_rolling_heatmap(
    rolling_df: pd.DataFrame, value_col: str, leverage_grid: np.ndarray, title: str,
    cmap: str = 'RdYlGn', as_pct: bool = True
):
    pivot = rolling_df.pivot(index='start_date', columns='leverage', values=value_col).sort_index()
    pivot = pivot.reindex(columns=leverage_grid)
    data = pivot.to_numpy() * (100.0 if as_pct else 1.0)

    n_rows = len(pivot)
    fig_height = max(4.0, min(18.0, 0.18 * n_rows))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(leverage_grid)))
    ax.set_xticklabels([f"{l:.1f}x" for l in leverage_grid], rotation=90, fontsize=7)
    tick_step = max(1, n_rows // 25)
    yticks = list(range(0, n_rows, tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([pivot.index[i].strftime('%Y-%m') for i in yticks], fontsize=7)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Window Start Date')
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def show_fig(fig):
    """Render a matplotlib figure then close it, so figures don't pile up in
    memory across Streamlit reruns within the same long-lived server process."""
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(page_title="Leveraged Market Simulator", layout="wide")
st.title("📈 Institutional-Grade Leveraged Market Index Simulation")

uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default file (F-F_Research_Data_Factors_daily.csv)", value=True)

if uploaded_file is not None:
    df_raw = load_data_from_file(uploaded_file)
    st.sidebar.success("Uploaded file loaded")
elif use_default:
    df_raw = load_default()
    if df_raw is None:
        st.sidebar.error("Default file not found. Please upload a CSV.")
        st.stop()
    st.sidebar.success("Default file loaded")
else:
    st.sidebar.warning("Please upload a file or use default.")
    st.stop()

df_raw = prepare_returns_data(df_raw)
min_year = int(df_raw['date'].dt.year.min())
max_year = int(df_raw['date'].dt.year.max())

st.sidebar.markdown("---")
st.sidebar.header("Parameters")
leverage = st.sidebar.slider("Leverage (x)", 1.0, 3.0, 2.0, 0.1)
spread_annual = st.sidebar.slider("Borrowing Spread (annual %)", 0.0, 2.0, 0.4, 0.05) / 100.0
expense_annual = st.sidebar.slider("Expense Ratio (annual %)", 0.0, 2.0, 0.95, 0.05) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Date Range")
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year), step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Chart Options")
use_log_scale = st.sidebar.checkbox("Use logarithmic scale", value=True)

df_filtered = df_raw[
    (df_raw['date'].dt.year >= year_range[0]) & (df_raw['date'].dt.year <= year_range[1])
]
if len(df_filtered) == 0:
    st.warning("No data for the selected year range. Please adjust the slider.")
    st.stop()

df_result, leveraged_stats, unleveraged_stats = simulate_leveraged_etf(
    df_filtered, leverage, spread_annual, expense_annual
)

st.subheader("Performance Summary – Leveraged")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Annualised Return", f"{leveraged_stats['ann_return']*100:.2f}%")
col2.metric("Annualised Volatility", f"{leveraged_stats['ann_vol']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{leveraged_stats['sharpe']:.2f}")
col4.metric("Sortino Ratio", f"{leveraged_stats['sortino']:.2f}")
col5.metric("Max Drawdown", f"{leveraged_stats['max_dd']*100:.2f}%")
col6.metric("CVaR (95%)", f"{leveraged_stats['cvar_95']*100:.2f}%")

st.subheader("Performance Summary – Market (1×, no costs)")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Annualised Return", f"{unleveraged_stats['ann_return']*100:.2f}%")
col2.metric("Annualised Volatility", f"{unleveraged_stats['ann_vol']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{unleveraged_stats['sharpe']:.2f}")
col4.metric("Sortino Ratio", f"{unleveraged_stats['sortino']:.2f}")
col5.metric("Max Drawdown", f"{unleveraged_stats['max_dd']*100:.2f}%")
col6.metric("CVaR (95%)", f"{unleveraged_stats['cvar_95']*100:.2f}%")

st.subheader("Growth of $1 — Cumulative Performance")
fig, ax = plt.subplots(figsize=(12, 6))
if use_log_scale:
    ax.plot(df_result['date'], df_result['Lev_NAV'], label=f'{leverage:.1f}x Leveraged', color='royalblue', linewidth=2)
    ax.plot(df_result['date'], df_result['Mkt_NAV'], label='Market 1x', color='green', linestyle='--', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('Growth of $1 (Log Scale)')
else:
    ax.plot(df_result['date'], (df_result['Lev_NAV'] - 1) * 100, label=f'{leverage:.1f}x Leveraged', color='royalblue')
    ax.plot(df_result['date'], (df_result['Mkt_NAV'] - 1) * 100, label='Market 1x', color='green', linestyle='--')
    ax.set_ylabel('Total Return (%)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
ax.legend()
show_fig(fig)

# ========== MONTE CARLO SIMULATION ==========
st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo Projection")
run_mc = st.sidebar.checkbox("Run Monte Carlo simulation", value=False)
if run_mc:
    n_years = st.sidebar.slider("Projection years", 10, 60, 40, 5)
    n_simulations = st.sidebar.selectbox("Number of simulations", [100, 500, 1000, 2000, 5000], index=2)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Model")
    model_type_map = {
        "Block Bootstrap": "block_bootstrap",
        "IID Bootstrap": "iid_bootstrap",
        "GARCH(1,1)": "garch",
        "Regime-Switching": "regime_switching"
    }
    model_type_label = st.sidebar.selectbox(
        "Model", 
        list(model_type_map.keys()),
        help="Regime-Switching uses a Markov chain to simulate Bull/Neutral/Crisis states. GARCH captures volatility clustering. Bootstrap preserves empirical ordering."
    )
    model_type = model_type_map[model_type_label]
    
    block_size = 20
    garch_dist = "Resampled (Fat Tails)"
    
    if model_type == "block_bootstrap":
        block_size = st.sidebar.slider("Block Size (days)", min_value=5, max_value=60, value=20, step=5)
    elif model_type == "garch":
        garch_dist = st.sidebar.selectbox(
            "Innovation Distribution (z_t)",
            ["Resampled (Fat Tails)", "Gaussian", "Student-t"],
            help="Resampled uses empirical standardized residuals. Gaussian/Student-t draw from parametric distributions."
        )
        if n_simulations > 2000:
            st.sidebar.warning("GARCH simulation may be slow for >2000 paths. Consider reducing simulations.")
    elif model_type == "regime_switching":
        st.sidebar.info("💡 **Regime-Switching:** Estimates 3 hidden states (Bull, Neutral, Crisis) from history and simulates transitions via a Markov chain. Excellent for capturing fat tails and crash clustering.")

    st.sidebar.markdown("---")
    random_seed = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
        help="Fixing the seed makes runs reproducible so parameter changes can be compared without MC noise."
    )

    mc_button = st.sidebar.button("Run Simulation")

    if mc_button:
        with st.spinner("Running Monte Carlo..."):
            mc_results = run_monte_carlo(
                df_filtered, leverage, spread_annual, expense_annual,
                n_simulations=n_simulations, n_years=n_years,
                block_size=block_size, model_type=model_type, garch_dist=garch_dist,
                random_seed=int(random_seed)
            )
        st.session_state['mc_results'] = mc_results

    if 'mc_results' in st.session_state:
        mc = st.session_state['mc_results']
        method_text = f"{mc['method']}"
        st.caption(f"Simulation method: {method_text}")
        if mc.get('garch_warning'):
            st.warning(f"⚠️ {mc['garch_warning']}")

        fig, ax = plt.subplots(figsize=(14, 7))
        days = np.arange(len(mc['lev']['percentiles'][0]))
        ax.fill_between(days, mc['lev']['percentiles'][0], mc['lev']['percentiles'][4], color='royalblue', alpha=0.1, label='Leveraged 1st–99th')
        ax.fill_between(days, mc['lev']['percentiles'][1], mc['lev']['percentiles'][3], color='royalblue', alpha=0.3, label='Leveraged 5th–95th')
        ax.plot(days, mc['lev']['percentiles'][2], color='royalblue', linewidth=2, label=f'{leverage:.1f}x Leveraged (median)')
        ax.fill_between(days, mc['mkt']['percentiles'][0], mc['mkt']['percentiles'][4], color='green', alpha=0.1, label='Market 1st–99th')
        ax.fill_between(days, mc['mkt']['percentiles'][1], mc['mkt']['percentiles'][3], color='green', alpha=0.3, label='Market 5th–95th')
        ax.plot(days, mc['mkt']['percentiles'][2], color='green', linestyle='--', linewidth=2, label='Market 1x (median)')
        ax.plot(days, mc['lev']['representative_path'], color='navy', linestyle=':', linewidth=1, alpha=0.8, label='Leveraged (one real path)')
        ax.plot(days, mc['mkt']['representative_path'], color='darkgreen', linestyle=':', linewidth=1, alpha=0.8, label='Market (one real path)')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('NAV Growth')
        ax.set_title('Projected NAV Distribution: Leveraged vs Market')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if use_log_scale: ax.set_yscale('log')
        show_fig(fig)

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(mc['lev']['max_dd_dist'], bins=50, alpha=0.7, color='royalblue', density=True)
        ax1.set_title('Leveraged Max Drawdown Distribution')
        ax2.hist(mc['mkt']['max_dd_dist'], bins=50, alpha=0.7, color='green', density=True)
        ax2.set_title('Market Max Drawdown Distribution')
        show_fig(fig2)

        lev = mc['lev']
        mkt = mc['mkt']
        prob_lev_beats = mc.get('prob_lev_beats_mkt', 0.0)
        
        comp_df = pd.DataFrame({
            'Statistic': [
                'Median Final NAV',
                'Worst Case (1st pctl)',
                'Best Case (99th pctl)',
                'Probability of Loss',
                'Probability of Doubling',
                'Probability of Ruin (<10¢)',
                'Probability of >70% DD',
                'Mean Final NAV',
                'Median Sharpe Ratio',
                'Median Sortino Ratio',
                'CVaR 95% (Terminal NAV)',
                'CVaR 95% (Ann. Return)',
                'CVaR 95% (Max Drawdown)',
                f'Prob. {leverage:.1f}x Outperforms 1x Market'
            ],
            'Leveraged': [
                f"{lev['median_final']:.2f}x",
                f"{lev['p1_final']:.2f}x",
                f"{lev['p99_final']:.2f}x",
                f"{lev['prob_loss']*100:.1f}%",
                f"{lev['prob_double']*100:.1f}%",
                f"{lev['prob_ruin']*100:.1f}%",
                f"{lev['prob_70pct_dd']*100:.1f}%",
                f"{lev['mean_final']:.2f}x",
                f"{lev['median_sharpe']:.2f}",
                f"{lev['median_sortino']:.2f}",
                f"{lev['cvar_terminal']:.2f}x",
                f"{lev['cvar_ann_ret']*100:.2f}%",
                f"{lev['cvar_max_dd']*100:.2f}%",
                f"{prob_lev_beats*100:.1f}%"
            ],
            'Market': [
                f"{mkt['median_final']:.2f}x",
                f"{mkt['p1_final']:.2f}x",
                f"{mkt['p99_final']:.2f}x",
                f"{mkt['prob_loss']*100:.1f}%",
                f"{mkt['prob_double']*100:.1f}%",
                f"{mkt['prob_ruin']*100:.1f}%",
                f"{mkt['prob_70pct_dd']*100:.1f}%",
                f"{mkt['mean_final']:.2f}x",
                f"{mkt['median_sharpe']:.2f}",
                f"{mkt['median_sortino']:.2f}",
                f"{mkt['cvar_terminal']:.2f}x",
                f"{mkt['cvar_ann_ret']*100:.2f}%",
                f"{mkt['cvar_max_dd']*100:.2f}%",
                f"{(1 - prob_lev_beats)*100:.1f}%"
            ]
        })
        st.table(comp_df)
        
        # ===== LEVERAGE DOMINANCE STRUCTURE =====
        st.subheader("📊 Leverage Dominance Structure")
        
        st.markdown("""
        - **Conditional dominance** shows leverage is regime-dependent.
        - **Time-weighted dominance** shows path instability over long horizons.
        - **Sustained dominance** shows whether leverage is consistently superior or only episodically superior.
        """)
        
        if 'dominance_analysis' in mc:
            dom = mc['dominance_analysis']
            
            # 1. Conditional Dominance
            st.markdown("#### 1. Conditional Dominance (Market Regimes)")
            cond_df = pd.DataFrame({
                'Regime': ['Crash (Bottom 25%)', 'Neutral (Middle 50%)', 'Bull (Top 25%)'],
                'P(2x > 1x)': [f"{dom['conditional']['crash']['prob_lev_wins']*100:.1f}%", 
                               f"{dom['conditional']['neutral']['prob_lev_wins']*100:.1f}%", 
                               f"{dom['conditional']['bull']['prob_lev_wins']*100:.1f}%"],
                'Mean Outperformance (2x - 1x)': [f"{dom['conditional']['crash']['mean_outperformance']:.2f}x", 
                                                  f"{dom['conditional']['neutral']['mean_outperformance']:.2f}x", 
                                                  f"{dom['conditional']['bull']['mean_outperformance']:.2f}x"],
                'P(Ruin < $0.10)': [f"{dom['conditional']['crash']['prob_ruin']*100:.1f}%", 
                                    f"{dom['conditional']['neutral']['prob_ruin']*100:.1f}%", 
                                    f"{dom['conditional']['bull']['prob_ruin']*100:.1f}%"],
                'Median Ratio (2x / 1x)': [f"{dom['conditional']['crash']['median_ratio']:.2f}", 
                                           f"{dom['conditional']['neutral']['median_ratio']:.2f}", 
                                           f"{dom['conditional']['bull']['median_ratio']:.2f}"]
            })
            st.table(cond_df)
            
            # 2. Time-Weighted Dominance
            st.markdown("#### 2. Time-Weighted Dominance")
            st.write(f"**Average fraction of time 2x is ahead:** {dom['time_weighted']['avg_time_in_lead']*100:.1f}%")
            
            horizons = dom['time_weighted']['horizons']
            
            fig_tw, ax_tw = plt.subplots(figsize=(10, 5))
            ax_tw.plot(list(horizons.keys()), [v*100 for v in horizons.values()], marker='o', color='royalblue', linewidth=2)
            ax_tw.set_xlabel('Time Horizon')
            ax_tw.set_ylabel('Probability 2x is Ahead (%)')
            ax_tw.set_title('Probability of Leveraged ETF Outperforming 1x Market Over Time')
            ax_tw.grid(True, alpha=0.3)
            ax_tw.set_ylim(0, 105)
            show_fig(fig_tw)
            
            # 3. Sustained Outperformance
            st.markdown("#### 3. Sustained Outperformance")
            st.write(f"**Median fraction of time 2x is ahead across all paths:** {dom['sustained']['median_fraction_ahead']*100:.1f}%")
            
            sus_df = pd.DataFrame({
                'Threshold (% of time ahead)': ['50%', '60%', '70%', '80%'],
                'Probability of Sustained Outperformance': [
                    f"{dom['sustained']['50%']*100:.1f}%",
                    f"{dom['sustained']['60%']*100:.1f}%",
                    f"{dom['sustained']['70%']*100:.1f}%",
                    f"{dom['sustained']['80%']*100:.1f}%"
                ]  
            })
            st.table(sus_df)
            
        # ===== ✅ CORRECTED KELLY CRITERION SECTION =====
        st.subheader("Optimal Position Sizing (Kelly Criterion)")
        st.warning("⚠️ **This is allocation to underlying risky asset, NOT leverage on leveraged ETF.**")
        
        st.markdown("""
        **Interpretation:**
        - **Full Kelly (f*)**: Theoretical optimal fraction of total capital to allocate to the underlying market index.
        - **Half / Quarter Kelly**: Risk-managed allocations (recommended for practical implementation to reduce volatility).
        - **Drawdown-Adjusted Kelly**: Penalizes allocation based on tail risk (CVaR).
        - **Log-Optimal Kelly**: Found via grid search maximizing expected log growth (geometric mean).
        """)

        uk = mc['kelly']['underlying']

        kelly_df = pd.DataFrame({
            'Metric': [
                'Full Kelly (f*)', 
                'Half Kelly (50%)', 
                'Quarter Kelly (25%)', 
                'Drawdown-Adjusted Kelly',
                'Log-Optimal Kelly (Grid Search)'
            ],
            'Leveraged Strategy (Underlying Allocation)': [
                f"{uk['full_kelly']:.3f}", f"{uk['half_kelly']:.3f}", f"{uk['quarter_kelly']:.3f}", 
                f"{uk['dd_adjusted_kelly']:.3f}", f"{uk['log_optimal_kelly']:.3f}"
            ],
            'Market Strategy (Underlying Allocation)': [
                f"{uk['full_kelly']:.3f}", f"{uk['half_kelly']:.3f}", f"{uk['quarter_kelly']:.3f}", 
                f"{uk['dd_adjusted_kelly']:.3f}", f"{uk['log_optimal_kelly']:.3f}"
            ]
        })
        st.table(kelly_df)
        
        st.info("💡 **How to use this:** Because Kelly is a property of the *underlying* asset, the optimal allocation is identical for both strategies. If you are using a Leveraged ETF, divide the Kelly fraction by the ETF's leverage to find your actual position size (e.g., for a 2x ETF and Full Kelly of 1.5, allocate `1.5 / 2 = 0.75` or 75% of your capital to the ETF).")
        # ===================================================================
    else:
        st.info("Click 'Run Simulation' in the sidebar to generate projections.")

# ========== OPTIMAL LEVERAGE EXPLORER ==========
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Optimal Leverage Explorer")
run_leverage_explorer = st.sidebar.checkbox("Enable Optimal Leverage Explorer", value=False)

if run_leverage_explorer:
    st.sidebar.markdown("**Leverage grid**")
    le_min_lev = st.sidebar.number_input("Min leverage", 0.5, 5.0, 1.0, 0.1, key="le_min_lev")
    le_max_lev = st.sidebar.number_input("Max leverage", 0.5, 5.0, 3.0, 0.1, key="le_max_lev")
    le_step = st.sidebar.number_input("Leverage step", 0.05, 1.0, 0.1, 0.05, key="le_step")
    le_ruin_threshold = st.sidebar.slider("Survival-optimal: max P(ruin) (%)", 0.1, 20.0, 1.0, 0.1, key="le_ruin_thr")

    st.sidebar.markdown("**Starting-Date Sensitivity (rolling windows)**")
    rw_window_years = st.sidebar.slider("Rolling window length (years)", 5, 40, 20, 5, key="rw_window_years")
    rw_step_years = st.sidebar.selectbox("Rolling step (years)", [1, 2, 5], index=0, key="rw_step_years")

    st.sidebar.markdown("**Monte Carlo settings (leverage sweep)**")
    le_model_type_map = {
        "Block Bootstrap": "block_bootstrap",
        "IID Bootstrap": "iid_bootstrap",
        "GARCH(1,1)": "garch",
        "Regime-Switching": "regime_switching"
    }
    le_model_label = st.sidebar.selectbox(
        "Model (leverage sweep)", list(le_model_type_map.keys()), key="le_model_label",
        help="The historical panel below always runs; this selects which Monte Carlo model the sweep uses. Switch models to compare optimal leverage across bootstrap, GARCH, and regime-switching."
    )
    le_model_type = le_model_type_map[le_model_label]
    le_n_simulations = st.sidebar.selectbox("Simulations (leverage sweep)", [100, 300, 500, 1000, 2000], index=2, key="le_n_sims")
    le_n_years = st.sidebar.slider("Projection years (leverage sweep)", 10, 60, 30, 5, key="le_n_years")

    le_block_size = 20
    le_garch_dist = "Resampled (Fat Tails)"
    if le_model_type == "block_bootstrap":
        le_block_size = st.sidebar.slider("Block size (leverage sweep)", 5, 60, 20, 5, key="le_block_size")
    elif le_model_type == "garch":
        le_garch_dist = st.sidebar.selectbox(
            "Innovation distribution (leverage sweep)",
            ["Resampled (Fat Tails)", "Gaussian", "Student-t"], key="le_garch_dist"
        )
        if le_n_simulations > 1000:
            st.sidebar.warning("GARCH leverage sweeps with >1000 simulations may be slow.")

    le_seed = st.sidebar.number_input("Random seed (leverage sweep)", 0, 2**31 - 1, 42, 1, key="le_seed")

    le_run_button = st.sidebar.button("Run Leverage Explorer (Monte Carlo)")

    st.markdown("---")
    st.header("🎯 Optimal Leverage Explorer")
    leverage_grid = get_leverage_grid(le_min_lev, le_max_lev, le_step)
    st.caption(
        f"Sweeping leverage from {le_min_lev:.1f}x to {le_max_lev:.1f}x in steps of {le_step:.2f}x "
        f"({len(leverage_grid)} levels), evaluated on the historical backtest and, on request, on Monte Carlo scenarios."
    )

    # ---------- Historical Optimization ----------
    st.subheader("📜 Historical Optimization")
    hist_grid_df = compute_leverage_grid_historical(
        df_filtered, leverage_grid, spread_annual, expense_annual, trading_days=252
    )
    hist_opt = find_historical_optimal_leverage(hist_grid_df)

    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    hc1.metric("Max CAGR Leverage", f"{hist_opt['max_cagr']:.1f}x")
    hc2.metric("Max Sharpe Leverage", f"{hist_opt['max_sharpe']:.1f}x")
    hc3.metric("Max Sortino Leverage", f"{hist_opt['max_sortino']:.1f}x")
    hc4.metric("Max Calmar Leverage", f"{hist_opt['max_calmar']:.1f}x")
    hc5.metric("Log-Optimal Leverage", f"{hist_opt['log_optimal']:.1f}x")

    display_hist_df = hist_grid_df.copy()
    display_hist_df['leverage'] = hist_grid_df['leverage'].map(lambda x: f"{x:.1f}x")
    for col in ['cagr', 'ann_vol', 'max_dd', 'cvar_95', 'total_return', 'geo_mean_daily']:
        display_hist_df[col] = hist_grid_df[col].map(lambda x: f"{x*100:.2f}%")
    display_hist_df['sharpe'] = hist_grid_df['sharpe'].map(lambda x: f"{x:.2f}")
    display_hist_df['sortino'] = hist_grid_df['sortino'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    display_hist_df['ulcer_index'] = hist_grid_df['ulcer_index'].map(lambda x: f"{x:.2f}")
    display_hist_df['calmar'] = hist_grid_df['calmar'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    display_hist_df['mean_log_growth'] = hist_grid_df['mean_log_growth'].map(lambda x: f"{x:.6f}")
    display_hist_df = display_hist_df.rename(columns={
        'leverage': 'Leverage', 'cagr': 'CAGR', 'ann_vol': 'Ann. Vol', 'sharpe': 'Sharpe', 'sortino': 'Sortino',
        'max_dd': 'Max DD', 'cvar_95': 'CVaR 95%', 'total_return': 'Total Return',
        'geo_mean_daily': 'Geo. Mean (Daily)', 'ulcer_index': 'Ulcer Index', 'calmar': 'Calmar',
        'mean_log_growth': 'Mean Log Growth'
    })
    st.dataframe(display_hist_df, hide_index=True)

    st.markdown("#### Efficient Frontier (Historical)")
    fig_ef, ax_ef = plt.subplots(figsize=(9, 5))
    ax_ef.plot(hist_grid_df['ann_vol'] * 100, hist_grid_df['cagr'] * 100, marker='o', markersize=4, color='royalblue', linewidth=1.5)
    lev_col = hist_grid_df['leverage'].to_numpy()
    for lev_val, label, color in [
        (hist_opt['max_cagr'], 'Max CAGR', 'crimson'),
        (hist_opt['max_sharpe'], 'Max Sharpe', 'darkorange'),
        (hist_opt['log_optimal'], 'Log-Optimal', 'purple'),
    ]:
        if lev_val is None or np.isnan(lev_val):
            continue
        idx = int(np.argmin(np.abs(lev_col - lev_val)))
        ax_ef.scatter([hist_grid_df['ann_vol'].iloc[idx] * 100], [hist_grid_df['cagr'].iloc[idx] * 100],
                      color=color, s=90, zorder=5, label=f"{label} ({lev_val:.1f}x)")
    ax_ef.set_xlabel('Annualized Volatility (%)')
    ax_ef.set_ylabel('CAGR (%)')
    ax_ef.set_title('Historical Efficient Frontier Across Leverage')
    ax_ef.grid(True, alpha=0.3)
    ax_ef.legend(fontsize=8)
    show_fig(fig_ef)

    st.markdown("#### Metrics vs. Leverage (Historical)")
    lc1, lc2 = st.columns(2)
    with lc1:
        show_fig(plot_leverage_line(lev_col, hist_grid_df['cagr'] * 100, 'CAGR vs Leverage', 'CAGR (%)',
                                      highlights=[(hist_opt['max_cagr'], 'Max CAGR')]))
        show_fig(plot_leverage_line(lev_col, hist_grid_df['max_dd'] * 100, 'Max Drawdown vs Leverage', 'Max Drawdown (%)'))
        show_fig(plot_leverage_line(lev_col, hist_grid_df['cvar_95'] * 100, 'CVaR (95%) vs Leverage', 'CVaR 95% (%)'))
    with lc2:
        show_fig(plot_leverage_line(lev_col, hist_grid_df['sharpe'], 'Sharpe Ratio vs Leverage', 'Sharpe',
                                      highlights=[(hist_opt['max_sharpe'], 'Max Sharpe')]))
        show_fig(plot_leverage_line(lev_col, hist_grid_df['sortino'], 'Sortino Ratio vs Leverage', 'Sortino',
                                      highlights=[(hist_opt['max_sortino'], 'Max Sortino')]))
        show_fig(plot_leverage_line(lev_col, hist_grid_df['calmar'], 'Calmar Ratio vs Leverage', 'Calmar',
                                      highlights=[(hist_opt['max_calmar'], 'Max Calmar')]))
        show_fig(plot_leverage_line(lev_col, hist_grid_df['ulcer_index'], 'Ulcer Index vs Leverage', 'Ulcer Index'))

    # ---------- Starting-Date Sensitivity (Rolling Windows) ----------
    st.markdown("---")
    st.subheader("📅 Starting-Date Sensitivity (Rolling Windows)")
    st.caption(
        f"Same historical data and cost model as above, but sliced into many overlapping {rw_window_years}-year "
        "windows starting in different years -- shows how much the 'best' leverage depends on when you happened "
        "to start investing, not just what the single full-period backtest says."
    )
    rolling_df = compute_rolling_window_analysis(
        df_filtered, leverage_grid, rw_window_years, spread_annual, expense_annual,
        trading_days=252, step_years=rw_step_years
    )
    if len(rolling_df) == 0:
        st.warning(
            f"The {rw_window_years}-year rolling window is longer than the selected date range "
            f"({len(df_filtered)} trading days available). Reduce the window length or widen the date range."
        )
    else:
        n_windows = rolling_df['start_date'].nunique()
        rolling_summary = summarize_rolling_window_analysis(rolling_df)
        rolling_opt = find_rolling_optimal_leverage(rolling_summary)

        rw1, rw2, rw3 = st.columns(3)
        rw1.metric("Best Median CAGR", f"{rolling_opt['best_median_cagr']:.1f}x")
        rw2.metric("Mildest Worst-Case Drawdown", f"{rolling_opt['best_worst_case_dd']:.1f}x")
        rw3.metric("Most Consistent (Narrowest CAGR Range)", f"{rolling_opt['narrowest_cagr_range']:.1f}x")
        st.caption(f"Based on {n_windows} overlapping {rw_window_years}-year windows (step: {rw_step_years} year(s)).")

        display_rolling_summary = rolling_summary.copy()
        display_rolling_summary['leverage'] = rolling_summary['leverage'].map(lambda x: f"{x:.1f}x")
        for col in ['median_cagr', 'p5_cagr', 'p95_cagr', 'median_max_dd', 'p5_max_dd', 'p95_max_dd', 'worst_max_dd']:
            display_rolling_summary[col] = rolling_summary[col].map(lambda x: f"{x*100:.2f}%")
        display_rolling_summary['median_calmar'] = rolling_summary['median_calmar'].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"
        )
        display_rolling_summary = display_rolling_summary.rename(columns={
            'leverage': 'Leverage', 'median_cagr': 'Median CAGR', 'p5_cagr': 'P5 CAGR', 'p95_cagr': 'P95 CAGR',
            'median_max_dd': 'Median Max DD', 'p5_max_dd': 'P5 Max DD', 'p95_max_dd': 'P95 Max DD',
            'worst_max_dd': 'Worst-Case Max DD', 'median_calmar': 'Median Calmar', 'n_windows': 'Windows'
        })
        st.dataframe(display_rolling_summary, hide_index=True)

        rwc1, rwc2 = st.columns(2)
        with rwc1:
            show_fig(plot_rolling_band(
                rolling_summary, 'cagr', 'CAGR Across Starting Dates (P5-P95 band)', 'CAGR (%)'
            ))
        with rwc2:
            show_fig(plot_rolling_band(
                rolling_summary, 'max_dd', 'Max Drawdown Across Starting Dates (P5-P95 band)', 'Max Drawdown (%)'
            ))

        with st.expander("📊 Heatmap: CAGR by Starting Date × Leverage"):
            show_fig(plot_rolling_heatmap(rolling_df, 'cagr', leverage_grid, 'CAGR by Starting Date and Leverage'))
        with st.expander("📊 Heatmap: Max Drawdown by Starting Date × Leverage"):
            show_fig(plot_rolling_heatmap(
                rolling_df, 'max_dd', leverage_grid, 'Max Drawdown by Starting Date and Leverage', cmap='RdYlGn_r'
            ))

        st.markdown("#### Interpretation")
        st.info(generate_rolling_interpretation(rolling_opt, rw_window_years, n_windows))

    # ---------- Monte Carlo Optimization ----------
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Optimization")

    if le_run_button:
        with st.spinner(f"Running leverage sweep across {len(leverage_grid)} levels ({le_model_label})..."):
            try:
                le_pairs, le_method_text, _, le_garch_warning = generate_scenario_pairs(
                    df_filtered, le_n_simulations, le_n_years, 252,
                    le_block_size, le_model_type, le_garch_dist, int(le_seed)
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()
            mc_grid_df, mc_terminal_navs = compute_leverage_grid_mc(
                le_pairs, leverage_grid, spread_annual, expense_annual, 252
            )
        st.session_state['leverage_mc_grid'] = {
            'df': mc_grid_df, 'method': le_method_text, 'ruin_threshold': le_ruin_threshold,
            'terminal_navs': mc_terminal_navs, 'leverage_grid': leverage_grid,
            'garch_warning': le_garch_warning
        }

    if 'leverage_mc_grid' in st.session_state:
        mc_grid_state = st.session_state['leverage_mc_grid']
        mc_grid_df = mc_grid_state['df']
        mc_ruin_threshold = mc_grid_state['ruin_threshold']
        mc_method = mc_grid_state['method']
        mc_opt = find_mc_optimal_leverage(mc_grid_df, mc_ruin_threshold)

        st.caption(f"Simulation method: {mc_method}")
        if mc_grid_state.get('garch_warning'):
            st.warning(f"⚠️ {mc_grid_state['garch_warning']}")

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Expected-Return Optimal", f"{mc_opt['expected_return_optimal']:.1f}x")
        mc2.metric("Utility-Optimal (CAGR)", f"{mc_opt['utility_optimal']:.1f}x")
        mc3.metric("Max Sharpe (MC)", f"{mc_opt['max_sharpe']:.1f}x")
        mc4.metric("Risk-Adjusted Optimal", f"{mc_opt['risk_adjusted_optimal']:.1f}x")
        mc5.metric(f"Survival-Optimal (P<{mc_ruin_threshold:.1f}%)", f"{mc_opt['survival_optimal']:.1f}x")

        display_mc_df = mc_grid_df.copy()
        display_mc_df['leverage'] = mc_grid_df['leverage'].map(lambda x: f"{x:.1f}x")
        for col in ['median_final_nav', 'mean_final_nav', 'cvar_terminal']:
            display_mc_df[col] = mc_grid_df[col].map(lambda x: f"{x:.2f}x")
        for col in ['median_cagr', 'prob_loss', 'prob_ruin', 'prob_beat_mkt', 'median_max_dd']:
            display_mc_df[col] = mc_grid_df[col].map(lambda x: f"{x*100:.2f}%")
        display_mc_df['median_sharpe'] = mc_grid_df['median_sharpe'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
        display_mc_df['median_sortino'] = mc_grid_df['median_sortino'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
        display_mc_df['prob_beat_prev_leverage'] = mc_grid_df['prob_beat_prev_leverage'].map(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "n/a"
        )
        display_mc_df = display_mc_df.rename(columns={
            'leverage': 'Leverage', 'median_final_nav': 'Median Final NAV', 'mean_final_nav': 'Mean Final NAV',
            'median_cagr': 'Median CAGR', 'prob_loss': 'P(Loss)', 'prob_ruin': 'P(Ruin)',
            'prob_beat_mkt': 'P(Beat 1x)', 'prob_beat_prev_leverage': 'P(Beat Next-Lower Lev.)',
            'cvar_terminal': 'CVaR 95% (Terminal)', 'median_max_dd': 'Median Max DD',
            'median_sharpe': 'Median Sharpe', 'median_sortino': 'Median Sortino'
        })
        st.dataframe(display_mc_df, hide_index=True)

        st.markdown("#### Heatmaps")
        mc_lev_arr = mc_grid_df['leverage'].to_numpy()
        show_fig(plot_leverage_heatmap_strip(mc_lev_arr, mc_grid_df['prob_beat_mkt'].to_numpy(),
                                               'Probability of Outperforming 1x Market (%)'))
        show_fig(plot_leverage_heatmap_strip(mc_lev_arr, mc_grid_df['median_final_nav'].to_numpy(),
                                               'Median Final NAV (x)', as_pct=False))
        show_fig(plot_leverage_heatmap_strip(mc_lev_arr, mc_grid_df['prob_ruin'].to_numpy(),
                                               'Probability of Ruin (%)', cmap='RdYlGn_r'))
        show_fig(plot_leverage_heatmap_strip(mc_lev_arr, np.abs(mc_grid_df['median_max_dd'].to_numpy()),
                                               'Median Maximum Drawdown (%)', cmap='RdYlGn_r'))

        st.markdown("#### Metrics vs. Leverage (Monte Carlo)")
        mcl1, mcl2 = st.columns(2)
        with mcl1:
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['median_cagr'] * 100, 'Median CAGR vs Leverage', 'Median CAGR (%)',
                                          highlights=[(mc_opt['utility_optimal'], 'Utility-Optimal')]))
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['median_max_dd'] * 100, 'Median Max Drawdown vs Leverage', 'Median Max DD (%)'))
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['prob_ruin'] * 100, 'Probability of Ruin vs Leverage', 'P(Ruin) (%)',
                                          highlights=[(mc_opt['survival_optimal'], 'Survival-Optimal')]))
        with mcl2:
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['median_final_nav'], 'Median Terminal Wealth vs Leverage', 'Median Final NAV (x)',
                                          highlights=[(mc_opt['expected_return_optimal'], 'Expected-Return Optimal')]))
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['median_sharpe'], 'Median Sharpe Ratio vs Leverage', 'Median Sharpe',
                                          highlights=[(mc_opt['max_sharpe'], 'Max Sharpe')]))
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['prob_beat_mkt'] * 100, 'Probability of Beating 1x vs Leverage', 'P(Beat 1x) (%)'))
            show_fig(plot_leverage_line(mc_lev_arr, mc_grid_df['cvar_terminal'], 'CVaR 95% Terminal NAV vs Leverage', 'CVaR 95% (x)'))

        st.markdown("#### Summary Table")
        summary_df = pd.DataFrame({
            'Metric': [
                'Highest CAGR (Historical)', 'Highest Sharpe (Historical)', 'Highest Calmar (Historical)',
                'Highest Log Growth (Historical)', 'Lowest Probability of Loss (MC)',
                'Highest Median Terminal Wealth (MC)', 'Highest Median CAGR (MC)', 'Highest Median Sharpe (MC)',
                'Best Risk-Adjusted Leverage (MC)', f'Survival-Optimal, P(ruin)<{mc_ruin_threshold:.1f}% (MC)'
            ],
            'Optimal Leverage': [
                f"{hist_opt['max_cagr']:.1f}x", f"{hist_opt['max_sharpe']:.1f}x", f"{hist_opt['max_calmar']:.1f}x",
                f"{hist_opt['log_optimal']:.1f}x", f"{mc_opt['lowest_prob_loss']:.1f}x",
                f"{mc_opt['expected_return_optimal']:.1f}x", f"{mc_opt['utility_optimal']:.1f}x",
                f"{mc_opt['max_sharpe']:.1f}x", f"{mc_opt['risk_adjusted_optimal']:.1f}x", f"{mc_opt['survival_optimal']:.1f}x"
            ]
        })
        st.table(summary_df)

        st.markdown("#### Interpretation")
        st.info(generate_leverage_interpretation(hist_opt, mc_opt, mc_method, mc_ruin_threshold))

        # ---------- Leverage Comparison Analysis ----------
        with st.expander("📐 Leverage Comparison Analysis", expanded=False):
            st.caption(
                "Every table below is derived from the terminal-NAV matrix already computed above "
                "(one row per simulation, one column per leverage level) -- no additional simulation is run."
            )
            comp_lev_grid = mc_grid_state['leverage_grid']
            comp_terminal_navs = mc_grid_state['terminal_navs']
            regret_threshold_pct = st.slider(
                "Regret threshold: 'finishes within X% of the best leverage' (%)",
                1.0, 50.0, 10.0, 1.0, key="le_regret_threshold"
            )
            comp = compute_leverage_comparison_tables(comp_terminal_navs, comp_lev_grid, regret_threshold_pct)

            if abs(comp['baseline_leverage'] - 1.0) > 1e-6:
                st.caption(
                    f"Grid does not include exactly 1.0x; using the closest available level "
                    f"({comp['baseline_leverage']:.2f}x) as the '1x' baseline below."
                )

            st.markdown(f"##### Probability of Beating {comp['baseline_leverage']:.1f}x")
            t1 = comp['beat_baseline'].style.format({
                'leverage': '{:.1f}x', 'prob_beat_baseline': '{:.1%}',
                'median_outperformance': '{:+.1%}', 'mean_outperformance': '{:+.1%}'
            }).highlight_max(subset=['prob_beat_baseline'], color='#c6f6d5').hide(axis='index')
            st.dataframe(t1)

            st.markdown("##### Probability of Beating the Previous (Next-Lower) Leverage")
            if len(comp['beat_prev']) > 0:
                t2 = comp['beat_prev'].style.format({
                    'leverage': '{:.1f}x', 'compared_against': '{:.1f}x', 'prob_win': '{:.1%}'
                }).hide(axis='index')
                st.dataframe(t2)
                st.caption("Values well below 50% signal diminishing (or negative) returns from the next increment of leverage.")
            else:
                st.caption("Need at least two leverage levels in the grid to compare adjacent steps.")

            st.markdown("##### Optimal Leverage Frequency (primary robustness metric)")
            t3 = comp['optimal_freq'].style.format({
                'leverage': '{:.1f}x', 'pct_optimal': '{:.1%}'
            }).highlight_max(subset=['pct_optimal'], color='#c6f6d5').hide(axis='index')
            st.dataframe(t3)
            st.caption(f"Sums to {comp['optimal_freq']['pct_optimal'].sum() * 100:.1f}% across all simulations (sanity check).")

            st.markdown("##### Average Rank (1 = best; lower is better)")
            t4 = comp['avg_rank'].style.format({
                'leverage': '{:.1f}x', 'avg_rank': '{:.2f}'
            }).highlight_min(subset=['avg_rank'], color='#c6f6d5').hide(axis='index')
            st.dataframe(t4)

            st.markdown(f"##### Probability of Regret (within {regret_threshold_pct:.0f}% of the best-performing leverage)")
            t5 = comp['regret'].style.format({
                'leverage': '{:.1f}x', 'prob_within_threshold': '{:.1%}'
            }).highlight_max(subset=['prob_within_threshold'], color='#c6f6d5').hide(axis='index')
            st.dataframe(t5)

            show_pairwise = st.checkbox("Show pairwise win matrix (P(row leverage > column leverage))", value=False, key="le_show_pairwise")
            if show_pairwise:
                st.markdown("##### Pairwise Win Matrix")
                pairwise_styled = comp['pairwise_win_matrix'].style.format(
                    "{:.0%}", na_rep="–"
                ).background_gradient(cmap='RdYlGn', axis=None, vmin=0.0, vmax=1.0)
                st.dataframe(pairwise_styled)
    else:
        st.info("Click 'Run Leverage Explorer (Monte Carlo)' in the sidebar to compute the Monte Carlo leverage sweep.")
        st.markdown("#### Interpretation (Historical Only)")
        st.info(generate_leverage_interpretation(hist_opt, None, None, le_ruin_threshold))

with st.expander("📊 Raw Data Preview (filtered)"):
    st.write(f"Total rows: {len(df_filtered)} (out of {len(df_raw)} total)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 10 rows")
        st.dataframe(df_filtered.head(10))
    with col2:
        st.write("Last 10 rows")
        st.dataframe(df_filtered.tail(10))

with st.expander("Show Full Simulation Data"):
    st.dataframe(df_result[['date', 'Mkt_Total', 'Net_Lev_Total', 'Lev_NAV', 'Mkt_NAV', 'Daily_Cost']])