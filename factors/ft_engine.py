"""Numerical engine for the local V2 factor construction pipeline.

Implements:
  1. Daily raw basket/spread returns
  2. Weekly (Friday close-to-close) resampling via compounded daily returns
  3. 156-week rolling OLS (with intercept) of the basket on stripping factors
  4. Beta upsampling: Friday betas forward-filled to daily; first valid beta
     back-filled through the warmup period
  5. In-sample daily residual: r_pure[t] = r_basket[t] - X[t] . beta[t]
     (intercept estimated but NOT subtracted from the residual)
  6. Volatility scaling: L[t] = min(TARGET_VOL / vol[t-1], MAX_LEV) with
     vol = 60-day rolling sample std (ddof=1) * sqrt(252);
     r_scaled[t] = L[t] * r_pure[t]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

TARGET_VOL = 0.10
MAX_LEVERAGE = 5.0
VOL_LOOKBACK = 60
TRADING_DAYS = 252
WINDOW_WEEKS = 156


def to_weekly(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound daily returns into Friday-ending weekly returns."""
    return (1.0 + daily_returns).resample("W-FRI").prod() - 1.0


def vol_scale(
    r_pure: pd.Series,
    target_vol: float = TARGET_VOL,
    lookback: int = VOL_LOOKBACK,
    max_lev: float = MAX_LEVERAGE,
) -> pd.Series:
    """Scale a pure return series to the target annualized volatility.

    Uses prior-day 60d sample std (ddof=1), annualized by sqrt(252),
    leverage capped at max_lev.
    """
    vol = r_pure.rolling(lookback).std(ddof=1) * np.sqrt(TRADING_DAYS)
    lev = (target_vol / vol.shift(1)).clip(upper=max_lev)
    return (lev * r_pure).dropna()


def rolling_strip(
    r_basket_daily: pd.Series,
    X_daily: pd.DataFrame,
    window_weeks: int = WINDOW_WEEKS,
    beta_timing: str = "same_friday",
) -> tuple[pd.Series, pd.DataFrame]:
    """Strip factor exposures via 156-week rolling OLS on weekly returns.

    Parameters
    ----------
    r_basket_daily : daily returns of the raw basket/spread (dependent variable)
    X_daily        : daily returns of the stripping factors (regressors)
    beta_timing    : 'same_friday'  -> beta fit on weeks [t-155, t] applies to
                                       daily dates (prev Friday, t] (ffill of
                                       Friday-indexed betas)  [documented conv.]
                     'next_week'    -> Friday beta applies to the FOLLOWING
                                       Mon..Fri (shift(1) then ffill)

    Returns (r_pure_daily, betas_daily).
    """
    df = pd.concat([r_basket_daily.rename("_y"), X_daily], axis=1, sort=True).dropna()
    y_d, X_d = df["_y"], df.drop(columns="_y")

    y_w = to_weekly(y_d).dropna()
    X_w = to_weekly(X_d).dropna()
    W = pd.concat([y_w.rename("_y"), X_w], axis=1, sort=True).dropna()

    exog = sm.add_constant(W.drop(columns="_y"))
    n = len(W)
    if n >= window_weeks:
        res = RollingOLS(W["_y"], exog, window=window_weeks).fit(params_only=True)
        betas_w = res.params.drop(columns="const")
    else:  # static OLS fallback per the docs
        ols = sm.OLS(W["_y"], exog).fit()
        betas_w = (
            pd.DataFrame([ols.params.drop("const")], index=[W.index[-1]])
            .reindex(W.index)
            .bfill()
        )

    if beta_timing == "next_week":
        betas_w = betas_w.shift(1)

    # Upsample to daily: ffill Friday betas, then bfill first valid beta (warmup)
    betas_d = betas_w.reindex(betas_w.index.union(y_d.index)).ffill().bfill()
    betas_d = betas_d.reindex(y_d.index)

    hedge = (betas_d[X_d.columns] * X_d).sum(axis=1)
    r_pure = (y_d - hedge).rename("r_pure")
    return r_pure, betas_d


def build_factor(
    r_basket_daily: pd.Series,
    X_daily: pd.DataFrame | None = None,
    beta_timing: str = "same_friday",
) -> pd.Series:
    """Full pipeline: (optional) rolling strip, then volatility scaling."""
    if X_daily is None or X_daily.shape[1] == 0:
        r_pure = r_basket_daily.dropna()
    else:
        r_pure, _ = rolling_strip(r_basket_daily, X_daily, beta_timing=beta_timing)
    return vol_scale(r_pure)
