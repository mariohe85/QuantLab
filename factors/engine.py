from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler


@dataclass
class ExposureResult:
    beta: pd.DataFrame
    inference: dict[str, pd.DataFrame]
    selected: list[str]
    adjusted_r2: float
    residual_vol: float
    observation_count: int
    inference_method: str


def _basket_return(returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    available = [ticker for ticker in tickers if ticker in returns]
    if not available:
        raise ValueError(f"No available tickers from {tickers}")
    return returns[available].mean(axis=1)


def construct_spreads(
    prices: pd.DataFrame, definitions: dict[str, dict[str, list[str]]]
) -> pd.DataFrame:
    """Construct arithmetic long-minus-short returns from adjusted prices."""
    returns = prices.pct_change(fill_method=None)
    factors = {}
    for name, definition in definitions.items():
        long = definition.get("long", [])
        long_leg = _basket_return(returns, long) if long else 0.0
        short = definition.get("short", [])
        factors[name] = long_leg - (_basket_return(returns, short) if short else 0.0)
    return pd.DataFrame(factors).dropna(how="all")


def rolling_purify(
    factor_returns: pd.DataFrame,
    dependencies: dict[str, list[str]],
    window: int = 126,
    min_periods: int = 63,
) -> pd.DataFrame:
    purified = factor_returns.copy()
    for target, parents in dependencies.items():
        valid_parents = [parent for parent in parents if parent in factor_returns]
        if target not in factor_returns or not valid_parents:
            continue
        target_returns = factor_returns[target]
        residuals = pd.Series(index=target_returns.index, dtype=float)
        for stop in range(min_periods, len(target_returns) + 1):
            start = max(0, stop - window)
            frame = pd.concat(
                [
                    target_returns.iloc[start:stop],
                    factor_returns[valid_parents].iloc[start:stop],
                ],
                axis=1,
            ).dropna()
            if len(frame) < min_periods:
                continue
            coefficients = np.linalg.lstsq(
                np.column_stack([np.ones(len(frame)), frame[valid_parents].values]),
                frame[target].values,
                rcond=None,
            )[0]
            row = factor_returns[valid_parents].iloc[stop - 1]
            if row.notna().all() and pd.notna(target_returns.iloc[stop - 1]):
                residuals.iloc[stop - 1] = (
                    target_returns.iloc[stop - 1]
                    - np.r_[1.0, row.values] @ coefficients
                )
        purified[target] = residuals
    return purified


def volatility_scale(
    returns: pd.DataFrame,
    window: int = 60,
    annual_target: float = 0.10,
    scale_cap: float = 5.0,
) -> pd.DataFrame:
    volatility = returns.rolling(window, min_periods=max(20, window // 2)).std().shift(
        1
    ) * np.sqrt(252)
    scale = (annual_target / volatility).clip(upper=scale_cap)
    return returns * scale


FACTOR_HORIZONS = (1, 3, 5, 10, 21, 63, 126, 252)


def horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    return (1 + returns).rolling(horizon).apply(np.prod, raw=True) - 1


def rolling_zscores(returns: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    cumulative = returns.rolling(horizon).sum()
    mean = cumulative.rolling(252, min_periods=horizon).mean()
    std = cumulative.rolling(252, min_periods=horizon).std()
    return (cumulative - mean) / std.replace(0, np.nan)


def estimate_exposure(
    stock_returns: pd.Series,
    factor_returns: pd.DataFrame,
    alpha: float = 0.0001,
    l1_ratio: float = 0.5,
    hac_lags: int = 5,
    min_obs: int = 60,
    selection_mode: str = "elastic_net",
) -> ExposureResult:
    frame = pd.concat([stock_returns.rename("stock"), factor_returns], axis=1).dropna()
    if len(frame) < min_obs:
        raise ValueError(f"At least {min_obs} complete observations are required")
    features = frame[factor_returns.columns]
    if selection_mode not in {"elastic_net", "fixed"}:
        raise ValueError("selection_mode must be elastic_net or fixed")
    standardized = StandardScaler().fit_transform(features)
    if selection_mode == "fixed":
        selected = list(features.columns)
    elif len(frame) >= max(30, len(features.columns) * 3):
        selector = ElasticNetCV(
            l1_ratio=[0.25, 0.5, 0.75, 0.95],
            cv=min(5, max(2, len(frame) // 20)),
            max_iter=10000,
            n_jobs=1,
        ).fit(standardized, frame["stock"])
        selected = list(features.columns[np.abs(selector.coef_) > 1e-10])
    else:
        selector = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000).fit(
            standardized, frame["stock"]
        )
        selected = list(features.columns[np.abs(selector.coef_) > 1e-10])
    if not selected:
        selected = [
            features.columns[
                int(np.argmax(np.abs(features.corrwith(frame["stock"]).values)))
            ]
        ]
    design = sm.add_constant(features[selected], has_constant="add")
    fit = sm.OLS(frame["stock"], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    confidence = fit.conf_int()
    inference = pd.DataFrame(
        {
            "beta": fit.params,
            "se": fit.bse,
            "t_stat": fit.tvalues,
            "p_value": fit.pvalues,
            "ci_low": confidence[0],
            "ci_high": confidence[1],
        }
    )
    beta = pd.DataFrame(
        [fit.params.drop("const")], index=[stock_returns.name or "stock"]
    )
    label = (
        "Fixed-model OLS/HAC"
        if selection_mode == "fixed"
        else "ElasticNetCV selection; conditional OLS/HAC"
    )
    return ExposureResult(
        beta,
        {"hac": inference},
        selected,
        float(fit.rsquared_adj),
        float(fit.resid.std() * np.sqrt(252)),
        int(fit.nobs),
        label,
    )


def rolling_exposures(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: int = 252,
) -> dict[str, ExposureResult]:
    return {
        ticker: estimate_exposure(
            stock_returns[ticker].iloc[-window:], factor_returns.iloc[-window:]
        )
        for ticker in stock_returns
    }


def factor_covariance(
    factor_returns: pd.DataFrame, annualize: int = 252
) -> pd.DataFrame:
    return factor_returns.cov().fillna(0) * annualize


def portfolio_decomposition(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_vol: pd.Series,
    expected_factor_returns: pd.Series | None = None,
) -> dict:
    aligned = exposures.reindex(index=weights.index, columns=factor_cov.index).fillna(0)
    beta = aligned.T @ weights
    factor_marginal = factor_cov @ beta
    factor_components = beta * factor_marginal
    factor_variance = float(beta @ factor_marginal)
    specific_variance = float(
        np.sum((weights * specific_vol.reindex(weights.index).fillna(0)) ** 2)
    )
    total = factor_variance + specific_variance
    volatility = float(np.sqrt(max(total, 0)))
    # Volatility is homogeneous of degree one in the weights, so Euler's theorem
    # makes these contributions add up to the portfolio volatility itself rather
    # than to its square. Reporting risk that way keeps every bar on the same
    # scale as the headline number instead of on a squared-percent scale.
    divisor = volatility if volatility > 0 else np.nan
    factor_contribution = factor_components / divisor
    expected = (
        float(beta @ expected_factor_returns.reindex(beta.index).fillna(0))
        if expected_factor_returns is not None
        else None
    )
    return {
        "exposure": beta,
        "factor_marginal": factor_marginal,
        "factor_components": factor_components,
        "factor_marginal_risk": factor_marginal / divisor,
        "factor_contribution": factor_contribution,
        "specific_contribution": specific_variance / divisor,
        "risk_percent": factor_contribution / divisor,
        "specific_risk_percent": specific_variance / total if total else np.nan,
        "factor_percent": (
            factor_components / factor_variance
            if factor_variance
            else factor_components * np.nan
        ),
        "factor_variance": factor_variance,
        "specific_variance": specific_variance,
        "predicted_variance": total,
        "predicted_volatility": volatility,
        "expected_factor_return": expected,
    }


def return_attribution(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_returns: pd.Series,
    realized_asset_returns: pd.Series,
) -> dict:
    aligned_weights = weights.reindex(exposures.index).fillna(0)
    portfolio_beta = exposures.T @ aligned_weights
    factor_contributions = (
        portfolio_beta.reindex(factor_returns.index).fillna(0) * factor_returns
    )
    realized_return = float(
        aligned_weights
        @ realized_asset_returns.reindex(aligned_weights.index).fillna(0)
    )
    factor_return = float(factor_contributions.sum())
    return {
        "realized_return": realized_return,
        "factor_contributions": factor_contributions,
        "factor_return": factor_return,
        "residual_contribution": realized_return - factor_return,
    }
