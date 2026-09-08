from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, skew, kurtosis, spearmanr


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    metrics: dict
    warnings: list[str]


def performance_statistics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    periods: int = 252,
    bootstrap_samples: int = 500,
    seed: int = 7,
) -> dict:
    r = returns.dropna()
    if len(r) < 20:
        raise ValueError("At least 20 observations are required")
    annual_return = float((1 + r).prod() ** (periods / len(r)) - 1)
    annual_vol = float(r.std() * np.sqrt(periods))
    downside = float(r[r < 0].std() * np.sqrt(periods))
    arithmetic_annual_return = float(r.mean() * periods)
    sharpe = arithmetic_annual_return / annual_vol if annual_vol > 0 else np.nan
    sortino = annual_return / downside if downside > 0 else np.nan
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    hac = sm.OLS(r.values, np.ones((len(r), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(5, len(r) // 5)}
    )
    rng = np.random.default_rng(seed)
    block = max(2, int(np.sqrt(len(r))))
    means = []
    values = r.values
    for _ in range(bootstrap_samples):
        starts = rng.integers(0, max(1, len(values) - block + 1), int(np.ceil(len(values) / block)))
        sample = np.concatenate([values[s:s + block] for s in starts])[: len(values)]
        means.append(sample.mean() * periods)
    sr_daily = r.mean() / r.std() if r.std() > 0 else np.nan
    sr_se = np.sqrt(max((1 - skew(r) * sr_daily + (kurtosis(r, fisher=False) - 1) * sr_daily**2 / 4) / len(r), 1e-16))
    probabilistic_sharpe = float(norm.cdf(sr_daily / sr_se)) if np.isfinite(sr_daily) else np.nan
    metrics = {
        "annual_return": annual_return,
        "arithmetic_annual_return": arithmetic_annual_return,
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
        "max_drawdown": max_drawdown,
        "daily_var_95_loss": float(-r.quantile(0.05)),
        "daily_cvar_95_loss": float(-r[r <= r.quantile(0.05)].mean()),
        "autocorrelation_1": float(r.autocorr(1)),
        "hac_mean_t_stat": float(hac.tvalues[0]),
        "hac_mean_p_value": float(hac.pvalues[0]),
        "bootstrap_annual_mean_ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "probabilistic_sharpe": probabilistic_sharpe,
    }
    if benchmark is not None:
        aligned = pd.concat([r, benchmark.rename("benchmark")], axis=1).dropna()
        excess = aligned.iloc[:, 0] - aligned["benchmark"]
        beta = float(aligned.iloc[:, 0].cov(aligned["benchmark"]) / aligned["benchmark"].var())
        alpha = float((aligned.iloc[:, 0].mean() - beta * aligned["benchmark"].mean()) * periods)
        tracking = float(excess.std() * np.sqrt(periods))
        metrics.update({
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking,
            "information_ratio": float(excess.mean() * periods / tracking) if tracking > 0 else np.nan,
        })
    return metrics


def selection_statistics(
    score_panel: pd.DataFrame,
    forward_return_panel: pd.DataFrame,
    quantiles: int = 5,
) -> dict:
    common_dates = score_panel.index.intersection(forward_return_panel.index)
    ics, spreads, hits = [], [], []
    for date in common_dates:
        joined = pd.concat([score_panel.loc[date], forward_return_panel.loc[date]], axis=1).dropna()
        if len(joined) < quantiles * 2:
            continue
        ics.append(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
        groups = pd.qcut(joined.iloc[:, 0].rank(method="first"), quantiles, labels=False)
        spread = joined.loc[groups == quantiles - 1].iloc[:, 1].mean() - joined.loc[groups == 0].iloc[:, 1].mean()
        spreads.append(spread)
        hits.append(spread > 0)
    ic = pd.Series(ics, dtype=float)
    return {
        "rank_ic": float(ic.mean()) if len(ic) else np.nan,
        "icir": float(ic.mean() / ic.std()) if len(ic) and ic.std() > 0 else np.nan,
        "top_minus_bottom": float(np.mean(spreads)) if spreads else np.nan,
        "hit_rate": float(np.mean(hits)) if hits else np.nan,
        "observations": len(ics),
    }


def screen_backtest_statistics(
    score_panel: pd.DataFrame,
    forward_return_panel: pd.DataFrame,
    selections: pd.DataFrame | None = None,
    transaction_cost_bps: float = 25,
    bootstrap_samples: int = 500,
    seed: int = 7,
) -> dict:
    base = selection_statistics(score_panel, forward_return_panel)
    spreads = []
    quantile_returns: dict[int, list[float]] = {number: [] for number in range(5)}
    for stamp in score_panel.index.intersection(forward_return_panel.index):
        joined = pd.concat([score_panel.loc[stamp], forward_return_panel.loc[stamp]], axis=1).dropna()
        if len(joined) < 10:
            continue
        groups = pd.qcut(joined.iloc[:, 0].rank(method="first"), 5, labels=False)
        values = joined.iloc[:, 1].groupby(groups).mean()
        for group, value in values.items():
            quantile_returns[int(group)].append(float(value))
        spreads.append(float(values.iloc[-1] - values.iloc[0]))
    spread_series = pd.Series(spreads, dtype=float)
    if len(spread_series) >= 3:
        hac = sm.OLS(spread_series, np.ones((len(spread_series), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(3, len(spread_series) // 3)}
        )
        base["spread_hac_t"] = float(hac.tvalues.iloc[0])
        base["spread_hac_p"] = float(hac.pvalues.iloc[0])
        rng = np.random.default_rng(seed)
        values = spread_series.to_numpy()
        block = max(2, int(np.sqrt(len(values))))
        means = []
        for _ in range(bootstrap_samples):
            starts = rng.integers(
                0,
                max(1, len(values) - block + 1),
                int(np.ceil(len(values) / block)),
            )
            sample = np.concatenate([values[start:start + block] for start in starts])[
                : len(values)
            ]
            means.append(float(sample.mean()))
        base["spread_bootstrap_ci"] = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    else:
        base.update({"spread_hac_t": np.nan, "spread_hac_p": np.nan, "spread_bootstrap_ci": [np.nan, np.nan]})
    base["quantile_returns"] = {
        str(group + 1): float(np.mean(values)) if values else np.nan
        for group, values in quantile_returns.items()
    }
    if selections is not None and len(selections) > 1:
        changes = selections.astype(bool).astype(int).diff().abs().sum(axis=1) / 2
        base["turnover"] = float(changes.mean())
        base["estimated_cost"] = float(changes.sum() * transaction_cost_bps / 10000)
    return base


def walk_forward(
    prices: pd.DataFrame,
    allocator: Callable[[pd.DataFrame], pd.Series],
    estimation_window: int = 126,
    rebalance_every: int = 21,
    execution_lag: int = 1,
    transaction_cost_bps: float = 5.0,
    benchmark_returns: pd.Series | None = None,
) -> BacktestResult:
    if execution_lag < 1:
        raise ValueError("Execution lag must be at least one day to prevent lookahead")
    prices = prices.sort_index()
    asset_returns = prices.pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    turnovers = pd.Series(0.0, index=prices.index)
    costs = pd.Series(0.0, index=prices.index)
    pending: dict[int, pd.Series] = {}
    for i in range(estimation_window, len(prices)):
        if (i - estimation_window) % rebalance_every == 0:
            history = prices.iloc[i - estimation_window:i].copy()
            target = allocator(history).reindex(prices.columns).fillna(0)
            if not np.isfinite(target).all():
                raise ValueError("Allocator produced non-finite weights")
            if (target < -1e-10).any():
                raise ValueError("Allocator produced negative weights for a long-only backtest")
            if not np.isclose(target.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Allocator weights must sum to one, got {target.sum():.8f}")
            execution_index = i + execution_lag - 1
            if execution_index < len(prices):
                pending[execution_index] = target
        if i in pending:
            target = pending.pop(i).reindex(prices.columns).fillna(0)
            turnover = float((target - current).abs().sum())
            current = target
            turnovers.iloc[i] = turnover
            costs.iloc[i] = turnover * transaction_cost_bps / 10000
        weights.iloc[i] = current
    gross = (weights * asset_returns).sum(axis=1)
    net = (gross - costs).iloc[estimation_window:]
    warnings = []
    if len(prices.columns) > 0:
        warnings.append("Universe membership must be point-in-time to avoid survivorship bias.")
    metrics = performance_statistics(net, benchmark_returns)
    metrics["turnover"] = float(turnovers.sum())
    metrics["transaction_cost"] = float(costs.sum())
    if len(net) < 3 * 252:
        warnings.append("Deflated Sharpe is unreliable with short history; probabilistic Sharpe is shown.")
    return BacktestResult(net, (1 + net).cumprod(), weights, turnovers, costs, metrics, warnings)
