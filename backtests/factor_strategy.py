from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult, performance_statistics


def monthly_event_backtest(
    prices: pd.DataFrame,
    allocator,
    *,
    estimation_window: int = 756,
    transaction_cost_bps: float = 25,
) -> BacktestResult:
    """Signal at month-end, execute next trading day, and drift weights between trades."""
    month_ends = set(
        prices.index.to_series().groupby(prices.index.to_period("M")).max().tolist()
    )
    execution_dates = {}
    for signal_date in sorted(month_ends):
        position = prices.index.get_loc(signal_date)
        if position + 1 < len(prices.index):
            execution_dates[prices.index[position + 1]] = signal_date
    returns = prices.pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)
    costs = pd.Series(0.0, index=prices.index)
    current = pd.Series(0.0, index=prices.columns)
    for position, stamp in enumerate(prices.index):
        if stamp in execution_dates and position >= estimation_window:
            signal_date = execution_dates[stamp]
            signal_position = prices.index.get_loc(signal_date)
            history = prices.iloc[
                max(0, signal_position - estimation_window + 1) : signal_position + 1
            ]
            try:
                target = allocator(history).reindex(prices.columns).fillna(0)
            except (ValueError, RuntimeError):
                target = None
            if target is not None:
                if not np.isclose(target.sum(), 1.0) or (target < 0).any():
                    raise ValueError(
                        "Allocator must return finite long-only weights summing to one"
                    )
                turnover.at[stamp] = float((target - current).abs().sum())
                costs.at[stamp] = turnover.at[stamp] * transaction_cost_bps / 10000
                current = target
        weights.loc[stamp] = current
        gross = float(current @ returns.loc[stamp].fillna(0))
        denominator = 1 + gross
        if denominator > 0:
            current = current * (1 + returns.loc[stamp].fillna(0)) / denominator
    net = (weights * returns).sum(axis=1) - costs
    net = net.iloc[estimation_window:]
    metrics = performance_statistics(net)
    metrics.update(
        {"turnover": float(turnover.sum()), "transaction_cost": float(costs.sum())}
    )
    return BacktestResult(
        net, (1 + net).cumprod(), weights, turnover, costs, metrics, []
    )
