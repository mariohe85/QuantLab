from __future__ import annotations

import pandas as pd
from django.db.models import Prefetch

from backtests.engine import performance_statistics
from factors.models import FactorBuild, StockExposureSnapshot, StockModelFit
from market_data.store import load_price_dataset

from .engine import composite_scores
from .models import Portfolio
from .stock_selection import parse_ticker_list, resolve_selection_filters


def _prices_for_build(build: FactorBuild | None) -> pd.DataFrame:
    return load_price_dataset().prices.sort_index()


def _simulate(
    prices: pd.DataFrame,
    decisions: list[dict],
    *,
    transaction_cost_bps: float = 0,
) -> dict:
    if "SPY" not in prices:
        raise ValueError("The selected price snapshot does not include SPY")
    execution_decisions = {}
    warnings = []
    for decision in decisions:
        position = prices.index.searchsorted(
            pd.Timestamp(decision["signal_date"]), side="right"
        )
        if position >= len(prices.index):
            continue
        available = [ticker for ticker in decision["tickers"] if ticker in prices]
        missing = sorted(set(decision["tickers"]) - set(available))
        if missing:
            warnings.append(
                f"{decision['signal_date']}: omitted missing prices for {', '.join(missing)}"
            )
        if not available:
            continue
        execution = prices.index[position]
        execution_decisions[execution] = {
            **decision,
            "execution_date": execution,
            "tickers": available,
        }
    if not execution_decisions:
        raise ValueError(
            "No historical rebalance can be aligned with the stored prices"
        )

    asset_returns = prices.pct_change(fill_method=None)
    active_index = prices.index[prices.index >= min(execution_decisions)]
    current = pd.Series(0.0, index=prices.columns)
    portfolio_returns = pd.Series(0.0, index=active_index)
    turnover = pd.Series(0.0, index=active_index)
    costs = pd.Series(0.0, index=active_index)
    rebalances = []
    for stamp in active_index:
        if stamp in execution_decisions:
            decision = execution_decisions[stamp]
            target = pd.Series(0.0, index=prices.columns)
            tickers = decision["tickers"]
            supplied = decision.get("weights") or {}
            if supplied:
                weights = pd.Series(supplied, dtype=float).reindex(tickers).fillna(0)
                total = float(weights.sum())
                if total <= 0:
                    continue
                target.loc[tickers] = weights / total
            else:
                target.loc[tickers] = 1 / len(tickers)
            turnover.at[stamp] = float((target - current).abs().sum())
            costs.at[stamp] = turnover.at[stamp] * transaction_cost_bps / 10000
            current = target
            rebalances.append(
                {
                    **decision,
                    "signal_date": pd.Timestamp(decision["signal_date"]).date(),
                    "execution_date": stamp.date(),
                    "weight": 1 / len(decision["tickers"]),
                    "weights": (
                        {ticker: float(target.at[ticker]) for ticker in decision["tickers"]}
                        if decision.get("weights")
                        else None
                    ),
                    "turnover": turnover.at[stamp],
                }
            )
        daily = asset_returns.loc[stamp].fillna(0)
        gross = float(current @ daily)
        portfolio_returns.at[stamp] = gross - costs.at[stamp]
        if 1 + gross > 0:
            current = current * (1 + daily) / (1 + gross)

    benchmark_returns = asset_returns["SPY"].reindex(active_index).fillna(0)
    portfolio_equity = (1 + portfolio_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    portfolio_drawdown = portfolio_equity / portfolio_equity.cummax() - 1
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1
    metrics = performance_statistics(portfolio_returns, benchmark_returns)
    benchmark_metrics = performance_statistics(benchmark_returns)
    metrics.update(
        {
            "turnover": float(turnover.sum()),
            "transaction_cost": float(costs.sum()),
        }
    )
    chart_rows = [
        {
            "date": stamp.date().isoformat(),
            "portfolio_equity": float(portfolio_equity.at[stamp]),
            "spy_equity": float(benchmark_equity.at[stamp]),
            "portfolio_drawdown": float(portfolio_drawdown.at[stamp]),
            "spy_drawdown": float(benchmark_drawdown.at[stamp]),
        }
        for stamp in active_index
    ]
    return {
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "chart_rows": chart_rows,
        "rebalances": rebalances,
        "warnings": list(dict.fromkeys(warnings)),
        "start": active_index[0].date(),
        "end": active_index[-1].date(),
        "observations": len(active_index),
    }


def monthly_selection_decisions(
    build: FactorBuild, config: dict, *, max_months: int | None = None
) -> list[dict]:
    weights = {
        name: float(value)
        for name, value in config.get("factor_weights", {}).items()
        if float(value) != 0
    }
    if not weights:
        raise ValueError("Select at least one factor to calculate history")
    directions = {
        name: -1 if int(config.get("directions", {}).get(name, 1)) < 0 else 1
        for name in weights
    }
    exposure_query = StockExposureSnapshot.objects.filter(
        factor__name__in=weights,
        factor__model_version=build.definition_version,
    ).select_related("factor")
    fits = (
        StockModelFit.objects.filter(
            build=build,
            model_level=config.get("model_level", "all_factors"),
            security__asset_type="stock",
        )
        .select_related("security")
        .prefetch_related(
            Prefetch("exposures", queryset=exposure_query, to_attr="history_exposures")
        )
        .order_by("period", "security__ticker")
    )
    by_period: dict = {}
    for fit in fits:
        by_period.setdefault(fit.period, []).append(fit)
    if not by_period:
        raise ValueError("No monthly model history is stored for this selection")

    allowed_sectors = {str(value) for value in config.get("sectors", []) if value}
    excluded = set(parse_ticker_list(config.get("excluded_tickers", [])))
    manual = set(parse_ticker_list(config.get("manual_tickers", [])))
    minimum_r2 = config.get("minimum_adjusted_r2")
    trading_days, coverage_floor = resolve_selection_filters(
        minimum_trading_days=config.get("minimum_trading_days"),
        minimum_coverage=config.get("minimum_coverage"),
    )
    top_n = int(config.get("top_n", 20))
    decisions = []
    for period in sorted(by_period):
        period_fits = by_period[period]
        records = {fit.security.ticker: fit for fit in period_fits}
        features = pd.DataFrame.from_dict(
            {
                ticker: {
                    exposure.factor.name: float(exposure.beta)
                    for exposure in fit.history_exposures
                }
                for ticker, fit in records.items()
            },
            orient="index",
        ).reindex(columns=weights, fill_value=0.0)
        scores = composite_scores(features.fillna(0), weights, directions)["composite"]
        eligible = pd.Series(True, index=scores.index)
        for ticker, fit in records.items():
            eligible.at[ticker] = (
                ticker not in excluded
                and (not allowed_sectors or fit.security.sector in allowed_sectors)
                and (not trading_days or fit.observation_count >= trading_days)
                and (
                    coverage_floor is None or fit.coverage >= coverage_floor
                )
                and (minimum_r2 is None or fit.adjusted_r2 >= float(minimum_r2))
            )
        model_selected = list(
            scores[eligible].nlargest(min(top_n, int(eligible.sum()))).index
        )
        manual_selected = sorted(manual - set(model_selected))
        selected = [*model_selected, *manual_selected]
        decisions.append(
            {
                "signal_date": max(fit.as_of for fit in period_fits),
                "tickers": selected,
                "model_count": len(model_selected),
                "manual_count": len(manual_selected),
            }
        )
    if max_months is not None:
        decisions = decisions[-int(max_months) :]
    return decisions


def replay_screen_history(build: FactorBuild, config: dict) -> dict:
    decisions = monthly_selection_decisions(build, config)
    result = _simulate(_prices_for_build(build), decisions)
    result["methodology"] = (
        "Fast stored-exposure replay. Monthly model selections execute on the next "
        "trading day and reset to equal weight; results are gross of transaction costs."
    )
    result["warnings"].append(
        "Current-universe membership and replication-mode factor scaling may introduce "
        "survivorship or look-ahead bias; use a walk-forward backtest for rigorous results."
    )
    return result


def replay_portfolio_history(
    portfolio: Portfolio,
    build: FactorBuild | None = None,
) -> dict:
    selection = portfolio.configuration.get("stock_selection", {})
    if selection:
        selected_build = FactorBuild.objects.get(
            pk=selection.get("factor_build_id") or build.pk
        )
        return replay_screen_history(
            selected_build,
            {
                **selection,
                "build_id": selected_build.pk,
            },
        )

    tickers = list(
        portfolio.holdings.select_related("security")
        .order_by("security__ticker")
        .values_list("security__ticker", flat=True)
    )
    if not tickers:
        raise ValueError("The portfolio has no holdings")
    prices = _prices_for_build(build)
    month_end_dates = (
        prices.index.to_series().groupby(prices.index.to_period("M")).max().tolist()
    )
    result = _simulate(
        prices,
        [{"signal_date": stamp, "tickers": tickers} for stamp in month_end_dates],
    )
    result["methodology"] = (
        "Fixed holdings reset to equal weight monthly on the trading day after month-end; "
        "results are gross of transaction costs."
    )
    return result
