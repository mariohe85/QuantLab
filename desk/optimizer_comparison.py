from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from optimization.engine import optimize_portfolio
from optimization.forecasts import (
    equal_sharpe_expected_returns,
    factor_implied_expected_returns,
    shrink_annualized_expected_returns,
    shrink_factor_premia,
)
from selection.history import _prices_for_build, _simulate, monthly_selection_decisions

from .workflows import _dataset_for_build, _risk_inputs_for_tickers


@dataclass(frozen=True)
class ComparisonMethod:
    key: str
    label: str
    objective: str | None
    expected_return_model: str | None


METHODS = (
    ComparisonMethod("equal_weight", "1/N", None, None),
    ComparisonMethod(
        "raw_mean_max_sharpe",
        "Raw mean max Sharpe",
        "max_sharpe",
        "raw_historical_mean",
    ),
    ComparisonMethod(
        "shrunk_mean_max_sharpe",
        "Shrunk mean max Sharpe",
        "max_sharpe",
        "historical_shrinkage",
    ),
    ComparisonMethod(
        "factor_premium_max_sharpe",
        "Factor premium max Sharpe",
        "max_sharpe",
        "factor_premium",
    ),
    ComparisonMethod("minimum_variance", "Minimum variance", "min_variance", None),
    ComparisonMethod(
        "maximum_diversification",
        "Maximum diversification",
        "max_sharpe",
        "equal_sharpe",
    ),
)


def expected_returns_for_method(
    method: ComparisonMethod,
    inputs: dict,
    parameters: dict,
) -> pd.Series | None:
    if method.expected_return_model is None:
        return None
    if method.expected_return_model == "raw_historical_mean":
        expected = inputs["stock_returns"].mean() * 252
    elif method.expected_return_model == "historical_shrinkage":
        expected = shrink_annualized_expected_returns(
            inputs["stock_returns"]
        ).expected_returns
    elif method.expected_return_model == "factor_premium":
        premia = shrink_factor_premia(
            inputs["factor_returns"],
            parameters.get("factor_premia", {}),
            prior_dispersion=float(parameters.get("premium_prior_dispersion", 0.03)),
        ).premia
        expected = factor_implied_expected_returns(
            inputs["exposures"],
            premia,
            risk_free_rate=float(inputs["risk_free_rate"]),
            standard_errors=inputs["standard_errors"],
            shrink_betas=bool(parameters.get("shrink_factor_betas", False)),
        ).expected_returns
    elif method.expected_return_model == "equal_sharpe":
        expected = equal_sharpe_expected_returns(
            inputs["covariance"],
            risk_free_rate=float(inputs["risk_free_rate"]),
            common_sharpe=float(parameters.get("common_sharpe", 0.5)),
        ).expected_returns
    else:
        raise ValueError(
            f"Unknown expected-return model: {method.expected_return_model}"
        )
    expected = expected.reindex(inputs["names"])
    if expected.isna().any():
        missing = ", ".join(expected[expected.isna()].index)
        raise ValueError(f"Expected returns are unavailable for: {missing}")
    return expected


def _optimizer_kwargs(inputs: dict, parameters: dict) -> dict:
    return {
        "previous_weights": inputs["original"],
        "weight_constraint_mode": parameters.get("weight_constraint_mode", "absolute"),
        "min_weight": float(parameters.get("min_weight", 0)),
        "max_weight": float(parameters.get("max_weight", 0.15)),
        "max_weight_cap": parameters.get("max_weight_cap"),
        "risk_aversion": float(parameters.get("risk_aversion", 1)),
        "risk_free_rate": float(inputs["risk_free_rate"]),
        "turnover_cap": parameters.get("turnover_cap"),
        "turnover_penalty": float(parameters.get("turnover_penalty", 0)),
        "transaction_cost": float(parameters.get("transaction_cost", 0)),
        "factor_exposures": inputs["exposures"],
        "factor_bounds": parameters.get("factor_bounds", {}),
        "relative_factor_bounds": parameters.get("relative_factor_bounds", {}),
        "benchmark_factor_exposures": inputs["spy_exposure"],
        "factor_covariance": inputs["factor_covariance"],
        "max_factor_variance": parameters.get("max_factor_variance"),
        "max_factor_components": parameters.get("max_factor_components"),
        "tracking_error_limit": parameters.get("tracking_error_limit"),
        "benchmark_weights": inputs["original"],
        "specific_variances": inputs["specific"].pow(2),
        "covariance_method": inputs["risk_model"],
    }


def optimized_decisions(
    *,
    build,
    dataset,
    decisions: list[dict],
    model_level: str,
    parameters: dict,
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    paths = {method.key: [] for method in METHODS}
    warnings = {method.key: [] for method in METHODS}
    for decision in decisions:
        names = list(decision["tickers"])
        equal = pd.Series(1 / len(names), index=names)
        paths["equal_weight"].append(dict(decision))
        try:
            risk = _risk_inputs_for_tickers(
                build=build,
                names=names,
                as_of=decision["signal_date"],
                parameters=parameters,
                dataset=dataset,
                model_level=model_level,
            )
            inputs = {
                **risk,
                "names": names,
                "original": equal,
                "holdings": [
                    SimpleNamespace(security=fit.security)
                    for fit in risk["latest"].values()
                ],
            }
        except (KeyError, TypeError, ValueError) as exc:
            for method in METHODS[1:]:
                paths[method.key].append(
                    {**decision, "weights": equal.to_dict(), "status": "fallback_equal"}
                )
                warnings[method.key].append(f"{decision['signal_date']}: {exc}")
            continue

        for method in METHODS[1:]:
            try:
                expected = expected_returns_for_method(method, inputs, parameters)
                result = optimize_portfolio(
                    inputs["covariance"],
                    expected,
                    objective=method.objective,
                    **_optimizer_kwargs(inputs, parameters),
                )
                if not result.success:
                    raise RuntimeError(result.message)
                weights = result.weights
                status = "succeeded"
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                weights = equal
                status = "fallback_equal"
                warnings[method.key].append(f"{decision['signal_date']}: {exc}")
            paths[method.key].append(
                {**decision, "weights": weights.to_dict(), "status": status}
            )
    return paths, warnings


def _summary_row(
    method: ComparisonMethod,
    gross: dict,
    net: dict,
    fallback_count: int,
    equal_sharpe: float,
) -> dict:
    metrics = gross["metrics"]
    net_metrics = net["metrics"]
    return {
        "key": method.key,
        "method": method.label,
        "annual_return": metrics["annual_return"],
        "annual_volatility": metrics["annual_volatility"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "cumulative_return": gross["chart_rows"][-1]["portfolio_equity"] - 1,
        "turnover": metrics["turnover"],
        "net_annual_return": net_metrics["annual_return"],
        "net_sharpe": net_metrics["sharpe"],
        "cost_drag": metrics["annual_return"] - net_metrics["annual_return"],
        "sharpe_vs_1_n": metrics["sharpe"] - equal_sharpe,
        "fallbacks": fallback_count,
    }


def compare_portfolio_optimizers(
    portfolio,
    build,
    *,
    months: int = 24,
    transaction_cost_bps: float = 10,
    parameters: dict | None = None,
) -> dict:
    selection = (portfolio.configuration or {}).get("stock_selection") or {}
    if not selection:
        raise ValueError("The portfolio must contain a Stock Selection configuration")
    model_level = selection.get("model_level", "all_factors")
    default_factor_premia = {
        "Market": 0.05,
        **{
            factor: 0.02
            for factor, weight in selection.get("factor_weights", {}).items()
            if float(weight) != 0 and factor != "Market"
        },
    }
    solve_parameters = {
        "lookback": 756,
        "risk_model": "factor_model",
        "residual_shrinkage": 0.8,
        "min_weight": 0,
        "max_weight": 0.15,
        "weight_constraint_mode": "absolute",
        "factor_premia": default_factor_premia,
        **(parameters or {}),
    }
    decisions = monthly_selection_decisions(
        build,
        {**selection, "build_id": build.pk, "model_level": model_level},
        max_months=months,
    )
    if not decisions:
        raise ValueError("No stored monthly selections are available")
    dataset = _dataset_for_build(build)
    paths, warnings = optimized_decisions(
        build=build,
        dataset=dataset,
        decisions=decisions,
        model_level=model_level,
        parameters=solve_parameters,
    )
    prices = _prices_for_build(build)
    gross_results = {key: _simulate(prices, path) for key, path in paths.items()}
    net_results = {
        key: _simulate(prices, path, transaction_cost_bps=transaction_cost_bps)
        for key, path in paths.items()
    }
    equal_sharpe = gross_results["equal_weight"]["metrics"]["sharpe"]
    rows = [
        _summary_row(
            method,
            gross_results[method.key],
            net_results[method.key],
            sum(
                decision.get("status") == "fallback_equal"
                for decision in paths[method.key]
            ),
            equal_sharpe,
        )
        for method in METHODS
    ]
    return {
        "portfolio": portfolio.name,
        "build_id": build.pk,
        "start": str(gross_results["equal_weight"]["start"]),
        "end": str(gross_results["equal_weight"]["end"]),
        "months": len(decisions),
        "transaction_cost_bps": transaction_cost_bps,
        "rows": sorted(rows, key=lambda row: row["sharpe"], reverse=True),
        "warnings": warnings,
        "methodology_warnings": [
            "The stored universe uses current S&P 500 members, creating survivorship bias.",
            "Replication-mode factor scaling may introduce look-ahead bias.",
            "Only 24 monthly selections are stored, so rankings are statistically fragile.",
        ],
    }
