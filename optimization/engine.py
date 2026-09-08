from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


@dataclass
class OptimizationResult:
    weights: pd.Series
    success: bool
    message: str
    diagnostics: dict


def estimate_covariance(
    returns: pd.DataFrame, method: str = "ledoit_wolf", annualize: int = 252
) -> pd.DataFrame:
    clean = returns.dropna()
    if len(clean) < 2:
        raise ValueError("At least two complete observations are required")
    if method == "sample":
        return clean.cov() * annualize
    if method == "diagonal":
        return pd.DataFrame(
            np.diag(clean.var().values * annualize),
            index=clean.columns,
            columns=clean.columns,
        )
    if method != "ledoit_wolf":
        raise ValueError(f"Unknown covariance estimator: {method}")
    matrix = LedoitWolf().fit(clean.values).covariance_ * annualize
    return pd.DataFrame(matrix, index=clean.columns, columns=clean.columns)


def nearest_psd(matrix: pd.DataFrame, floor: float = 1e-10) -> pd.DataFrame:
    symmetric = (matrix.values + matrix.values.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    repaired = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
    return pd.DataFrame(repaired, index=matrix.index, columns=matrix.columns)


def factor_risk_covariance(
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    residual_volatility: pd.Series,
    residual_correlation: pd.DataFrame | None = None,
    shrinkage: float = 0.8,
) -> pd.DataFrame:
    factors = exposures.columns.intersection(factor_cov.index)
    x = exposures[factors].fillna(0)
    omega = nearest_psd(factor_cov.reindex(index=factors, columns=factors))
    specific_var = (
        residual_volatility.reindex(x.index).fillna(residual_volatility.median()).pow(2)
    )
    diagonal = np.diag(specific_var.values)
    if residual_correlation is not None:
        corr = (
            residual_correlation.reindex(index=x.index, columns=x.index)
            .fillna(0)
            .values
        )
        sigma = np.sqrt(specific_var.values)
        empirical = corr * np.outer(sigma, sigma)
        specific = shrinkage * diagonal + (1 - shrinkage) * empirical
    else:
        specific = diagonal
    total = x.values @ omega.values @ x.values.T + specific
    return nearest_psd(pd.DataFrame(total, index=x.index, columns=x.index))


def _risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    return weights * marginal / max(variance, 1e-16)


def optimize_portfolio(
    covariance: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    objective: str = "min_variance",
    previous_weights: pd.Series | None = None,
    weight_constraint_mode: str = "absolute",
    min_weight: float = 0.0,
    max_weight: float = 0.15,
    max_weight_cap: float | None = None,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    turnover_cap: float | None = None,
    turnover_penalty: float = 0.0,
    transaction_cost: float = 0.0,
    sectors: pd.Series | None = None,
    sector_bounds: dict[str, tuple[float, float]] | None = None,
    factor_exposures: pd.DataFrame | None = None,
    factor_bounds: dict[str, tuple[float, float]] | None = None,
    benchmark_weights: pd.Series | None = None,
    relative_factor_bounds: dict[str, tuple[float, float]] | None = None,
    benchmark_factor_exposures: pd.Series | None = None,
    factor_covariance: pd.DataFrame | None = None,
    max_factor_variance: float | None = None,
    max_factor_components: dict[str, float] | None = None,
    risk_budgets: pd.Series | None = None,
    covariance_method: str = "provided",
    tracking_error_limit: float | None = None,
    specific_variances: pd.Series | None = None,
) -> OptimizationResult:
    names = list(covariance.index)
    cov = covariance.reindex(index=names, columns=names).values
    n = len(names)
    if n == 0:
        raise ValueError("Covariance matrix is empty")
    if (
        objective in {"max_return", "max_sharpe", "mean_variance"}
        and expected_returns is None
    ):
        raise ValueError(f"{objective} requires an explicit expected return model")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    mu = (
        (
            expected_returns
            if expected_returns is not None
            else pd.Series(0.0, index=names)
        )
        .reindex(names)
        .fillna(0)
        .values
    )
    prev = (
        (
            previous_weights
            if previous_weights is not None
            else pd.Series(1 / n, index=names)
        )
        .reindex(names)
        .fillna(0)
        .values
    )
    if weight_constraint_mode == "absolute":
        lower_bounds = np.full(n, min_weight)
        upper_bounds = np.full(n, max_weight)
    elif weight_constraint_mode == "relative":
        if min_weight < 0 or max_weight < 0:
            raise ValueError("Relative decrease and increase must be non-negative")
        lower_bounds = np.maximum(0.0, prev - min_weight)
        upper_bounds = np.minimum(1.0, prev + max_weight)
        if max_weight_cap is not None:
            upper_bounds = np.minimum(upper_bounds, max_weight_cap)
    else:
        raise ValueError(f"Unknown weight constraint mode: {weight_constraint_mode}")
    if (
        not np.isfinite(lower_bounds).all()
        or not np.isfinite(upper_bounds).all()
        or (lower_bounds < 0).any()
        or (upper_bounds > 1).any()
        or (lower_bounds > upper_bounds).any()
    ):
        raise ValueError("Stock weight bounds must be finite and between zero and one")
    if lower_bounds.sum() > 1 + 1e-9 or upper_bounds.sum() < 1 - 1e-9:
        raise ValueError(
            "Stock weight bounds cannot produce a fully invested portfolio"
        )
    bounds = list(zip(lower_bounds, upper_bounds, strict=True))
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if turnover_cap is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, cap=turnover_cap: cap - np.sum(np.abs(w - prev)),
            }
        )
    if tracking_error_limit is not None and benchmark_factor_exposures is None:
        benchmark = (
            benchmark_weights.reindex(names).fillna(0).values
            if benchmark_weights is not None
            else prev
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, b=benchmark, cap=tracking_error_limit: float(
                    cap**2 - (w - b) @ cov @ (w - b)
                ),
            }
        )
    if sectors is not None:
        aligned_sectors = sectors.reindex(names)
        for sector, (lower, upper) in (sector_bounds or {}).items():
            mask = (aligned_sectors == sector).astype(float).values
            constraints += [
                {"type": "ineq", "fun": lambda w, m=mask, lo=lower: float(w @ m - lo)},
                {"type": "ineq", "fun": lambda w, m=mask, hi=upper: float(hi - w @ m)},
            ]
    x = (
        factor_exposures.reindex(index=names).fillna(0)
        if factor_exposures is not None
        else None
    )
    if (
        tracking_error_limit is not None
        and benchmark_factor_exposures is not None
        and x is not None
        and factor_covariance is not None
    ):
        factors = [factor for factor in x.columns if factor in factor_covariance]
        xf = x[factors].values
        omega = factor_covariance.reindex(index=factors, columns=factors).values
        benchmark_beta = benchmark_factor_exposures.reindex(factors).fillna(0).values
        specific_var = (
            specific_variances.reindex(names).fillna(0).values
            if specific_variances is not None
            else np.zeros(n)
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(
                    tracking_error_limit**2
                    - (
                        (xf.T @ w - benchmark_beta)
                        @ omega
                        @ (xf.T @ w - benchmark_beta)
                        + np.sum(w**2 * specific_var)
                    )
                ),
            }
        )
    for factor, (lower, upper) in (factor_bounds or {}).items():
        if x is not None and factor in x:
            exposure = x[factor].values
            constraints += [
                {
                    "type": "ineq",
                    "fun": lambda w, e=exposure, lo=lower: float(w @ e - lo),
                },
                {
                    "type": "ineq",
                    "fun": lambda w, e=exposure, hi=upper: float(hi - w @ e),
                },
            ]
    if x is not None and benchmark_weights is not None:
        benchmark = benchmark_weights.reindex(names).fillna(0).values
        for factor, (lower, upper) in (relative_factor_bounds or {}).items():
            if factor in x:
                exposure = x[factor].values
                base = (
                    float(benchmark_factor_exposures.get(factor, 0))
                    if benchmark_factor_exposures is not None
                    else float(benchmark @ exposure)
                )
                constraints += [
                    {
                        "type": "ineq",
                        "fun": lambda w, e=exposure, lo=lower, b=base: float(
                            w @ e - b - lo
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda w, e=exposure, hi=upper, b=base: float(
                            hi - (w @ e - b)
                        ),
                    },
                ]
    if (
        x is not None
        and factor_covariance is not None
        and max_factor_variance is not None
    ):
        factors = [c for c in x.columns if c in factor_covariance]
        xf = x[factors].values
        fcov = factor_covariance.reindex(index=factors, columns=factors).values
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(
                    max_factor_variance - (xf.T @ w) @ fcov @ (xf.T @ w)
                ),
            }
        )
    if x is not None and factor_covariance is not None and max_factor_components:
        factors = [c for c in x.columns if c in factor_covariance]
        xf = x[factors].values
        fcov = factor_covariance.reindex(index=factors, columns=factors).values
        for factor, maximum in max_factor_components.items():
            if factor in factors:
                position = factors.index(factor)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, p=position, cap=maximum: float(
                            cap - abs((xf.T @ w)[p] * (fcov @ (xf.T @ w))[p])
                        ),
                    }
                )

    def loss(w: np.ndarray) -> float:
        variance = float(w @ cov @ w)
        turnover = float(np.sum(np.abs(w - prev)))
        costs = (turnover_penalty + transaction_cost) * turnover
        if objective == "equal_weight":
            return float(np.sum((w - 1 / n) ** 2) + costs)
        if objective == "max_return":
            return -float(mu @ w) + costs
        if objective in {"mean_variance", "factor_score_utility"}:
            return risk_aversion * variance / 2 - float(mu @ w) + costs
        if objective == "max_sharpe":
            excess_return = float(mu @ w) - risk_free_rate
            return -excess_return / np.sqrt(max(variance, 1e-16)) + costs
        if objective in {"risk_parity", "risk_budget"}:
            if (
                objective == "risk_budget"
                and x is not None
                and factor_covariance is not None
                and risk_budgets is not None
                and set(risk_budgets.index).issubset(set(x.columns))
            ):
                factors = [name for name in x.columns if name in factor_covariance]
                beta = x[factors].values.T @ w
                omega = factor_covariance.reindex(index=factors, columns=factors).values
                components = beta * (omega @ beta)
                shares = components / max(float(components.sum()), 1e-16)
                target = risk_budgets.reindex(factors).fillna(0).values
                target = target / max(target.sum(), 1e-16)
                return float(np.sum((shares - target) ** 2) + costs)
            budgets = (
                risk_budgets.reindex(names).fillna(0).values
                if risk_budgets is not None
                else np.full(n, 1 / n)
            )
            budgets = budgets / budgets.sum()
            return float(np.sum((_risk_contributions(w, cov) - budgets) ** 2) + costs)
        if objective == "max_diversification":
            standalone = np.sqrt(np.maximum(np.diag(cov), 0))
            return -float(standalone @ w) / np.sqrt(max(variance, 1e-16)) + costs
        if objective != "min_variance":
            raise ValueError(f"Unknown objective: {objective}")
        return variance + costs

    initial = np.clip(prev, lower_bounds, upper_bounds)
    for _ in range(n * 2):
        difference = 1.0 - initial.sum()
        if abs(difference) <= 1e-12:
            break
        room = upper_bounds - initial if difference > 0 else initial - lower_bounds
        available = room > 1e-12
        if not available.any():
            break
        change = min(abs(difference) / available.sum(), float(room[available].min()))
        initial[available] += np.sign(difference) * change
    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weights = pd.Series(result.x, index=names)
    exposures = x.T @ weights if x is not None else pd.Series(dtype=float)
    sector_weights = (
        weights.groupby(sectors.reindex(names)).sum()
        if sectors is not None
        else pd.Series(dtype=float)
    )
    violations = []
    for i, constraint in enumerate(constraints):
        value = float(constraint["fun"](weights.values))
        if (constraint["type"] == "eq" and abs(value) > 1e-5) or (
            constraint["type"] == "ineq" and value < -1e-5
        ):
            violations.append({"constraint": i, "slack": value})
    for position, (name, weight) in enumerate(weights.items()):
        lower = lower_bounds[position]
        upper = upper_bounds[position]
        if weight < lower - 1e-7 or weight > upper + 1e-7:
            violations.append(
                {
                    "constraint": f"bound:{name}",
                    "slack": float(min(weight - lower, upper - weight)),
                }
            )
    variance = float(weights @ covariance @ weights)
    constraint_slack = {
        "fully_invested": float(1e-5 - abs(weights.sum() - 1)),
        "stock_weight_bounds": {
            name: {
                "minimum": float(weights[name] - lower_bounds[position]),
                "maximum": float(upper_bounds[position] - weights[name]),
            }
            for position, name in enumerate(names)
        },
    }
    if turnover_cap is not None:
        constraint_slack["turnover_cap"] = float(
            turnover_cap - np.sum(np.abs(weights.values - prev))
        )
    for sector, (lower, upper) in (sector_bounds or {}).items():
        sector_weight = float(sector_weights.get(sector, 0))
        constraint_slack[f"sector:{sector}:minimum"] = sector_weight - lower
        constraint_slack[f"sector:{sector}:maximum"] = upper - sector_weight
    if x is not None:
        portfolio_exposures = x.T @ weights
        for factor, (lower, upper) in (factor_bounds or {}).items():
            if factor in portfolio_exposures:
                value = float(portfolio_exposures[factor])
                constraint_slack[f"factor:{factor}:minimum"] = value - lower
                constraint_slack[f"factor:{factor}:maximum"] = upper - value
        benchmark = (
            benchmark_factor_exposures
            if benchmark_factor_exposures is not None
            else x.T
            @ (
                benchmark_weights.reindex(names).fillna(0)
                if benchmark_weights is not None
                else pd.Series(prev, index=names)
            )
        )
        for factor, (lower, upper) in (relative_factor_bounds or {}).items():
            if factor in portfolio_exposures:
                relative = float(portfolio_exposures[factor] - benchmark.get(factor, 0))
                constraint_slack[f"relative_factor:{factor}:minimum"] = relative - lower
                constraint_slack[f"relative_factor:{factor}:maximum"] = upper - relative
    diagnostics = {
        "condition_number": float(np.linalg.cond(cov)),
        "covariance_method": covariance_method,
        "weight_constraint_mode": weight_constraint_mode,
        "effective_weight_bounds": {
            name: [float(lower_bounds[position]), float(upper_bounds[position])]
            for position, name in enumerate(names)
        },
        "expected_return": float(weights @ pd.Series(mu, index=names)),
        "excess_return": float(weights @ pd.Series(mu, index=names)) - risk_free_rate,
        "risk_free_rate": float(risk_free_rate),
        "ex_ante_volatility": float(np.sqrt(max(variance, 0))),
        "turnover": float(np.sum(np.abs(weights.values - prev))),
        "factor_exposures": exposures.to_dict(),
        "sector_weights": sector_weights.to_dict(),
        "risk_contributions": pd.Series(
            _risk_contributions(weights.values, cov), index=names
        ).to_dict(),
        "constraint_slack": constraint_slack,
        "violations": violations,
        "feasible": bool(result.success and not violations),
    }
    return OptimizationResult(
        weights,
        bool(result.success and not violations),
        str(result.message),
        diagnostics,
    )
