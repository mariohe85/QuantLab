from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if std > 0 else pd.Series(0.0, index=values.index)


# ElasticNet leaves a spike of exact-zero betas in the cross-section, which shrinks
# the standard deviation and pushes genuinely exposed names past |z| = 6. Left
# unclipped, one such loading dominates a weighted composite.
COMPOSITE_ZSCORE_CAP = 3.0


def scoring_zscore(values: pd.Series, direction: int = 1) -> pd.Series:
    return (cross_sectional_zscore(values) * direction).clip(
        -COMPOSITE_ZSCORE_CAP, COMPOSITE_ZSCORE_CAP
    )


def composite_scores(
    features: pd.DataFrame,
    weights: dict[str, float],
    directions: dict[str, int] | None = None,
) -> pd.DataFrame:
    directions = directions or {}
    missing = [name for name, weight in weights.items() if weight and name not in features]
    if missing:
        raise ValueError(f"Configured factors are unavailable: {', '.join(sorted(missing))}")
    used = [name for name in weights if name in features]
    if not used:
        raise ValueError("No configured features exist in the supplied frame")
    normalized = pd.DataFrame({
        name: scoring_zscore(features[name], directions.get(name, 1)) for name in used
    })
    denominator = sum(abs(weights[name]) for name in used)
    if denominator == 0:
        raise ValueError("Composite weights cannot all be zero")
    score = sum(normalized[name] * weights[name] for name in used) / denominator
    return normalized.assign(composite=score).sort_values("composite", ascending=False)


def select_top_n(
    features: pd.DataFrame,
    weights: dict[str, float],
    top_n: int,
    directions: dict[str, int] | None = None,
    sectors: pd.Series | None = None,
    allowed_sectors: list[str] | None = None,
    minimums: dict[str, float] | None = None,
    maximums: dict[str, float] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=features.index)
    for name, value in (minimums or {}).items():
        if name in features:
            mask &= features[name] >= value
    for name, value in (maximums or {}).items():
        if name in features:
            mask &= features[name] <= value
    if sectors is not None and allowed_sectors:
        mask &= sectors.reindex(features.index).isin(allowed_sectors)
    scores = composite_scores(features.loc[mask], weights, directions)
    return scores.iloc[: min(top_n, len(scores))]


def rank_ic(scores: pd.Series, forward_returns: pd.Series) -> float:
    joined = pd.concat([scores, forward_returns], axis=1).dropna()
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")) if len(joined) > 2 else np.nan


def preview_screen(
    features: pd.DataFrame,
    weights: dict[str, float],
    filters: dict,
    sectors: pd.Series,
    regime_interactions: dict[str, float] | None = None,
    directions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return every eligible name with transparent pass/fail and score components."""
    working = features.copy()
    regime_interactions = regime_interactions or {}
    directions = directions or {}
    effective_weights = dict(weights)
    for factor, regime in regime_interactions.items():
        if factor in working:
            column = f"{factor}_regime"
            working[column] = working[factor] * float(regime)
            effective_weights[column] = effective_weights.get(factor, 1.0)
    missing = [
        name for name, weight in effective_weights.items()
        if weight and name not in working
    ]
    if missing:
        raise ValueError(f"Configured factors are unavailable: {', '.join(sorted(missing))}")
    used = [name for name in effective_weights if name in working]
    normalized = pd.DataFrame(
        {
            name: scoring_zscore(working[name], int(directions.get(name, 1)))
            for name in used
        },
        index=working.index,
    )
    denominator = sum(abs(effective_weights[name]) for name in used)
    if not used or denominator == 0:
        raise ValueError("Screen requires at least one available non-zero factor weight")
    composite = sum(
        normalized[name] * effective_weights[name] for name in used
    ) / denominator
    passed = pd.Series(True, index=working.index)
    rules = {ticker: {} for ticker in working.index}
    allowed_sectors = filters.get("sectors")
    if allowed_sectors:
        result = sectors.reindex(working.index).isin(allowed_sectors)
        passed &= result
        for ticker in working.index:
            rules[ticker]["sector"] = bool(result[ticker])
    aliases = {
        "market_cap": "market_cap",
        "beta": "Market",
        "significance": "beta_significance",
        "adjusted_r2": "adjusted_r2",
        "residual_volatility": "residual_volatility",
        "coverage": "coverage",
    }
    for rule, column in aliases.items():
        if rule not in filters:
            continue
        if column not in working:
            raise ValueError(f"Configured filter '{rule}' requires unavailable field '{column}'")
        bounds = filters.get(rule, {})
        result = pd.Series(True, index=working.index)
        values = pd.to_numeric(working[column], errors="coerce")
        if bounds.get("min") is not None:
            result &= values >= float(bounds["min"])
        if bounds.get("max") is not None:
            result &= values <= float(bounds["max"])
        passed &= result
        for ticker in working.index:
            rules[ticker][rule] = bool(result[ticker])
    rank = composite.where(passed).rank(ascending=False, method="first")
    percentile = composite.rank(pct=True)
    return pd.DataFrame(
        {
            "passed": passed,
            "composite": composite,
            "rank": rank,
            "percentile": percentile,
            "components": [
                normalized.loc[ticker].to_dict() for ticker in working.index
            ],
            "rules": [rules[ticker] for ticker in working.index],
        },
        index=working.index,
    ).sort_values(["passed", "composite"], ascending=[False, False])
