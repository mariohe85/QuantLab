from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from django.db import models, transaction
from django.db.models import Prefetch

from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorModelCatalog,
    StockExposureSnapshot,
    StockModelFit,
)
from market_data.models import Security

from .engine import composite_scores
from .models import ScreenDefinition, ScreenResult, ScreenRun
from .services import ensure_price_securities

# FactorsToday universe floor: one year of trading. The 756-day window is the
# maximum lookback for a more stable beta, not a required age.
DEFAULT_MIN_TRADING_DAYS = 252
LEGACY_COVERAGE_DEFAULT = 0.8


def resolve_selection_filters(
    minimum_trading_days=None,
    minimum_coverage=None,
) -> tuple[int, float | None]:
    """Return (minimum trading days, optional coverage ratio).

    A stored 80% coverage filter was the old default and meant ~605 days of
    the 756-day window. When the days floor is left unset, that 80% is dropped
    so names with a year of history remain eligible.
    """
    days_was_set = minimum_trading_days not in (None, "")
    days = int(minimum_trading_days) if days_was_set else DEFAULT_MIN_TRADING_DAYS
    coverage = None if minimum_coverage in (None, "") else float(minimum_coverage)
    if (
        not days_was_set
        and coverage is not None
        and abs(coverage - LEGACY_COVERAGE_DEFAULT) < 1e-9
    ):
        coverage = None
    return days, coverage


def parse_ticker_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        values = value.replace("\n", ",").split(",")
    else:
        values = value
    result = []
    for item in values:
        ticker = str(item).strip().upper().replace(".", "-")
        if ticker and ticker not in result:
            result.append(ticker)
    return result


def compatible_factors(build: FactorBuild, model_level: str) -> list[FactorDefinition]:
    catalog = (
        FactorModelCatalog.objects.filter(
            model_version=build.definition_version,
            slug=model_level,
        )
        .prefetch_related("memberships__factor")
        .first()
    )
    if not catalog:
        return []
    return [
        membership.factor
        for membership in catalog.memberships.all()
        if membership.factor.active
    ]


def latest_selection_fits(build: FactorBuild, model_level: str):
    latest_as_of = StockModelFit.objects.filter(
        build=build, model_level=model_level
    ).aggregate(latest=models.Max("as_of"))["latest"]
    if not latest_as_of:
        return latest_as_of, StockModelFit.objects.none()
    return latest_as_of, StockModelFit.objects.filter(
        build=build,
        model_level=model_level,
        as_of=latest_as_of,
        security__asset_type="stock",
    )


def build_selection_preview(
    *,
    build: FactorBuild,
    model_level: str,
    factor_weights: dict[str, float],
    directions: dict[str, int] | None = None,
    top_n: int = 20,
    sectors: Iterable[str] | None = None,
    minimum_coverage: float | None = None,
    minimum_adjusted_r2: float | None = None,
    excluded_tickers: Iterable[str] = (),
    manual_tickers: Iterable[str] = (),
    minimum_trading_days: int | None = None,
) -> dict:
    factors = compatible_factors(build, model_level)
    by_name = {factor.name: factor for factor in factors}
    numeric_weights = {name: float(weight) for name, weight in factor_weights.items()}
    if any(weight < 0 for weight in numeric_weights.values()):
        raise ValueError(
            "Factor weights must be non-negative; use direction to invert a factor"
        )
    weights = {
        name: weight
        for name, weight in numeric_weights.items()
        if name in by_name and weight != 0
    }
    unknown = sorted(set(factor_weights) - set(by_name))
    if unknown:
        raise ValueError(f"Factors unavailable in {model_level}: {', '.join(unknown)}")
    if not weights:
        raise ValueError("Select at least one factor with a non-zero weight")
    if not 1 <= int(top_n) <= 100:
        raise ValueError("Stock limit must be between 1 and 100")

    direction_map = {
        name: -1 if int((directions or {}).get(name, 1)) < 0 else 1 for name in weights
    }
    excluded = set(parse_ticker_list(excluded_tickers))
    manual = set(parse_ticker_list(manual_tickers))
    overlap = excluded & manual
    if overlap:
        raise ValueError(
            f"Tickers cannot be both manual and excluded: {', '.join(sorted(overlap))}"
        )
    manual_securities = ensure_price_securities(sorted(manual)) if manual else {}

    latest_as_of, fit_query = latest_selection_fits(build, model_level)
    exposure_query = StockExposureSnapshot.objects.filter(
        factor__name__in=weights,
        factor__model_version=build.definition_version,
    ).select_related("factor")
    fits = list(
        fit_query.select_related("security")
        .prefetch_related(
            Prefetch(
                "exposures",
                queryset=exposure_query,
                to_attr="selection_exposures",
            )
        )
        .order_by("security__ticker")
    )
    if not fits:
        raise ValueError(
            "No latest stock models are available for this build and model"
        )

    records = {}
    for fit in fits:
        exposures = {item.factor.name: item for item in fit.selection_exposures}
        records[fit.security.ticker] = {
            "fit": fit,
            "security": fit.security,
            "exposures": exposures,
            "betas": {
                name: float(exposures[name].beta) if name in exposures else 0.0
                for name in weights
            },
        }
    features = pd.DataFrame.from_dict(
        {ticker: row["betas"] for ticker, row in records.items()},
        orient="index",
        dtype=float,
    )
    scored = composite_scores(features, weights, direction_map)
    composite = scored["composite"]
    percentiles = composite.rank(pct=True, method="average") * 100

    allowed_sectors = {str(item) for item in sectors or [] if str(item)}
    eligible = pd.Series(True, index=features.index)
    trading_days, coverage_floor = resolve_selection_filters(
        minimum_trading_days=minimum_trading_days,
        minimum_coverage=minimum_coverage,
    )
    if allowed_sectors:
        eligible &= pd.Series(
            {
                ticker: row["security"].sector in allowed_sectors
                for ticker, row in records.items()
            }
        )
    if trading_days:
        eligible &= pd.Series(
            {
                ticker: row["fit"].observation_count >= trading_days
                for ticker, row in records.items()
            }
        )
    if coverage_floor is not None:
        eligible &= pd.Series(
            {
                ticker: row["fit"].coverage >= coverage_floor
                for ticker, row in records.items()
            }
        )
    if minimum_adjusted_r2 is not None:
        eligible &= pd.Series(
            {
                ticker: row["fit"].adjusted_r2 >= float(minimum_adjusted_r2)
                for ticker, row in records.items()
            }
        )
    eligible.loc[list(excluded & set(eligible.index))] = False

    model_selected = set(
        composite[eligible].nlargest(min(int(top_n), int(eligible.sum()))).index
    )
    final_tickers = [
        ticker
        for ticker in composite.sort_values(ascending=False).index
        if ticker in model_selected
    ]
    final_tickers.extend(sorted(manual - set(final_tickers)))

    rows = []
    for ticker in composite.sort_values(ascending=False).index:
        record = records[ticker]
        is_manual = ticker in manual
        included = ticker in model_selected or is_manual
        selected_exposures = sum(name in record["exposures"] for name in weights)
        rows.append(
            {
                **record,
                "ticker": ticker,
                "score": float(composite[ticker]),
                "percentile": float(percentiles[ticker]),
                "components": {
                    name: float(scored.at[ticker, name]) for name in weights
                },
                "component_rows": [
                    {"name": name, "value": float(scored.at[ticker, name])}
                    for name in weights
                ],
                "confidence": selected_exposures / len(weights),
                "eligible": bool(eligible[ticker]),
                "included": included,
                "selection_source": (
                    "manual"
                    if is_manual
                    else ("model" if ticker in model_selected else "")
                ),
            }
        )
    for ticker in sorted(manual - set(records)):
        rows.append(
            {
                "fit": None,
                "security": manual_securities[ticker],
                "exposures": {},
                "betas": {},
                "ticker": ticker,
                "score": None,
                "percentile": None,
                "components": {},
                "component_rows": [],
                "confidence": 0.0,
                "eligible": False,
                "included": True,
                "selection_source": "manual",
                "rank": None,
            }
        )
    included_rows = sorted(
        (row for row in rows if row["included"]),
        key=lambda row: (
            row["selection_source"] == "manual",
            -(row["score"] if row["score"] is not None else float("-inf")),
            row["ticker"],
        ),
    )
    for rank, row in enumerate(
        (row for row in rows if row["fit"] is not None), start=1
    ):
        row["rank"] = rank
    equal_weight = 1 / len(included_rows) if included_rows else 0
    for row in included_rows:
        row["portfolio_weight"] = equal_weight
    return {
        "as_of": latest_as_of,
        "rows": rows,
        "chart_rows": rows[:5] + list(reversed(rows[-5:])),
        "included_rows": included_rows,
        "final_tickers": final_tickers,
        "model_selected_count": len(model_selected),
        "manual_count": len(manual - model_selected),
        "final_count": len(included_rows),
        "factor_weights": weights,
        "directions": direction_map,
        "sectors": sorted({fit.security.sector for fit in fits if fit.security.sector}),
        "universe_count": len(fits),
    }


@transaction.atomic
def persist_selection_run(definition: ScreenDefinition, preview: dict) -> ScreenRun:
    run = ScreenRun.objects.create(
        definition=definition,
        as_of=preview["as_of"],
        status="succeeded",
        diagnostics={
            "selected": preview["final_tickers"],
            "top_n": definition.top_n,
            "manual_tickers": definition.filters.get("manual_tickers", []),
            "excluded_tickers": definition.filters.get("excluded_tickers", []),
            "source": "stock_selection",
        },
    )
    securities = {
        item.ticker: item
        for item in Security.objects.filter(
            ticker__in=[row["ticker"] for row in preview["rows"]]
        )
    }
    ScreenResult.objects.bulk_create(
        [
            ScreenResult(
                run=run,
                security=securities[row["ticker"]],
                model_fit=row.get("fit"),
                passed=row["included"],
                rank=(
                    preview["final_tickers"].index(row["ticker"]) + 1
                    if row["included"]
                    else None
                ),
                percentile=(
                    row["percentile"] / 100
                    if row.get("percentile") is not None
                    else None
                ),
                score=row["score"],
                components=row["components"],
                rule_results={
                    "eligible": row["eligible"],
                    "selection_source": row["selection_source"],
                    "confidence": row["confidence"],
                },
            )
            for row in preview["rows"]
        ]
    )
    return run
