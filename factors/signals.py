from __future__ import annotations

from collections import defaultdict

import pandas as pd
from django.db import models
from django.db.models import Prefetch

from selection.engine import cross_sectional_zscore

from .models import (
    FactorBuild,
    FactorDefinition,
    FactorModelCatalog,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)


def signal_build_options(model_level: str) -> list[dict]:
    """Describe canonical model datasets and honest per-stock history depth."""
    options = []
    builds = FactorBuild.objects.filter(status="succeeded", is_canonical=True).order_by(
        "-as_of", "-id"
    )
    for build in builds:
        fits = StockModelFit.objects.filter(
            build=build,
            model_level=model_level,
            security__asset_type="stock",
        )
        latest = fits.aggregate(
            latest_as_of=models.Max("as_of"),
        )
        latest_as_of = latest["latest_as_of"]
        stock_count = (
            fits.filter(as_of=latest_as_of, security__asset_type="stock")
            .values("security_id")
            .distinct()
            .count()
            if latest_as_of
            else 0
        )
        months_by_stock = list(
            fits.values("security_id")
            .annotate(months=models.Count("period", distinct=True))
            .values_list("months", flat=True)
        )
        months_by_stock.sort()
        options.append(
            {
                "build": build,
                "as_of": latest_as_of,
                "stock_count": stock_count,
                "median_months": (
                    months_by_stock[len(months_by_stock) // 2] if months_by_stock else 0
                ),
                "maximum_months": max(months_by_stock, default=0),
                "history_stock_count": sum(months > 1 for months in months_by_stock),
            }
        )
    return options


def select_signal_build(options: list[dict], requested_id: str = ""):
    if requested_id.isdigit():
        selected = next(
            (option for option in options if option["build"].pk == int(requested_id)),
            None,
        )
        if selected:
            return selected
    return max(
        options,
        key=lambda option: (
            option["stock_count"],
            option["as_of"] or option["build"].as_of,
            option["build"].pk,
        ),
        default=None,
    )


def catalog_design_factors(
    build: FactorBuild | None, model_level: str
) -> list[FactorDefinition]:
    """Every active design factor of one model, in catalog position order."""
    if not build:
        return []
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


def catalog_factor_groups(build: FactorBuild | None, model_level: str) -> list[dict]:
    grouped = defaultdict(list)
    for factor in catalog_design_factors(build, model_level):
        grouped[factor.family].append(factor)
    order = (
        "market",
        "macro",
        "style",
        "sector",
        "industry",
        "country",
        "thematic",
    )
    return [
        {"family": family, "factors": grouped[family]}
        for family in order
        if grouped[family]
    ]


def flatten_factor_groups(groups: list[dict]) -> list[FactorDefinition]:
    return [factor for group in groups for factor in group["factors"]]


def _factor_return(build: FactorBuild, factor: FactorDefinition, as_of):
    return (
        FactorObservation.objects.filter(
            build=build,
            factor=factor,
            date__lte=as_of,
        )
        .order_by("-date")
        .first()
    )


def rank_factor_exposures(
    build: FactorBuild,
    model_level: str,
    factor: FactorDefinition,
    direction: int = 1,
    sector: str = "",
) -> dict:
    latest_as_of = StockModelFit.objects.filter(
        build=build,
        model_level=model_level,
    ).aggregate(as_of=models.Max("as_of"))["as_of"]
    if not latest_as_of:
        return {
            "as_of": None,
            "rows": [],
            "chart_rows": [],
            "factor_observation": None,
            "stock_count": 0,
            "selected_count": 0,
            "sectors": [],
        }

    exposure_query = StockExposureSnapshot.objects.filter(factor=factor)
    fits = (
        StockModelFit.objects.filter(
            build=build,
            model_level=model_level,
            as_of=latest_as_of,
            security__asset_type="stock",
        )
        .select_related("security")
        .prefetch_related(
            Prefetch("exposures", queryset=exposure_query, to_attr="signal_exposures")
        )
        .order_by("security__ticker")
    )
    all_sectors = sorted({fit.security.sector for fit in fits if fit.security.sector})
    if sector:
        fits = [fit for fit in fits if fit.security.sector == sector]
    else:
        fits = list(fits)

    records = []
    for fit in fits:
        exposure = fit.signal_exposures[0] if fit.signal_exposures else None
        records.append(
            {
                "fit": fit,
                "security": fit.security,
                "beta": float(exposure.beta) if exposure else 0.0,
                "selected": exposure is not None,
                "t_stat": exposure.t_stat if exposure else None,
                "p_value": exposure.p_value if exposure else None,
            }
        )
    betas = pd.Series(
        {record["security"].ticker: record["beta"] for record in records},
        dtype=float,
    )
    scores = cross_sectional_zscore(betas) * direction
    ranks = scores.rank(ascending=False, method="first")
    percentiles = scores.rank(pct=True, method="average") * 100
    factor_observation = _factor_return(build, factor, latest_as_of)
    latest_return = (
        factor_observation.scaled_return if factor_observation is not None else None
    )
    for record in records:
        ticker = record["security"].ticker
        record.update(
            {
                "rank": int(ranks[ticker]),
                "score": float(scores[ticker]),
                "percentile": float(percentiles[ticker]),
                "contribution": (
                    record["beta"] * latest_return
                    if latest_return is not None
                    else None
                ),
            }
        )
    records.sort(key=lambda record: (record["rank"], record["security"].ticker))
    chart_records = records[:5] + list(reversed(records[-5:]))
    return {
        "as_of": latest_as_of,
        "rows": records,
        "chart_rows": chart_records,
        "factor_observation": factor_observation,
        "stock_count": len(records),
        "selected_count": sum(record["selected"] for record in records),
        "sectors": all_sectors,
    }


def _universe_percentiles(
    build: FactorBuild,
    model_level: str,
    factor: FactorDefinition,
    as_of_dates: list,
    direction: int = 1,
) -> dict:
    """Cross-sectional percentile of every stock's beta at each stored month-end.

    Mirrors rank_factor_exposures: the whole fitted universe participates, and
    factors ElasticNet dropped stay in at beta 0.
    """
    if not as_of_dates:
        return {}
    universe = StockModelFit.objects.filter(
        build=build,
        model_level=model_level,
        as_of__in=as_of_dates,
        security__asset_type="stock",
    ).values_list("as_of", "security__ticker", "id")
    betas_by_date = defaultdict(dict)
    fit_dates = {}
    for as_of, stock, fit_id in universe:
        betas_by_date[as_of][stock] = 0.0
        fit_dates[fit_id] = (as_of, stock)
    exposures = StockExposureSnapshot.objects.filter(
        factor=factor, model_fit_id__in=fit_dates
    ).values_list("model_fit_id", "beta")
    for fit_id, beta in exposures:
        as_of, stock = fit_dates[fit_id]
        betas_by_date[as_of][stock] = float(beta)

    percentiles = {}
    for as_of, betas in betas_by_date.items():
        series = pd.Series(betas, dtype=float)
        scores = cross_sectional_zscore(series) * direction
        percentiles[as_of] = scores.rank(pct=True, method="average") * 100
    return percentiles


def company_factor_history(
    build: FactorBuild,
    model_level: str,
    factor: FactorDefinition,
    ticker: str,
    direction: int = 1,
) -> list[dict]:
    exposure_query = StockExposureSnapshot.objects.filter(factor=factor)
    fits = (
        StockModelFit.objects.filter(
            build=build,
            model_level=model_level,
            security__ticker=ticker,
        )
        .select_related("security")
        .prefetch_related(
            Prefetch("exposures", queryset=exposure_query, to_attr="signal_exposures")
        )
        .order_by("as_of")
    )
    fits = list(fits)
    percentiles = _universe_percentiles(
        build,
        model_level,
        factor,
        [fit.as_of for fit in fits],
        direction=direction,
    )
    rows = []
    for fit in fits:
        exposure = fit.signal_exposures[0] if fit.signal_exposures else None
        ranked = percentiles.get(fit.as_of)
        rows.append(
            {
                "as_of": fit.as_of,
                "beta": float(exposure.beta) if exposure else 0.0,
                "selected": exposure is not None,
                "percentile": (
                    float(ranked[ticker])
                    if ranked is not None and ticker in ranked
                    else None
                ),
                "t_stat": exposure.t_stat if exposure else None,
                "p_value": exposure.p_value if exposure else None,
                "adjusted_r2": fit.adjusted_r2,
                "residual_volatility": fit.residual_volatility,
                "coverage": fit.coverage,
                "observation_count": fit.observation_count,
            }
        )
    return rows
