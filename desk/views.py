from __future__ import annotations

import csv
import json
import logging
import math

import pandas as pd
from django.db import models
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import urlencode
from django.views.decorators.http import require_POST

from backtests.models import BacktestRun
from factors.catalog import factor_construction_details
from factors.engine import factor_covariance, portfolio_decomposition
from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorObservation,
    StockModelFit,
)
from factors.proxyfactorlib import PROXY_MODEL_VERSION
from factors.signals import (
    catalog_design_factors,
    catalog_factor_groups,
    company_factor_history,
    flatten_factor_groups,
    rank_factor_exposures,
    select_signal_build,
    signal_build_options,
)
from jobs.models import Job
from jobs.runner import request_cancellation, submit_job
from market_data.models import PriceSnapshot, Security
from optimization.models import OptimizationScenario, OptimizationStudy
from selection.history import replay_portfolio_history, replay_screen_history
from selection.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioRiskSnapshot,
    ScreenDefinition,
    ScreenRun,
)
from selection.services import (
    clone_portfolio,
    parse_holdings_csv,
    portfolio_from_screen,
    save_portfolio,
    save_selection_portfolio,
    validate_holdings,
)
from selection.stock_selection import (
    build_selection_preview,
    compatible_factors,
    parse_ticker_list,
    persist_selection_run,
    resolve_selection_filters,
)

logger = logging.getLogger("quantlab.desk")

MODEL_LEVELS = (
    ("base", "Base"),
    ("base_sector", "Base + Sector"),
    ("base_sector_industry", "Base + Sector + Industry"),
    ("all_factors", "All Factors"),
)


def _selected_factor_build(request):
    all_builds = FactorBuild.objects.filter(
        is_canonical=True, definition_version=PROXY_MODEL_VERSION
    ).order_by("-as_of", "-id")
    builds = all_builds[:20]
    selected_id = request.GET.get("build_id") or request.POST.get("build_id")
    selected = (
        all_builds.filter(pk=selected_id).first()
        if selected_id and selected_id.isdigit()
        else None
    )
    return builds, selected or all_builds.first()


def _model_level_options(build, ticker=""):
    fits = (
        StockModelFit.objects.filter(build=build)
        if build
        else StockModelFit.objects.none()
    )
    if ticker:
        fits = fits.filter(security__ticker=ticker)
    available = {
        row["model_level"]: row
        for row in fits.values("model_level")
        .order_by()
        .annotate(
            fit_count=models.Count("id"),
            factor_count=models.Max("design_factor_count"),
        )
    }
    return [
        {
            "key": key,
            "label": label,
            "available": key in available,
            "fit_count": available.get(key, {}).get("fit_count", 0),
            "factor_count": available.get(key, {}).get("factor_count", 0),
        }
        for key, label in MODEL_LEVELS
    ]


FACTOR_PANEL_WINDOWS = (1, 5, 21, 63, 252)


def _factor_panel_cells(observation):
    cells = []
    for window in FACTOR_PANEL_WINDOWS:
        key = str(window)
        # The one-day horizon is the stored daily return, so it needs no lookup.
        horizon_return = (
            observation.scaled_return if window == 1 else observation.horizons.get(key)
        )
        cells.append(
            {
                "window": window,
                "return_percent": (
                    None if horizon_return is None else float(horizon_return) * 100
                ),
                "zscore": observation.zscores.get(key),
            }
        )
    return cells


def _factor_page_state(request):
    builds, selected_build = _selected_factor_build(request)
    observations = FactorObservation.objects.none()
    selected_families = [item for item in request.GET.getlist("family") if item]
    if selected_build:
        latest_dates = (
            FactorObservation.objects.filter(build=selected_build)
            .values("factor_id")
            .annotate(last=models.Max("date"))
        )
        newest = models.Q()
        for row in latest_dates:
            newest |= models.Q(factor_id=row["factor_id"], date=row["last"])
        observations = (
            FactorObservation.objects.filter(build=selected_build)
            .filter(newest)
            .select_related("factor")
            .order_by("factor__sort_order", "factor__name")
            if latest_dates
            else FactorObservation.objects.none()
        )
        if selected_families:
            observations = observations.filter(factor__family__in=selected_families)
    panel_rows = []
    panel_dates = set()
    for item in observations:
        panel_dates.add(item.date)
        panel_rows.append({"observation": item, "cells": _factor_panel_cells(item)})
    return {
        "selected_factor_build": selected_build,
        "factor_builds": builds,
        "selected_families": selected_families,
        "factor_families": list(
            FactorDefinition.objects.filter(
                active=True, model_version=PROXY_MODEL_VERSION
            )
            .order_by("family")
            .values_list("family", flat=True)
            .distinct()
        ),
        "factor_observations": observations,
        "factor_panel_rows": panel_rows,
        "factor_panel_dates": sorted(panel_dates),
        "factor_panel_windows": FACTOR_PANEL_WINDOWS,
        "factor_panel_column_count": 3 + 2 * len(FACTOR_PANEL_WINDOWS),
    }


def _stock_page_state(
    ticker: str, build=None, model_level="base_sector", beta_factor=""
):
    fits = (
        StockModelFit.objects.filter(security__ticker=ticker, build=build)
        .select_related("security", "build")
        .prefetch_related("exposures__factor")
        .order_by("-as_of", "model_level")
        if ticker and build
        else StockModelFit.objects.none()
    )
    level_fits = list(fits.filter(model_level=model_level))
    selected_fit = level_fits[0] if level_fits else None
    history_rows = [
        {
            "as_of": item.as_of,
            "alpha": item.alpha,
            "adjusted_r2": item.adjusted_r2,
            "residual_volatility": item.residual_volatility,
            "active_factor_count": item.active_factor_count,
            "design_factor_count": item.design_factor_count,
            "observation_count": item.observation_count,
            "coverage": item.coverage,
        }
        for item in reversed(level_fits)
    ]
    beta_series = {}
    for item in level_fits:
        for exposure in item.exposures.all():
            beta_series.setdefault(exposure.factor.name, {})[item.as_of] = exposure.beta
    beta_options = sorted(beta_series)
    selected_beta_factor = beta_factor if beta_factor in beta_series else ""
    if not selected_beta_factor and beta_options:
        if selected_fit:
            ranked = sorted(
                selected_fit.exposures.all(),
                key=lambda item: abs(item.beta),
                reverse=True,
            )
            selected_beta_factor = next(
                (
                    item.factor.name
                    for item in ranked
                    if item.factor.name in beta_series
                ),
                beta_options[0],
            )
        else:
            selected_beta_factor = beta_options[0]
    beta_history_rows = [
        {"as_of": stamp, "beta": value}
        for stamp, value in sorted(beta_series.get(selected_beta_factor, {}).items())
    ]
    factor_rows = []
    latest_returns = {}
    if build:
        latest_returns = {
            item.factor_id: item.scaled_return
            for item in FactorObservation.objects.filter(build=build, date=build.as_of)
        }
    if selected_fit:
        for exposure in selected_fit.exposures.all():
            factor_return = latest_returns.get(exposure.factor_id)
            factor_rows.append(
                {
                    "level": selected_fit.model_level,
                    "factor": exposure.factor.name,
                    "beta": exposure.beta,
                    "t_stat": exposure.t_stat,
                    "p_value": exposure.p_value,
                    "confidence_low": exposure.confidence_low,
                    "confidence_high": exposure.confidence_high,
                    "as_of": selected_fit.as_of,
                    "factor_return": factor_return,
                    "contribution": (
                        exposure.beta * factor_return
                        if factor_return is not None
                        else None
                    ),
                }
            )
        factor_rows.sort(key=lambda row: abs(row["beta"]), reverse=True)
    securities = (
        Security.objects.filter(model_fits__build=build).distinct().order_by("ticker")
        if build
        else Security.objects.none()
    )
    return {
        "stock_fits": level_fits[:36],
        "stock_history_rows": history_rows,
        "stock_beta_history_rows": beta_history_rows,
        "stock_beta_factor_options": beta_options,
        "selected_stock_beta_factor": selected_beta_factor,
        "stock_exposure_rows": factor_rows,
        "selected_stock_fit": selected_fit,
        "stock_options": securities,
        "model_levels": _model_level_options(build, ticker),
        "selected_model_level": model_level,
    }


def _holdings_build_coverage(security_ids, model_level):
    """Count how many of these securities each build models at one level."""
    if not security_ids:
        return {}
    return {
        row["build_id"]: row["modeled"]
        for row in StockModelFit.objects.filter(
            security_id__in=security_ids,
            model_level=model_level,
        )
        .values("build_id")
        .annotate(modeled=models.Count("security_id", distinct=True))
    }


def _portfolio_build(request, model_level, default_build):
    """Prefer the build that models the most holdings unless one was requested."""
    if request.GET.get("build_id") or request.POST.get("build_id"):
        return default_build
    portfolio_id = request.GET.get("portfolio_id") or request.POST.get("portfolio_id")
    if not (portfolio_id and portfolio_id.isdigit()):
        return default_build
    security_ids = list(
        PortfolioHolding.objects.filter(portfolio_id=portfolio_id).values_list(
            "security_id", flat=True
        )
    )
    coverage = _holdings_build_coverage(security_ids, model_level)
    if not coverage:
        return default_build
    best_id = max(coverage.items(), key=lambda item: (item[1], item[0]))[0]
    if default_build and coverage.get(default_build.pk, 0) >= coverage[best_id]:
        return default_build
    return (
        FactorBuild.objects.filter(
            pk=best_id, definition_version=PROXY_MODEL_VERSION
        ).first()
        or default_build
    )


def _risk_contribution_rows(decomposition):
    """Order the Euler risk split largest first and close it with specific risk.

    Every entry is a contribution to volatility, so the column adds up to the
    portfolio's predicted volatility and the shares add up to one.
    """
    if not decomposition:
        return []
    volatility = decomposition["predicted_volatility"]
    if not volatility:
        return []
    rows = [
        {
            "name": factor,
            "contribution": value,
            "percent": 100 * value / volatility,
            "specific": False,
        }
        for factor, value in decomposition["factor_contribution"].items()
    ]
    rows.sort(key=lambda row: row["contribution"], reverse=True)
    specific = decomposition["specific_contribution"]
    rows.append(
        {
            "name": "Specific",
            "contribution": specific,
            "percent": 100 * specific / volatility,
            "specific": True,
        }
    )
    return rows


def _portfolio_factor_state(request, build, model_level):
    portfolios = Portfolio.objects.filter(archived=False).order_by("name")
    selected_id = request.GET.get("portfolio_id") or request.POST.get("portfolio_id")
    selected = (
        portfolios.filter(pk=selected_id).first()
        if selected_id and selected_id.isdigit()
        else None
    )
    manual_text = request.POST.get("holdings", "")
    error = ""
    holdings = []
    if manual_text:
        try:
            parsed = parse_holdings_csv(manual_text)
            if not parsed:
                raise ValueError("Enter at least one ticker and weight.")
            if any(row["weight"] is None for row in parsed):
                raise ValueError(
                    "Manual analysis requires weights; share pricing is unavailable here."
                )
            parsed = validate_holdings(parsed)
            known = {
                item.ticker: item
                for item in Security.objects.filter(
                    ticker__in=[row["ticker"] for row in parsed]
                )
            }
            holdings = [
                {"security": known[row["ticker"]], "weight": row["weight"]}
                for row in parsed
            ]
        except ValueError as exc:
            error = str(exc)
    elif selected:
        saved = list(selected.holdings.select_related("security"))
        weighted = [item for item in saved if item.weight is not None]
        total = sum(item.weight for item in weighted)
        if total > 0:
            holdings = [
                {"security": item.security, "weight": item.weight / total}
                for item in weighted
            ]
        elif saved:
            error = "This portfolio has shares but no weights; refresh pricing before factor analysis."

    portfolio_tab = request.GET.get("portfolio_tab", "holdings")
    if manual_text:
        portfolio_tab = "exposure"
    if portfolio_tab not in {"holdings", "exposure", "performance"}:
        portfolio_tab = "holdings"
    fit_map = {}
    if build and holdings:
        fit_ids = {}
        fit_candidates = (
            StockModelFit.objects.filter(
                build=build,
                as_of__lte=build.as_of,
                security_id__in=[holding["security"].pk for holding in holdings],
                model_level=model_level,
            )
            .values_list("id", "security_id")
            .order_by("security_id", "-as_of")
        )
        for fit_id, security_id in fit_candidates:
            fit_ids.setdefault(security_id, fit_id)
        for fit in StockModelFit.objects.filter(
            pk__in=fit_ids.values()
        ).prefetch_related("exposures__factor"):
            fit_map[fit.security_id] = fit
    exposure_totals = {}
    covered_weight = 0.0
    specific_variance = 0.0
    holding_rows = []
    matrix_scores = {}
    for holding in holdings:
        fit = fit_map.get(holding["security"].pk)
        exposure_map = (
            {exposure.factor.name: exposure.beta for exposure in fit.exposures.all()}
            if fit
            else {}
        )
        for factor, beta in exposure_map.items():
            matrix_scores[factor] = matrix_scores.get(factor, 0) + abs(
                holding["weight"] * beta
            )
        holding_rows.append(
            {
                **holding,
                "fit": fit,
                "exposure_map": exposure_map,
            }
        )
        if not fit:
            continue
        covered_weight += holding["weight"]
        specific_variance += (holding["weight"] * fit.residual_volatility) ** 2
        for exposure in fit.exposures.all():
            exposure_totals[exposure.factor.name] = (
                exposure_totals.get(exposure.factor.name, 0.0)
                + holding["weight"] * exposure.beta
            )
    catalog_names = [
        factor.name for factor in catalog_design_factors(build, model_level)
    ]
    matrix_factors = catalog_names or sorted(
        matrix_scores, key=lambda name: matrix_scores[name], reverse=True
    )
    for row in holding_rows:
        row["matrix_values"] = [
            {
                "factor": factor,
                "beta": row["exposure_map"].get(factor, 0) if row["fit"] else None,
            }
            for factor in matrix_factors
        ]
    exposure_rows = [
        {"factor": factor, "exposure": value}
        for factor, value in sorted(
            exposure_totals.items(), key=lambda item: abs(item[1]), reverse=True
        )
    ]
    live_risk = None
    if build and covered_weight > 0:
        modeled = [row for row in holding_rows if row["fit"]]
        modeled_weights = pd.Series(
            {
                row["security"].ticker: row["weight"] / covered_weight
                for row in modeled
            },
            dtype=float,
        )
        exposures = pd.DataFrame.from_dict(
            {
                row["security"].ticker: row["exposure_map"]
                for row in modeled
            },
            orient="index",
        ).fillna(0.0)
        factor_returns = pd.DataFrame.from_records(
            FactorObservation.objects.filter(
                build=build,
                date__lte=build.as_of,
                factor__name__in=exposures.columns,
            ).values("date", "factor__name", "scaled_return")
        )
        if not factor_returns.empty:
            factor_returns = factor_returns.pivot(
                index="date", columns="factor__name", values="scaled_return"
            )
        if len(factor_returns.index) >= 2:
            live_risk = portfolio_decomposition(
                modeled_weights,
                exposures,
                factor_covariance(factor_returns),
                pd.Series(
                    {
                        row["security"].ticker: row["fit"].residual_volatility
                        for row in modeled
                    },
                    dtype=float,
                ),
            )
            live_risk["component_risk"] = live_risk["factor_contribution"].to_dict()
    contribution_rows = _risk_contribution_rows(live_risk)
    if live_risk:
        marginal = live_risk["factor_marginal_risk"]
        contribution = live_risk["factor_contribution"]
        volatility = live_risk["predicted_volatility"]
        for row in exposure_rows:
            value = contribution.get(row["factor"])
            row["marginal_risk"] = marginal.get(row["factor"])
            row["contribution"] = value
            row["risk_percent"] = (
                100 * value / volatility if value is not None and volatility else None
            )
    modeled_count = sum(row["fit"] is not None for row in holding_rows)
    live_total_variance = live_risk["predicted_variance"] if live_risk else None
    live_specific_risk = (
        live_risk["specific_variance"] ** 0.5 if live_risk else None
    )
    live_factor_risk = (
        max(live_risk["factor_variance"], 0) ** 0.5 if live_risk else None
    )
    coverage_by_build = _holdings_build_coverage(
        [holding["security"].pk for holding in holdings], model_level
    )
    better_builds = [
        {"build": item, "modeled": coverage_by_build[item.pk]}
        for item in FactorBuild.objects.filter(
            is_canonical=True,
            definition_version=PROXY_MODEL_VERSION,
            pk__in=[
                key
                for key, value in coverage_by_build.items()
                if value > modeled_count and (not build or key != build.pk)
            ],
        ).order_by("-id")
    ]
    portfolio_history = None
    portfolio_history_error = ""
    if portfolio_tab == "performance" and selected:
        try:
            portfolio_history = _history_page_context(
                replay_portfolio_history(selected, build)
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            portfolio_history_error = str(exc)
    return {
        "factor_portfolios": portfolios,
        "selected_factor_portfolio": selected,
        "manual_holdings": manual_text,
        "portfolio_analysis_error": error,
        "factor_portfolio_holdings": holding_rows,
        "factor_portfolio_exposures": exposure_rows,
        "factor_portfolio_coverage": covered_weight if holdings else None,
        "factor_portfolio_specific_volatility": (
            live_specific_risk
            if live_specific_risk is not None
            else (specific_variance**0.5 if holdings else None)
        ),
        "factor_portfolio_factor_risk": live_factor_risk,
        "factor_portfolio_total_risk": (
            live_risk["predicted_volatility"] if live_risk else None
        ),
        "factor_portfolio_specific_risk_share": (
            100 * live_risk["specific_variance"] / live_total_variance
            if live_total_variance and live_risk
            else None
        ),
        "factor_portfolio_factor_risk_share": (
            100 * live_risk["factor_variance"] / live_total_variance
            if live_total_variance and live_risk
            else None
        ),
        "factor_portfolio_risk": live_risk,
        "factor_portfolio_contributions": contribution_rows,
        "portfolio_workspace_tab": portfolio_tab,
        "portfolio_matrix_factors": matrix_factors,
        "portfolio_modeled_holdings": modeled_count,
        "portfolio_better_builds": better_builds,
        "portfolio_history": portfolio_history,
        "portfolio_history_error": portfolio_history_error,
        "portfolio_weighted_adjusted_r2": (
            sum(
                row["weight"] * row["fit"].adjusted_r2
                for row in holding_rows
                if row["fit"]
            )
            / covered_weight
            if covered_weight
            else None
        ),
    }


OPTIMIZER_OBJECTIVE_LABELS = {
    "min_variance": "Minimum variance",
    "max_return": "Maximum return",
    "max_sharpe": "Maximum Sharpe ratio",
    "mean_variance": "Maximum return minus ½ risk",
}
OPTIMIZER_RETURN_MODEL_LABELS = {
    "": "Not used",
    "equal_sharpe": "Equal Sharpe (vol × k)",
    "factor_premium": "Factor implied (beta × premium)",
    "historical_shrinkage": "Historical shrinkage",
    "user_supplied": "User supplied",
}
def _percent_text(value, decimals=1):
    if value is None:
        return None
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return None


def _bound_text(bounds, scale=1, decimals=2, suffix=""):
    """Render a stored [min, max] pair, hiding the sentinel open ends."""
    try:
        lower, upper = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError):
        return "—"
    open_low = lower <= -1_000
    open_high = upper >= 1_000
    if open_low and open_high:
        return "unbounded"
    if open_low:
        return f"≤ {upper * scale:.{decimals}f}{suffix}"
    if open_high:
        return f"≥ {lower * scale:.{decimals}f}{suffix}"
    return (
        f"{lower * scale:.{decimals}f}{suffix} to {upper * scale:.{decimals}f}{suffix}"
    )


def _optimization_setting_rows(configuration, model_level=""):
    """Readable "what did this run ask for" rows built from stored parameters."""
    configuration = configuration or {}
    objective = configuration.get("objective", "min_variance")
    return_model = configuration.get("expected_return_model", "")
    rows = [
        {
            "label": "Objective",
            "value": OPTIMIZER_OBJECTIVE_LABELS.get(objective, objective or "—"),
        },
        {
            "label": "Expected returns",
            "value": OPTIMIZER_RETURN_MODEL_LABELS.get(
                return_model, return_model or "Not used"
            ),
        },
    ]
    rows.append({"label": "Risk model", "value": "Factor model"})
    rows.append(
        {
            "label": "Model level",
            "value": dict(MODEL_LEVELS).get(
                model_level or configuration.get("model_level", ""), "—"
            ),
        }
    )
    if objective == "mean_variance":
        rows.append(
            {
                "label": "Risk weight",
                "value": f"{float(configuration.get('risk_aversion', 1)):g}",
            }
        )
    if return_model == "equal_sharpe" and configuration.get("common_sharpe"):
        rows.append(
            {
                "label": "Common Sharpe (k)",
                "value": f"{float(configuration['common_sharpe']):.2f}",
            }
        )
    if configuration.get("lookback"):
        rows.append(
            {
                "label": "Lookback",
                "value": f"{int(configuration['lookback'])} observations",
            }
        )
    if configuration.get("risk_free_rate") is not None:
        rows.append(
            {
                "label": "Risk-free fallback",
                "value": _percent_text(configuration["risk_free_rate"], 2) or "—",
            }
        )
    if configuration.get("as_of"):
        rows.append({"label": "As of", "value": configuration["as_of"]})
    return rows


def _optimization_constraint_rows(configuration):
    """Every guardrail the solver was given, grouped for display."""
    configuration = configuration or {}
    rows = []
    relative = configuration.get("weight_constraint_mode") == "relative"
    minimum = _percent_text(configuration.get("min_weight"), 1)
    maximum = _percent_text(configuration.get("max_weight"), 1)
    if minimum or maximum:
        rows.append(
            {
                "group": "Stock weights",
                "label": (
                    "Relative to current weight" if relative else "Absolute per name"
                ),
                "value": f"{minimum or '0.0%'} to {maximum or 'unbounded'}",
            }
        )
    if configuration.get("max_weight_cap") is not None:
        rows.append(
            {
                "group": "Stock weights",
                "label": "Hard ceiling on any name",
                "value": _percent_text(configuration["max_weight_cap"], 1) or "—",
            }
        )
    for key, label in (
        ("turnover_cap", "Turnover cap"),
        ("tracking_error_limit", "Tracking error limit"),
        ("transaction_cost", "Transaction cost"),
        ("turnover_penalty", "Turnover penalty"),
    ):
        if configuration.get(key) is not None:
            rows.append(
                {
                    "group": "Portfolio",
                    "label": label,
                    "value": _percent_text(configuration[key], 2) or "—",
                }
            )
    if configuration.get("max_factor_variance") is not None:
        rows.append(
            {
                "group": "Portfolio",
                "label": "Maximum factor variance share",
                "value": f"{float(configuration['max_factor_variance']):.2f}",
            }
        )
    for name, bounds in sorted((configuration.get("factor_bounds") or {}).items()):
        rows.append(
            {
                "group": "Factor beta",
                "label": name,
                "value": _bound_text(bounds),
            }
        )
    for name, bounds in sorted(
        (configuration.get("relative_factor_bounds") or {}).items()
    ):
        rows.append(
            {
                "group": "Factor beta vs SPY",
                "label": name,
                "value": _bound_text(bounds),
            }
        )
    for name, cap in sorted((configuration.get("max_factor_components") or {}).items()):
        rows.append(
            {
                "group": "Factor beta",
                "label": f"{name} component cap",
                "value": f"{float(cap):.2f}",
            }
        )
    for name, budget in sorted((configuration.get("risk_budgets") or {}).items()):
        rows.append(
            {
                "group": "Risk budget",
                "label": name,
                "value": _percent_text(budget, 1) or "—",
            }
        )
    for name, bounds in sorted((configuration.get("sector_bounds") or {}).items()):
        rows.append(
            {
                "group": "Sector weight",
                "label": name,
                "value": _bound_text(bounds, scale=100, decimals=1, suffix="%"),
            }
        )
    return rows


def _optimizer_sector_rows(holdings, configuration):
    """Portfolio weight per sector next to the bound the solver was given.

    Sector bounds limit weight, not factor beta, so they cannot be read off the
    factor exposure chart.
    """
    bounds = (configuration or {}).get("sector_bounds") or {}
    totals = {}
    for item in holdings:
        sector = item.security.sector or "Unclassified"
        entry = totals.setdefault(sector, {"original": 0.0, "optimized": 0.0})
        entry["original"] += item.original_weight or 0.0
        entry["optimized"] += item.optimized_weight or 0.0
    rows = []
    for sector in sorted(totals, key=lambda name: -totals[name]["optimized"]):
        entry = totals[sector]
        bound = bounds.get(sector)
        within = None
        if bound:
            within = bool(
                float(bound[0]) - 1e-6 <= entry["optimized"] <= float(bound[1]) + 1e-6
            )
        rows.append(
            {
                "sector": sector,
                "original": entry["original"],
                "optimized": entry["optimized"],
                "change": entry["optimized"] - entry["original"],
                "bound": (
                    _bound_text(bound, scale=100, decimals=1, suffix="%")
                    if bound
                    else ""
                ),
                "within": within,
            }
        )
    return rows


def _optimization_study_rows(studies):
    """Study list entries carrying enough settings to read without clicking."""
    rows = []
    for study in studies:
        configuration = study.configuration or {}
        variants = list(study.variants.all())
        succeeded = [item for item in variants if item.status == "succeeded"]
        constraints = _optimization_constraint_rows(configuration)
        rows.append(
            {
                "study": study,
                "settings": _optimization_setting_rows(
                    configuration, study.model_level
                ),
                "constraints": constraints,
                "variant_count": len(variants),
                "succeeded_count": len(succeeded),
                "variants": sorted(variants, key=lambda item: item.pk),
            }
        )
    return rows


def _optimizer_page_state(request):
    selected_id = request.GET.get("scenario_id")
    studies = (
        OptimizationStudy.objects.select_related("portfolio", "factor_build")
        .order_by("-created_at")[:20]
    )
    scenarios = (
        OptimizationScenario.objects.select_related("study", "portfolio")
        .order_by("-created_at")[:50]
    )
    # Results are shown only for a study or variant named in the URL.  Opening
    # or refreshing the page therefore starts on the builder, and stored runs
    # are opened deliberately from the history list.
    selected = (
        OptimizationScenario.objects.select_related("study", "portfolio")
        .filter(pk=selected_id)
        .first()
        if selected_id
        else None
    )
    selected_study_id = request.GET.get("study_id")
    selected_study = (
        OptimizationStudy.objects.filter(pk=selected_study_id).first()
        if selected_study_id
        else (selected.study if selected else None)
    )
    variants = (
        list(selected_study.variants.select_related("portfolio").order_by("id"))
        if selected_study
        else []
    )
    if selected_study and (selected is None or selected.study_id != selected_study.pk):
        selected = variants[0] if variants else None
    holdings = (
        selected.holdings.select_related("security").order_by("security__ticker")
        if selected
        else []
    )
    rows = []
    for item in holdings:
        trade = (
            (item.optimized_weight or 0) - (item.original_weight or 0)
            if item.optimized_weight is not None
            else None
        )
        rows.append({"holding": item, "trade": trade})
    portfolio_id = request.GET.get("portfolio_id", "")
    selected_portfolio = (
        Portfolio.objects.filter(pk=portfolio_id).first()
        if portfolio_id.isdigit()
        else (selected_study.portfolio if selected_study else None)
    )
    build_id = request.GET.get("factor_build_id", "")
    selected_build = (
        FactorBuild.objects.filter(
            pk=build_id, definition_version=PROXY_MODEL_VERSION
        ).first()
        if build_id.isdigit()
        else (selected_study.factor_build if selected_study else None)
    )
    if selected_build is None:
        selected_build = (
            FactorBuild.objects.filter(
                stock_fits__isnull=False,
                is_canonical=True,
                definition_version=PROXY_MODEL_VERSION,
            )
            .distinct()
            .order_by("-as_of", "-id")
            .first()
        )
    model_level = request.GET.get("model_level", "")
    if not model_level and selected_portfolio:
        model_level = selected_portfolio.configuration.get("stock_selection", {}).get(
            "model_level", ""
        )
    if not model_level:
        model_level = selected_study.model_level if selected_study else "base_sector"
    constraint_groups = catalog_factor_groups(selected_build, model_level)
    available_factors = [
        factor for group in constraint_groups for factor in group["factors"]
    ]
    selection_factors = set(
        (selected_portfolio.configuration if selected_portfolio else {})
        .get("stock_selection", {})
        .get("factor_weights", {})
    )
    default_premium_names = {"Market"} | selection_factors
    default_premiums = [
        {
            "factor": factor,
            "premium": 5 if factor.name == "Market" else 2,
        }
        for factor in available_factors
        if factor.name in default_premium_names
    ]
    default_premium_keys = {row["factor"].field_key for row in default_premiums}
    sector_options = (
        sorted(
            {
                item.security.sector
                for item in selected_portfolio.holdings.select_related("security")
                if item.security.sector
            }
        )
        if selected_portfolio
        else []
    )
    comparison = selected.comparison if selected else {}
    original_exposures = comparison.get("original", {}).get("exposures", {})
    optimized_exposures = comparison.get("optimized", {}).get("exposures", {}) or {}
    scenario_configuration = selected.configuration if selected else {}
    absolute_bounds = scenario_configuration.get("factor_bounds") or {}
    relative_bounds = scenario_configuration.get("relative_factor_bounds") or {}
    exposure_rows = []
    for name in sorted(set(original_exposures) | set(optimized_exposures)):
        original_beta = original_exposures.get(name)
        optimized_beta = optimized_exposures.get(name)
        bound = absolute_bounds.get(name) or relative_bounds.get(name)
        within = None
        if bound and optimized_beta is not None:
            within = bool(
                float(bound[0]) - 1e-6 <= optimized_beta <= float(bound[1]) + 1e-6
            )
        exposure_rows.append(
            {
                "name": name,
                "original": original_beta if original_beta is not None else 0,
                "optimized": optimized_beta if optimized_beta is not None else 0,
                "change": (optimized_beta or 0) - (original_beta or 0),
                "bound": _bound_text(bound) if bound else "",
                "bound_kind": (
                    "vs SPY"
                    if name in relative_bounds and name not in absolute_bounds
                    else ""
                ),
                "within": within,
            }
        )
    sector_rows = _optimizer_sector_rows(holdings, scenario_configuration)
    original_metrics = comparison.get("original", {}) if selected else {}
    comparison_rows = []
    if selected:
        original_return = original_metrics.get("expected_return")
        original_vol = original_metrics.get("expected_volatility")
        comparison_rows = [
            {
                "label": "Expected return",
                "original": original_return,
                "optimized": selected.expected_return,
                "change": (
                    None
                    if original_return is None or selected.expected_return is None
                    else selected.expected_return - original_return
                ),
            },
            {
                "label": "Expected volatility",
                "original": original_vol,
                "optimized": selected.expected_volatility,
                "change": (
                    None
                    if original_vol is None or selected.expected_volatility is None
                    else selected.expected_volatility - original_vol
                ),
            },
            {
                "label": "Turnover",
                "original": original_metrics.get("turnover", 0),
                "optimized": selected.turnover,
                "change": selected.turnover,
            },
        ]
    selected_backtest = None
    backtest_job = None
    optimizer_history = None
    if selected:
        selected_backtest = (
            selected.backtests.filter(status="succeeded")
            .order_by("-finished_at", "-id")
            .first()
        )
        backtest_job = (
            Job.objects.filter(
                kind="scenario_backtest",
                parameters__scenario_id=selected.pk,
                status__in=["queued", "running", "cancel_requested"],
            )
            .order_by("-id")
            .first()
        )
        if selected_backtest:
            optimizer_history = _optimizer_history_context(selected_backtest)
    selected_configuration = (
        selected.configuration
        if selected
        else (selected_study.configuration if selected_study else {})
    )
    return {
        "optimization_studies": studies,
        "optimizer_study_rows": _optimization_study_rows(studies),
        "optimizer_setting_rows": _optimization_setting_rows(
            selected_configuration,
            selected_study.model_level if selected_study else "",
        ),
        "optimizer_constraint_rows": _optimization_constraint_rows(
            selected_configuration
        ),
        "scenarios": scenarios,
        "selected_optimization_study": selected_study,
        "selected_scenario": selected,
        "selected_scenario_holdings": holdings,
        "selected_scenario_rows": rows,
        "selected_optimizer_portfolio_id": str(
            selected_portfolio.pk if selected_portfolio else portfolio_id
        ),
        "selected_optimizer_build_id": str(
            selected_build.pk if selected_build else build_id
        ),
        "selected_optimizer_model_level": model_level,
        "optimizer_model_levels": _model_level_options(selected_build),
        "optimizer_constraint_factor_groups": constraint_groups,
        "optimizer_default_premiums": default_premiums,
        "optimizer_default_premium_keys": default_premium_keys,
        "optimizer_sector_options": sector_options,
        "optimizer_exposure_rows": exposure_rows,
        "optimizer_sector_rows": sector_rows,
        "optimizer_comparison_rows": comparison_rows,
        "scenario_backtest_job": backtest_job,
        "selected_scenario_backtest": selected_backtest,
        "optimizer_history": optimizer_history,
        "optimizer_can_replay": bool(
            (
                (
                    selected.portfolio.configuration
                    if selected and selected.portfolio
                    else {}
                )
                or (
                    selected_study.portfolio.configuration
                    if selected_study and selected_study.portfolio_id
                    else {}
                )
            ).get("stock_selection")
        ),
    }


def _aggregate_factor_returns(observations, frequency: str) -> list[dict]:
    """Compound daily factor returns into chart-ready periods."""

    def bucket_key(item):
        stamp = item["date"]
        if frequency == "weekly":
            iso_year, iso_week, _ = stamp.isocalendar()
            return iso_year, iso_week
        if frequency == "monthly":
            return stamp.year, stamp.month
        return (stamp,)

    periods = []
    current_key = None
    compounded = 1.0
    latest_date = None
    latest_index = None
    for item in observations:
        if item["scaled_return"] is None:
            continue
        key = bucket_key(item)
        if current_key is not None and key != current_key:
            periods.append(
                {
                    "date": latest_date.isoformat(),
                    "return": compounded - 1.0,
                    "cumulative_index": latest_index,
                }
            )
            compounded = 1.0
        current_key = key
        compounded *= 1.0 + item["scaled_return"]
        latest_date = item["date"]
        latest_index = item["cumulative_index"]
    if current_key is not None:
        periods.append(
            {
                "date": latest_date.isoformat(),
                "return": compounded - 1.0,
                "cumulative_index": latest_index,
            }
        )
    return periods


def factor_return_detail(request, build_id: int, factor_id: int):
    """Return one factor's construction and frequency-aware return history."""
    build = FactorBuild.objects.filter(
        pk=build_id,
        is_canonical=True,
        definition_version=PROXY_MODEL_VERSION,
    ).first()
    if build is None:
        return JsonResponse({"error": "Factor build not found."}, status=404)
    factor = FactorDefinition.objects.filter(
        pk=factor_id, model_version=build.definition_version, active=True
    ).first()
    if factor is None:
        return JsonResponse({"error": "Factor not found."}, status=404)
    observations = list(
        FactorObservation.objects.filter(build=build, factor=factor)
        .order_by("date")
        .values("date", "scaled_return", "cumulative_index")
    )
    if not observations:
        return JsonResponse(
            {"error": "No return history is stored for this factor and build."},
            status=404,
        )
    construction = factor_construction_details(factor.name)
    return JsonResponse(
        {
            "factor": {
                "id": factor.pk,
                "name": factor.name,
                "family": factor.family,
                "description": construction["summary"],
                "provenance": factor.get_provenance_badge_display(),
                "construction": construction,
            },
            "build": {
                "id": build.pk,
                "label": build.display_label,
                "as_of": build.as_of.isoformat(),
            },
            "series": {
                frequency: _aggregate_factor_returns(observations, frequency)
                for frequency in ("daily", "weekly", "monthly")
            },
        }
    )


def _matching_signal_history_job(build_id: int, model_level: str, ticker: str):
    for job in Job.objects.filter(
        kind="monthly_exposures",
        status__in=["queued", "running", "cancel_requested"],
    ).order_by("-id"):
        parameters = job.parameters
        levels = parameters.get("levels", [])
        tickers = parameters.get("tickers", [])
        if (
            parameters.get("factor_build_id") == build_id
            and model_level in levels
            and ticker in tickers
            and int(parameters.get("max_months", 0)) >= 24
        ):
            return job
    return None


def _signals_page_state(request, model_level: str) -> dict:
    build_options = signal_build_options(model_level)
    selected_option = select_signal_build(
        build_options, request.GET.get("build_id", "")
    )
    selected_build = selected_option["build"] if selected_option else None
    groups = catalog_factor_groups(selected_build, model_level)
    factors = flatten_factor_groups(groups)
    requested_factor = request.GET.get("factor", "Market")
    selected_factor = next(
        (factor for factor in factors if factor.name == requested_factor),
        factors[0] if factors else None,
    )
    direction = -1 if request.GET.get("direction") == "-1" else 1
    sector = request.GET.get("sector", "").strip()
    ranking = (
        rank_factor_exposures(
            selected_build,
            model_level,
            selected_factor,
            direction=direction,
            sector=sector,
        )
        if selected_build and selected_factor
        else {
            "as_of": None,
            "rows": [],
            "chart_rows": [],
            "factor_observation": None,
            "stock_count": 0,
            "selected_count": 0,
            "sectors": [],
        }
    )
    family = request.GET.get("family", "").strip()
    visible_groups = (
        [group for group in groups if group["family"] == family] if family else groups
    )
    ticker = request.GET.get("ticker", "").strip().upper()
    history = (
        company_factor_history(
            selected_build,
            model_level,
            selected_factor,
            ticker,
            direction=direction,
        )
        if selected_build and selected_factor and ticker
        else []
    )
    history_job = None
    requested_job = request.GET.get("history_job_id", "")
    if requested_job.isdigit():
        history_job = Job.objects.filter(pk=requested_job).first()
    if not history_job and selected_build and ticker and len(history) < 24:
        history_job = _matching_signal_history_job(
            selected_build.pk, model_level, ticker
        )
    return {
        "signal_build_options": build_options,
        "selected_signal_build_option": selected_option,
        "selected_signal_build": selected_build,
        "signal_model_levels": _model_level_options(selected_build),
        "signal_factor_groups": visible_groups,
        "signal_factor_families": [group["family"] for group in groups],
        "selected_signal_family": family,
        "selected_signal_factor": selected_factor,
        "signal_direction": direction,
        "selected_signal_sector": sector,
        "signal_ranking": ranking,
        "signal_ranking_rows": ranking["rows"],
        "signal_chart_rows": ranking["chart_rows"],
        "selected_signal_ticker": ticker,
        "signal_history_rows": history,
        "signal_history_job": history_job,
        "signal_history_target": 24,
    }


DEFAULT_SELECTION_STYLE_FACTORS = ("Momentum", "Value", "Quality", "Growth")
DEFAULT_SELECTION_STYLE_WEIGHT = 25.0


def _default_selection_config() -> dict:
    model_level = "base_sector"
    option = select_signal_build(signal_build_options(model_level))
    build = option["build"] if option else None
    factors = compatible_factors(build, model_level) if build else []
    available = {factor.name for factor in factors}
    preferred = [
        name for name in DEFAULT_SELECTION_STYLE_FACTORS if name in available
    ]
    selected = preferred or [factor.name for factor in factors[:1]]
    if preferred:
        weights = {name: DEFAULT_SELECTION_STYLE_WEIGHT for name in selected}
    elif selected:
        weights = {name: round(100 / len(selected), 4) for name in selected}
    else:
        weights = {}
    return {
        "name": "V2 Factor Selection",
        "build_id": build.pk if build else None,
        "model_level": model_level,
        "factor_weights": weights,
        "directions": {name: 1 for name in selected},
        "top_n": 20,
        "sectors": [],
        "minimum_trading_days": 252,
        "minimum_coverage": None,
        "minimum_adjusted_r2": None,
        "manual_tickers": [],
        "excluded_tickers": [],
    }


def _screen_selection_config(screen: ScreenDefinition) -> dict:
    filters = screen.filters or {}
    trading_days, coverage = resolve_selection_filters(
        minimum_trading_days=filters.get("minimum_trading_days"),
        minimum_coverage=filters.get("minimum_coverage"),
    )
    return {
        "name": screen.name,
        "build_id": screen.factor_build_id,
        "model_level": screen.model_level,
        "factor_weights": screen.factor_weights,
        "directions": screen.directions,
        "top_n": screen.top_n,
        "sectors": filters.get("sectors", []),
        "minimum_trading_days": trading_days,
        "minimum_coverage": coverage,
        "minimum_adjusted_r2": filters.get("minimum_adjusted_r2"),
        "manual_tickers": filters.get("manual_tickers", []),
        "excluded_tickers": filters.get("excluded_tickers", []),
    }


def _selection_preview_from_config(config: dict) -> tuple[FactorBuild, dict]:
    build = FactorBuild.objects.get(pk=config["build_id"])
    preview = build_selection_preview(
        build=build,
        model_level=config["model_level"],
        factor_weights=config["factor_weights"],
        directions=config.get("directions"),
        top_n=config.get("top_n", 20),
        sectors=config.get("sectors"),
        minimum_trading_days=config.get("minimum_trading_days"),
        minimum_coverage=config.get("minimum_coverage"),
        minimum_adjusted_r2=config.get("minimum_adjusted_r2"),
        excluded_tickers=config.get("excluded_tickers", []),
        manual_tickers=config.get("manual_tickers", []),
    )
    return build, preview


def _stock_selection_page_state(request) -> dict:
    screens = ScreenDefinition.objects.filter(active=True).order_by("-updated_at")
    selected_id = request.GET.get("screen_id", "")
    selected_screen = (
        screens.filter(pk=selected_id).first() if selected_id.isdigit() else None
    )
    if selected_screen and request.GET.get("draft") != "1":
        config = _screen_selection_config(selected_screen)
        config["screen_id"] = selected_screen.pk
    elif request.GET.get("new") == "1":
        config = _default_selection_config()
        config["screen_id"] = None
    else:
        config = (
            request.session.get("stock_selection_preview")
            or _default_selection_config()
        )

    model_level = config.get("model_level", "base_sector")
    build_options = signal_build_options(model_level)
    selected_option = select_signal_build(
        build_options, str(config.get("build_id") or "")
    )
    selected_build = selected_option["build"] if selected_option else None
    if selected_build:
        config["build_id"] = selected_build.pk
    request.session["stock_selection_preview"] = config
    groups = catalog_factor_groups(selected_build, model_level)
    selected_weights = config.get("factor_weights", {})
    directions = config.get("directions", {})
    factor_groups = [
        {
            "family": group["family"],
            "factors": [
                {
                    "definition": factor,
                    "selected": factor.name in selected_weights,
                    "weight": selected_weights.get(factor.name, 0),
                    "direction": directions.get(factor.name, 1),
                }
                for factor in group["factors"]
            ],
        }
        for group in groups
    ]
    preview = None
    history = None
    history_error = ""
    error = ""
    if selected_build and selected_weights:
        try:
            _, preview = _selection_preview_from_config(config)
        except ValueError as exc:
            error = str(exc)
        if preview:
            try:
                history = _history_page_context(
                    replay_screen_history(selected_build, config)
                )
            except (FileNotFoundError, TypeError, ValueError) as exc:
                history_error = str(exc)
    return {
        "selection_screens": screens,
        "selected_selection_screen": selected_screen,
        "selection_config": config,
        "selection_build_options": build_options,
        "selected_selection_build": selected_build,
        "selection_model_levels": _model_level_options(selected_build),
        "selection_factor_groups": factor_groups,
        "selection_preview": preview,
        "selection_preview_rows": preview["rows"] if preview else [],
        "selection_included_rows": preview["included_rows"] if preview else [],
        "selection_error": error,
        "selection_history": history,
        "selection_history_error": history_error,
        "selection_portfolios": Portfolio.objects.filter(archived=False).order_by(
            "-updated_at"
        )[:10],
    }


def _format_backtest_metric(value, kind="number"):
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "percent":
        return f"{numeric * 100:.2f}%"
    return f"{numeric:.2f}"


def _history_page_context(result: dict) -> dict:
    portfolio = result["metrics"]
    benchmark = result["benchmark_metrics"]
    definitions = (
        ("Annualized return", "annual_return", "percent"),
        ("Annualized volatility", "annual_volatility", "percent"),
        ("Sharpe ratio", "sharpe", "number"),
        ("Sortino ratio", "sortino", "number"),
        ("Maximum drawdown", "max_drawdown", "percent"),
        ("Alpha vs SPY", "alpha", "percent"),
        ("Beta vs SPY", "beta", "number"),
        ("Tracking error", "tracking_error", "percent"),
        ("Total turnover", "turnover", "percent"),
    )
    return {
        **result,
        "metric_rows": [
            {
                "label": label,
                "portfolio": _format_backtest_metric(portfolio.get(key), kind),
                "spy": _format_backtest_metric(benchmark.get(key), kind),
            }
            for label, key, kind in definitions
        ],
    }


def _optimizer_history_context(run: BacktestRun) -> dict:
    payload = run.metrics or {}
    definitions = (
        ("Annualized return", "annual_return", "percent"),
        ("Annualized volatility", "annual_volatility", "percent"),
        ("Sharpe ratio", "sharpe", "number"),
        ("Sortino ratio", "sortino", "number"),
        ("Maximum drawdown", "max_drawdown", "percent"),
        ("Alpha vs SPY", "alpha", "percent"),
        ("Beta vs SPY", "beta", "number"),
        ("Tracking error", "tracking_error", "percent"),
        ("Total turnover", "turnover", "percent"),
    )
    original = payload.get("original", {})
    optimized = payload.get("optimized", {})
    spy = payload.get("spy", {})
    rebalances = []
    for item in payload.get("rebalances", []):
        original_weights = item.get("original_weights") or {}
        optimized_weights = item.get("optimized_weights") or {}
        tickers = item.get("tickers") or sorted(
            set(original_weights) | set(optimized_weights)
        )
        rebalances.append(
            {
                **item,
                "tickers": tickers,
                "holding_rows": [
                    {
                        "ticker": ticker,
                        "original": original_weights.get(ticker),
                        "optimized": optimized_weights.get(ticker),
                    }
                    for ticker in tickers
                ],
            }
        )
    return {
        "run": run,
        "start": payload.get("start"),
        "end": payload.get("end"),
        "observations": payload.get("observations"),
        "methodology": payload.get("methodology", ""),
        "warnings": payload.get("warnings", run.warnings or []),
        "chart_rows": payload.get("chart_rows", []),
        "rebalances": rebalances,
        "metric_rows": [
            {
                "label": label,
                "original": _format_backtest_metric(original.get(key), kind),
                "optimized": _format_backtest_metric(optimized.get(key), kind),
                "spy": _format_backtest_metric(spy.get(key), kind),
            }
            for label, key, kind in definitions
        ],
    }


def dashboard(request):
    return workspace_page(request, "factors")


def workspace_page(request, page: str):
    allowed = {
        "factors",
        "stock",
        "portfolio",
        "signals",
        "stock-selection",
        "optimizer",
    }
    if page in {"portfolios", "screens"}:
        return redirect("desk:page", page="stock-selection")
    if page == "stocks":
        query = {}
        for key in ("ticker", "model_level", "build_id", "beta_factor"):
            if request.GET.get(key):
                query[key] = request.GET[key]
        suffix = f"?{urlencode(query)}" if query else ""
        return redirect(f"{reverse('desk:page', args=['stock'])}{suffix}")
    if page not in allowed:
        raise Http404
    if page == "factors" and request.GET.get("tab") in {"stock", "portfolio"}:
        target = request.GET["tab"]
        query = request.GET.copy()
        query.pop("tab", None)
        suffix = f"?{query.urlencode()}" if query else ""
        return redirect(f"{reverse('desk:page', args=[target])}{suffix}")
    latest_build = (
        FactorBuild.objects.filter(
            is_canonical=True, definition_version=PROXY_MODEL_VERSION
        )
        .order_by("-as_of", "-id")
        .first()
    )
    factor_tab = {
        "factors": "universe",
        "stock": "stock",
        "portfolio": "portfolio",
    }.get(page, "")
    ticker = (
        request.GET.get("ticker", "").strip().upper() if factor_tab == "stock" else ""
    )
    default_model_level = "all_factors" if page == "signals" else "base_sector"
    model_level = (
        request.GET.get("model_level")
        or request.POST.get("model_level")
        or default_model_level
    )
    if model_level not in dict(MODEL_LEVELS):
        model_level = "base_sector"
    factor_state = _factor_page_state(request) if page == "factors" else {}
    if page in {"stock", "portfolio"}:
        _, selected_build = _selected_factor_build(request)
        if page == "portfolio":
            selected_build = _portfolio_build(request, model_level, selected_build)
    else:
        selected_build = factor_state.get("selected_factor_build", latest_build)
    signal_state = (
        _signals_page_state(request, model_level) if page == "signals" else {}
    )
    selection_state = (
        _stock_selection_page_state(request) if page == "stock-selection" else {}
    )
    stock_state = (
        _stock_page_state(
            ticker,
            selected_build,
            model_level,
            request.GET.get("beta_factor", "").strip(),
        )
        if factor_tab == "stock"
        else {
            "stock_fits": StockModelFit.objects.none(),
            "stock_history_rows": [],
            "stock_beta_history_rows": [],
            "stock_beta_factor_options": [],
            "selected_stock_beta_factor": "",
            "stock_exposure_rows": [],
            "selected_stock_fit": None,
            "stock_options": Security.objects.none(),
            "model_levels": _model_level_options(selected_build),
            "selected_model_level": model_level,
        }
    )
    portfolio_factor_state = (
        _portfolio_factor_state(request, selected_build, model_level)
        if factor_tab == "portfolio"
        else {}
    )
    optimizer_state = (
        _optimizer_page_state(request)
        if page == "optimizer"
        else {
            "selected_scenario": None,
            "selected_scenario_holdings": [],
            "selected_scenario_rows": [],
        }
    )
    portfolios = Portfolio.objects.filter(archived=False).order_by("name")
    latest_risk = {}
    for snapshot in PortfolioRiskSnapshot.objects.select_related("portfolio").order_by(
        "portfolio_id", "-as_of"
    ):
        latest_risk.setdefault(snapshot.portfolio_id, snapshot.as_of)
    portfolio_rows = [
        {"portfolio": item, "latest_risk_as_of": latest_risk.get(item.id)}
        for item in portfolios
    ]
    latest_screen_run = ScreenRun.objects.order_by("-created_at").first()
    return render(
        request,
        "desk/platform.html",
        {
            "page": page,
            "factor_workspace_tab": factor_tab,
            "price_snapshots": PriceSnapshot.objects.order_by("-created_at")[:20],
            "factor_builds": FactorBuild.objects.filter(
                is_canonical=True, definition_version=PROXY_MODEL_VERSION
            ).order_by("-as_of", "-id")[:20],
            "portfolios": portfolios,
            "portfolio_rows": portfolio_rows,
            "latest_portfolio_risk_dates": latest_risk,
            "screens": ScreenDefinition.objects.all().order_by("name"),
            "screen_runs": ScreenRun.objects.order_by("-created_at")[:20],
            "scenarios": (
                optimizer_state["scenarios"]
                if page == "optimizer"
                else OptimizationScenario.objects.order_by("-created_at")[:20]
            ),
            "jobs": Job.objects.order_by("-created_at")[:30],
            "ticker": ticker,
            "stock_fits": stock_state["stock_fits"],
            "stock_history_rows": stock_state["stock_history_rows"],
            "stock_beta_history_rows": stock_state["stock_beta_history_rows"],
            "stock_beta_factor_options": stock_state["stock_beta_factor_options"],
            "selected_stock_beta_factor": stock_state["selected_stock_beta_factor"],
            "stock_exposure_rows": stock_state["stock_exposure_rows"],
            "selected_stock_fit": stock_state["selected_stock_fit"],
            "stock_options": stock_state["stock_options"],
            "model_levels": stock_state["model_levels"],
            "selected_model_level": stock_state["selected_model_level"],
            "factor_observations": factor_state.get(
                "factor_observations",
                (
                    FactorObservation.objects.filter(
                        build=latest_build, date=latest_build.as_of
                    ).select_related("factor")
                    if latest_build
                    else FactorObservation.objects.none()
                ),
            ),
            "factor_history": factor_state.get(
                "factor_history",
                (
                    FactorObservation.objects.filter(
                        build=latest_build, factor__name="Market"
                    ).order_by("-date")[:126]
                    if latest_build
                    else []
                ),
            ),
            "factor_panel_rows": factor_state.get("factor_panel_rows", []),
            "factor_panel_windows": factor_state.get(
                "factor_panel_windows", FACTOR_PANEL_WINDOWS
            ),
            "factor_panel_column_count": factor_state.get(
                "factor_panel_column_count", 3 + 2 * len(FACTOR_PANEL_WINDOWS)
            ),
            "selected_factor_build": factor_state.get(
                "selected_factor_build", selected_build
            ),
            "selected_factor": factor_state.get("selected_factor", "Market"),
            "selected_scenario": optimizer_state.get("selected_scenario"),
            "selected_scenario_holdings": optimizer_state.get(
                "selected_scenario_holdings", []
            ),
            "selected_scenario_rows": optimizer_state.get("selected_scenario_rows", []),
            "selected_optimizer_portfolio_id": optimizer_state.get(
                "selected_optimizer_portfolio_id", ""
            ),
            "selected_optimizer_build_id": optimizer_state.get(
                "selected_optimizer_build_id", ""
            ),
            "selected_optimizer_model_level": optimizer_state.get(
                "selected_optimizer_model_level", "base_sector"
            ),
            "optimizer_model_levels": optimizer_state.get("optimizer_model_levels", []),
            "optimization_studies": optimizer_state.get("optimization_studies", []),
            "optimizer_study_rows": optimizer_state.get("optimizer_study_rows", []),
            "optimizer_setting_rows": optimizer_state.get("optimizer_setting_rows", []),
            "optimizer_constraint_rows": optimizer_state.get(
                "optimizer_constraint_rows", []
            ),
            "selected_optimization_study": optimizer_state.get(
                "selected_optimization_study"
            ),
            "optimizer_constraint_factor_groups": optimizer_state.get(
                "optimizer_constraint_factor_groups", []
            ),
            "optimizer_default_premiums": optimizer_state.get(
                "optimizer_default_premiums", []
            ),
            "optimizer_default_premium_keys": optimizer_state.get(
                "optimizer_default_premium_keys", set()
            ),
            "optimizer_sector_options": optimizer_state.get(
                "optimizer_sector_options", []
            ),
            "optimizer_exposure_rows": optimizer_state.get(
                "optimizer_exposure_rows", []
            ),
            "optimizer_sector_rows": optimizer_state.get("optimizer_sector_rows", []),
            "optimizer_comparison_rows": optimizer_state.get(
                "optimizer_comparison_rows", []
            ),
            "scenario_backtest_job": optimizer_state.get("scenario_backtest_job"),
            "selected_scenario_backtest": optimizer_state.get(
                "selected_scenario_backtest"
            ),
            "optimizer_history": optimizer_state.get("optimizer_history"),
            "optimizer_can_replay": optimizer_state.get("optimizer_can_replay", False),
            "screen_results": (
                latest_screen_run.results.select_related("security").order_by("rank")
                if latest_screen_run
                else []
            ),
            **portfolio_factor_state,
            **factor_state,
            **signal_state,
            **selection_state,
        },
    )


@require_POST
def preview_stock_selection(request):
    try:
        build = FactorBuild.objects.get(
            pk=int(request.POST["build_id"]),
            definition_version=PROXY_MODEL_VERSION,
        )
        model_level = request.POST.get("model_level", "base_sector")
        if model_level not in dict(MODEL_LEVELS):
            raise ValueError("Unknown model level")
        weights = {}
        directions = {}
        for factor in compatible_factors(build, model_level):
            if _factor_field(request.POST, "factor_", factor) != "on":
                continue
            weight = float(_factor_field(request.POST, "weight_", factor) or 1)
            if weight <= 0:
                raise ValueError(f"{factor.name}: weight must be greater than zero")
            weights[factor.name] = weight
            directions[factor.name] = (
                -1 if _factor_field(request.POST, "direction_", factor) == "-1" else 1
            )

        def optional_float(name):
            value = request.POST.get(name, "").strip()
            return float(value) if value else None

        def optional_int(name):
            value = request.POST.get(name, "").strip()
            return int(value) if value else None

        config = {
            "screen_id": (
                int(request.POST["screen_id"])
                if request.POST.get("screen_id", "").isdigit()
                else None
            ),
            "name": request.POST.get("name", "V2 Factor Selection").strip(),
            "build_id": build.pk,
            "model_level": model_level,
            "factor_weights": weights,
            "directions": directions,
            "top_n": int(request.POST.get("top_n", 20)),
            "sectors": request.POST.getlist("sectors"),
            "minimum_trading_days": optional_int("minimum_trading_days"),
            "minimum_coverage": optional_float("minimum_coverage"),
            "minimum_adjusted_r2": optional_float("minimum_adjusted_r2"),
            "manual_tickers": parse_ticker_list(request.POST.get("manual_tickers", "")),
            "excluded_tickers": parse_ticker_list(
                request.POST.get("excluded_tickers", "")
            ),
        }
        _selection_preview_from_config(config)
        request.session["stock_selection_preview"] = config
    except (KeyError, TypeError, ValueError, FactorBuild.DoesNotExist) as exc:
        return HttpResponse(str(exc), status=400)
    query = {"draft": 1}
    if config["screen_id"]:
        query["screen_id"] = config["screen_id"]
    return redirect(
        f"{reverse('desk:page', args=['stock-selection'])}?{urlencode(query)}"
    )


@require_POST
def save_stock_selection_screen(request):
    config = request.session.get("stock_selection_preview")
    if not config:
        return HttpResponse("Preview a stock selection before saving it", status=400)
    try:
        build, preview = _selection_preview_from_config(config)
        name = request.POST.get("name", config.get("name", "")).strip()
        if not name:
            raise ValueError("Screen name is required")
        screen_id = config.get("screen_id")
        duplicate = ScreenDefinition.objects.filter(name=name)
        if screen_id:
            duplicate = duplicate.exclude(pk=screen_id)
        if duplicate.exists():
            raise ValueError(f"A screen named '{name}' already exists")
        screen = (
            get_object_or_404(ScreenDefinition, pk=screen_id)
            if screen_id
            else ScreenDefinition()
        )
        screen.name = name
        screen.description = "Weighted V2 factor beta z-score selection"
        screen.factor_build = build
        screen.as_of = preview["as_of"]
        screen.model_level = config["model_level"]
        screen.factor_weights = config["factor_weights"]
        screen.directions = config["directions"]
        screen.filters = {
            "sectors": config.get("sectors", []),
            "minimum_trading_days": config.get("minimum_trading_days"),
            "minimum_coverage": config.get("minimum_coverage"),
            "minimum_adjusted_r2": config.get("minimum_adjusted_r2"),
            "manual_tickers": config.get("manual_tickers", []),
            "excluded_tickers": config.get("excluded_tickers", []),
        }
        screen.top_n = config["top_n"]
        screen.active = True
        screen.save()
        persist_selection_run(screen, preview)
        config["screen_id"] = screen.pk
        config["name"] = name
        request.session["stock_selection_preview"] = config
    except (TypeError, ValueError, FactorBuild.DoesNotExist) as exc:
        return HttpResponse(str(exc), status=400)
    return redirect(
        f"{reverse('desk:page', args=['stock-selection'])}?{urlencode({'screen_id': screen.pk})}"
    )


def _refresh_portfolio_risk(portfolio, build, model_level="") -> bool:
    """Store a risk snapshot so factor risk is available without a manual refresh.

    Best effort: a portfolio is still worth keeping when its covariance inputs
    are incomplete, and the exposure page already renders without a snapshot.
    """
    from .workflows import monitor_portfolio

    parameters = {"portfolio_id": portfolio.pk, "factor_build_id": build.pk}
    if model_level:
        parameters["model_level"] = model_level
    try:
        monitor_portfolio(parameters, lambda *_: None)
    except Exception:
        logger.warning(
            "Could not store a risk snapshot for portfolio %s", portfolio.pk,
            exc_info=True,
        )
        return False
    return True


@require_POST
def create_stock_selection_portfolio(request):
    config = request.session.get("stock_selection_preview")
    if not config:
        return HttpResponse(
            "Preview a stock selection before creating a portfolio", status=400
        )
    try:
        build, preview = _selection_preview_from_config(config)
        name = request.POST.get("name", "").strip()
        if not name:
            raise ValueError("Portfolio name is required")
        portfolio = save_selection_portfolio(name, preview)
        portfolio.configuration = {
            **portfolio.configuration,
            "stock_selection": {
                "screen_id": config.get("screen_id"),
                "factor_build_id": build.pk,
                "model_level": config["model_level"],
                "factor_weights": config["factor_weights"],
                "directions": config["directions"],
                "top_n": config["top_n"],
                "sectors": config.get("sectors", []),
                "minimum_trading_days": config.get("minimum_trading_days"),
                "minimum_coverage": config.get("minimum_coverage"),
                "minimum_adjusted_r2": config.get("minimum_adjusted_r2"),
                "manual_tickers": config.get("manual_tickers", []),
                "excluded_tickers": config.get("excluded_tickers", []),
            },
        }
        portfolio.save(update_fields=["configuration", "updated_at"])
    except (TypeError, ValueError, FactorBuild.DoesNotExist) as exc:
        return HttpResponse(str(exc), status=400)
    _refresh_portfolio_risk(portfolio, build, config["model_level"])

    action = request.POST.get("action", "factors")
    if action == "optimizer":
        query = {
            "portfolio_id": portfolio.pk,
            "factor_build_id": build.pk,
            "model_level": config["model_level"],
        }
        return redirect(
            f"{reverse('desk:page', args=['optimizer'])}?{urlencode(query)}"
        )
    query = {
        "portfolio_tab": "exposure",
        "build_id": build.pk,
        "portfolio_id": portfolio.pk,
        "model_level": config["model_level"],
    }
    return redirect(f"{reverse('desk:page', args=['portfolio'])}?{urlencode(query)}")


@require_POST
def create_portfolio(request):
    try:
        content = request.POST.get("holdings", "")
        if request.FILES.get("csv"):
            content = request.FILES["csv"].read().decode("utf-8-sig")
        rows = parse_holdings_csv(content)
        portfolio = save_portfolio(
            request.POST["name"].strip(),
            rows,
            equal_weight=request.POST.get("equal_weight") == "on",
        )
    except (KeyError, ValueError) as exc:
        return HttpResponse(str(exc), status=400)
    return redirect(
        f"{reverse('desk:page', args=['portfolio'])}?{urlencode({'portfolio_id': portfolio.pk})}"
    )


@require_POST
def create_manual_portfolio(request):
    try:
        name = request.POST.get("name", "").strip()
        if not name:
            raise ValueError("Portfolio name is required")
        if Portfolio.objects.filter(name=name).exists():
            raise ValueError(f"A portfolio named '{name}' already exists")
        tickers = parse_ticker_list(request.POST.get("tickers", ""))
        if not tickers:
            raise ValueError("Enter at least one ticker")
        portfolio = save_portfolio(
            name,
            [{"ticker": ticker, "weight": 1.0} for ticker in tickers],
            source="manual",
            equal_weight=True,
        )
    except ValueError as exc:
        return HttpResponse(str(exc), status=400)
    return redirect(
        f"{reverse('desk:page', args=['portfolio'])}?{urlencode({'portfolio_id': portfolio.pk})}"
    )


@require_POST
def edit_portfolio(request, pk: int):
    portfolio = get_object_or_404(Portfolio, pk=pk)
    try:
        content = request.POST.get("holdings", "")
        if request.FILES.get("csv"):
            content = request.FILES["csv"].read().decode("utf-8-sig")
        save_portfolio(
            request.POST.get("name", portfolio.name).strip(),
            parse_holdings_csv(content),
            portfolio=portfolio,
            equal_weight=request.POST.get("equal_weight") == "on",
        )
    except ValueError as exc:
        return HttpResponse(str(exc), status=400)
    return redirect(
        f"{reverse('desk:page', args=['portfolio'])}?{urlencode({'portfolio_id': portfolio.pk})}"
    )


@require_POST
def clone_saved_portfolio(request, pk: int):
    portfolio = get_object_or_404(Portfolio, pk=pk)
    clone = clone_portfolio(
        portfolio, request.POST.get("name", f"{portfolio.name} Copy")
    )
    return redirect(
        f"{reverse('desk:page', args=['portfolio'])}?{urlencode({'portfolio_id': clone.pk})}"
    )


@require_POST
def archive_portfolio(request, pk: int):
    portfolio = get_object_or_404(Portfolio, pk=pk)
    portfolio.archived = True
    portfolio.save(update_fields=["archived"])
    return redirect("desk:page", page="stock-selection")


@require_POST
def create_screen(request):
    try:
        ScreenDefinition.objects.create(
            name=request.POST["name"].strip(),
            factor_build_id=int(request.POST["factor_build_id"]),
            as_of=request.POST.get("as_of") or None,
            model_level=request.POST.get("model_level", "base_sector"),
            factor_weights=json.loads(request.POST.get("factor_weights", "{}")),
            directions=json.loads(request.POST.get("directions", "{}")),
            regime_interactions=json.loads(
                request.POST.get("regime_interactions", "{}")
            ),
            filters=json.loads(request.POST.get("filters", "{}")),
            top_n=min(50, max(10, int(request.POST.get("top_n", 20)))),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return HttpResponse(str(exc), status=400)
    return redirect("desk:page", page="screens")


@require_POST
def create_portfolio_from_screen(request, pk: int):
    run = get_object_or_404(ScreenRun, pk=pk)
    try:
        portfolio_from_screen(
            run,
            request.POST.get("name", f"{run.definition.name} Portfolio"),
            int(request.POST.get("top_n", run.definition.top_n)),
        )
    except ValueError as exc:
        return HttpResponse(str(exc), status=400)
    return redirect("desk:page", page="portfolios")


def portfolio_detail(request, pk: int):
    portfolio = get_object_or_404(Portfolio, pk=pk)
    holdings = list(
        portfolio.holdings.select_related("security").order_by("security__ticker")
    )
    latest_risk = (
        portfolio.risk_snapshots.select_related("factor_build")
        .order_by("-as_of")
        .first()
    )
    scenarios = portfolio.optimization_scenarios.order_by("-created_at")[:20]
    scenario_id = request.GET.get("scenario_id")
    selected_scenario = (
        scenarios.filter(pk=scenario_id).first() if scenario_id else scenarios.first()
    )
    holding_rows = []
    exposure_rows = []
    specific_contribution = None
    specific_risk_percent = None
    if latest_risk and latest_risk.predicted_volatility:
        specific_contribution = (
            latest_risk.specific_variance / latest_risk.predicted_volatility
        )
        specific_risk_percent = (
            100 * specific_contribution / latest_risk.predicted_volatility
        )
    if latest_risk and latest_risk.factor_build_id:
        fits = (
            StockModelFit.objects.filter(
                build=latest_risk.factor_build,
                security__ticker__in=[item.security.ticker for item in holdings],
                as_of__lte=latest_risk.as_of,
            )
            .order_by("security_id", "-as_of")
            .prefetch_related("exposures__factor")
        )
        latest_fit = {}
        for fit in fits:
            latest_fit.setdefault(fit.security_id, fit)
        for holding in holdings:
            fit = latest_fit.get(holding.security_id)
            exposures = (
                {item.factor.name: item.beta for item in fit.exposures.all()}
                if fit
                else {}
            )
            holding_rows.append(
                {
                    "ticker": holding.security.ticker,
                    "name": holding.security.name,
                    "weight": holding.weight,
                    "shares": holding.shares,
                    "alpha": fit.alpha if fit else None,
                    "adjusted_r2": fit.adjusted_r2 if fit else None,
                    "model_level": fit.model_level if fit else None,
                    "exposures": exposures,
                }
            )
        volatility = latest_risk.predicted_volatility
        for factor, value in latest_risk.exposures.items():
            contribution = latest_risk.component_risk.get(factor)
            exposure_rows.append(
                {
                    "factor": factor,
                    "exposure": value,
                    "active_exposure": latest_risk.active_exposures.get(factor),
                    "component_risk": contribution,
                    "marginal_risk": latest_risk.marginal_risk.get(factor),
                    "risk_percent": (
                        100 * contribution / volatility
                        if contribution is not None and volatility
                        else None
                    ),
                }
            )
        exposure_rows.sort(
            key=lambda row: (
                row["component_risk"] if row["component_risk"] is not None else -math.inf
            ),
            reverse=True,
        )
    return render(
        request,
        "desk/platform.html",
        {
            "page": "portfolio_detail",
            "portfolio_detail": portfolio,
            "portfolio_holdings_rows": holding_rows,
            "portfolio_latest_risk": latest_risk,
            "portfolio_exposure_rows": exposure_rows,
            "portfolio_specific_contribution": specific_contribution,
            "portfolio_specific_risk_percent": specific_risk_percent,
            "portfolio_scenarios": scenarios,
            "portfolio_selected_scenario": selected_scenario,
            "portfolio_selected_scenario_holdings": (
                selected_scenario.holdings.select_related("security").order_by(
                    "security__ticker"
                )
                if selected_scenario
                else []
            ),
            "factor_builds": FactorBuild.objects.filter(
                definition_version=PROXY_MODEL_VERSION
            ).order_by("-id")[:20],
            "factor_definitions": FactorDefinition.objects.filter(
                active=True, model_version=PROXY_MODEL_VERSION
            ).order_by("sort_order"),
            "jobs": Job.objects.order_by("-created_at")[:30],
        },
    )


def _factor_field(post, prefix: str, factor) -> str:
    """Read a per-factor form field, tolerating forms rendered for another model."""
    value = post.get(f"{prefix}{factor.field_key}")
    return post.get(f"{prefix}{factor.pk}", "") if value is None else value


def _parse_optimization_parameters(request) -> dict:
    parameters = {}
    int_fields = {"portfolio_id", "factor_build_id", "lookback"}
    percent_fields = {
        "min_weight",
        "max_weight",
        "max_weight_cap",
        "turnover_cap",
        "tracking_error_limit",
        "residual_shrinkage",
        "turnover_penalty",
        "transaction_cost",
        "risk_free_rate",
    }
    float_fields = {
        "max_factor_variance",
        "risk_aversion",
        "common_sharpe",
    }
    for key in int_fields:
        if request.POST.get(key, "").strip():
            parameters[key] = int(request.POST[key])
    for key in float_fields:
        if request.POST.get(key, "").strip():
            parameters[key] = float(request.POST[key])
    for key in percent_fields:
        if request.POST.get(key, "").strip():
            parameters[key] = float(request.POST[key]) / 100
    for key in (
        "name",
        "model_level",
        "objective",
        "expected_return_model",
        "weight_constraint_mode",
    ):
        if request.POST.get(key, "").strip():
            parameters[key] = request.POST[key].strip()
    parameters["risk_model"] = "factor_model"
    if request.POST.get("expected_returns", "").strip():
        supplied_returns = json.loads(request.POST["expected_returns"])
        parameters["expected_returns"] = {
            ticker: float(value) / 100 for ticker, value in supplied_returns.items()
        }
    build = FactorBuild.objects.get(
        pk=parameters["factor_build_id"],
        definition_version=PROXY_MODEL_VERSION,
    )
    model_level = parameters.get("model_level", "base_sector")
    if model_level not in dict(MODEL_LEVELS):
        raise ValueError("Unknown model level")
    factors = compatible_factors(build, model_level)

    def optional_float(name):
        value = request.POST.get(name, "").strip()
        return float(value) if value else None

    def factor_float(prefix, factor):
        value = _factor_field(request.POST, prefix, factor).strip()
        return float(value) if value else None

    factor_bounds = {}
    relative_bounds = {}
    component_caps = {}
    risk_budgets = {}
    factor_premia = {}
    for factor in factors:
        lower = factor_float("factor_min_", factor)
        upper = factor_float("factor_max_", factor)
        if lower is not None or upper is not None:
            factor_bounds[factor.name] = [
                lower if lower is not None else -1_000_000,
                upper if upper is not None else 1_000_000,
            ]
        lower = factor_float("relative_min_", factor)
        upper = factor_float("relative_max_", factor)
        if lower is not None or upper is not None:
            relative_bounds[factor.name] = [
                lower if lower is not None else -1_000_000,
                upper if upper is not None else 1_000_000,
            ]
        cap = factor_float("component_cap_", factor)
        if cap is not None:
            component_caps[factor.name] = cap
        budget = factor_float("risk_budget_", factor)
        if budget is not None:
            risk_budgets[factor.name] = budget / 100
        premium = factor_float("factor_premium_", factor)
        if premium is not None:
            factor_premia[factor.name] = premium / 100
    if factor_bounds:
        parameters["factor_bounds"] = factor_bounds
    if relative_bounds:
        parameters["relative_factor_bounds"] = relative_bounds
    if component_caps:
        parameters["max_factor_components"] = component_caps
    if risk_budgets:
        parameters["risk_budgets"] = risk_budgets
    if factor_premia:
        parameters["factor_premia"] = factor_premia
    if parameters.get("expected_return_model") == "factor_premium":
        if not factor_premia:
            raise ValueError("Add at least one factor premium")
        parameters["shrink_factor_betas"] = (
            request.POST.get("shrink_factor_betas") == "on"
        )
    if parameters.get("expected_return_model") == "equal_sharpe":
        if parameters.get("common_sharpe") is None:
            parameters["common_sharpe"] = 0.5
        if parameters["common_sharpe"] <= 0:
            raise ValueError("Common Sharpe must be greater than zero")
    sector_bounds = {}
    for index in range(1, 51):
        sector = request.POST.get(f"sector_name_{index}", "").strip()
        if not sector:
            continue
        lower = optional_float(f"sector_min_{index}")
        upper = optional_float(f"sector_max_{index}")
        if lower is not None or upper is not None:
            sector_bounds[sector] = [
                lower / 100 if lower is not None else 0,
                upper / 100 if upper is not None else 1,
            ]
    if sector_bounds:
        parameters["sector_bounds"] = sector_bounds
    for name, bounds in {
        **parameters.get("factor_bounds", {}),
        **parameters.get("relative_factor_bounds", {}),
        **parameters.get("sector_bounds", {}),
    }.items():
        if bounds[0] > bounds[1]:
            raise ValueError(f"{name}: minimum cannot exceed maximum")
    return parameters


@require_POST
def launch_domain_job(request, kind: str):
    allowed = {
        "monthly_exposures",
        "screen_run",
        "optimization",
        "risk_refresh",
        "proxy_factor_build",
        "canonical_update",
        "factor_catalog_sync",
    }
    if kind not in allowed:
        raise Http404
    if kind == "optimization":
        try:
            parameters = _parse_optimization_parameters(request)
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            FactorBuild.DoesNotExist,
        ) as exc:
            return HttpResponse(str(exc), status=400)
        job = submit_job(kind, parameters)
        return render(
            request,
            "desk/job.html",
            {"job": job, "reload_on_success": True},
        )
    parameters = {}
    json_fields = {
        "factor_bounds",
        "relative_factor_bounds",
        "risk_budgets",
        "max_factor_components",
        "expected_returns",
        "sector_bounds",
    }
    float_fields = {
        "max_weight",
        "turnover_cap",
        "tracking_error_limit",
        "max_factor_variance",
        "residual_shrinkage",
        "turnover_penalty",
        "transaction_cost",
    }
    int_fields = {
        "weekly_window",
        "weekly_min_periods",
        "volatility_window",
        "momentum_lookback",
        "momentum_skip",
        "characteristic_window",
        "growth_beta_window",
        "minimum_observations",
        "trailing_days",
        "estimation_window",
        "max_months",
        "top_n_by_market_cap",
        "workers",
    }
    try:
        for key, value in request.POST.items():
            if key == "csrfmiddlewaretoken" or value == "":
                continue
            if key.endswith("_id") or key in int_fields:
                parameters[key] = int(value)
            elif key in float_fields:
                parameters[key] = float(value)
            elif key in json_fields:
                parameters[key] = json.loads(value)
            elif key in {
                "levels",
                "cost_scenarios",
                "factors",
                "portfolio_ids",
                "tickers",
            }:
                parameters[key] = [
                    item.strip() for item in value.split(",") if item.strip()
                ]
            else:
                parameters[key] = value
    except (ValueError, json.JSONDecodeError) as exc:
        return HttpResponse(str(exc), status=400)
    job = submit_job(kind, parameters)
    return render(request, "desk/job.html", {"job": job})


@require_POST
def open_signal_company(request):
    build = get_object_or_404(FactorBuild, pk=request.POST.get("build_id"))
    ticker = request.POST.get("ticker", "").strip().upper()
    model_level = request.POST.get("model_level", "base_sector")
    if model_level not in dict(MODEL_LEVELS):
        return HttpResponse("Unknown model level", status=400)
    if not Security.objects.filter(ticker=ticker, asset_type="stock").exists():
        return HttpResponse("Unknown stock ticker", status=400)

    month_count = (
        StockModelFit.objects.filter(
            build=build,
            security__ticker=ticker,
            model_level=model_level,
        )
        .values("period")
        .distinct()
        .count()
    )
    history_job = None
    if month_count < 24:
        history_job = _matching_signal_history_job(build.pk, model_level, ticker)
        if history_job is None:
            history_job = submit_job(
                "monthly_exposures",
                {
                    "factor_build_id": build.pk,
                    "levels": [model_level],
                    "tickers": [ticker],
                    "trailing_days": 756,
                    "max_months": 24,
                    "update_mode": "backfill",
                },
            )
    query = {
        "build_id": build.pk,
        "model_level": model_level,
        "factor": request.POST.get("factor", "Market"),
        "direction": request.POST.get("direction", "1"),
        "ticker": ticker,
    }
    if request.POST.get("family"):
        query["family"] = request.POST["family"]
    if request.POST.get("sector"):
        query["sector"] = request.POST["sector"]
    if history_job:
        query["history_job_id"] = history_job.pk
    return redirect(f"{reverse('desk:page', args=['signals'])}?{urlencode(query)}")


@require_POST
def cancel_job(request, pk: int):
    request_cancellation(get_object_or_404(Job, pk=pk))
    return redirect("desk:page", page="factors")


def _job_success_url(job):
    """Page a finished job should open, when its result names one.

    Optimizer results render only for a study named in the URL, so a completed
    run has to land on its own study instead of reloading the empty builder it
    was queued from.
    """
    result = job.result or {}
    if job.kind != "optimization" or not result.get("study_id"):
        return None
    url = f"{reverse('desk:page', args=['optimizer'])}?study_id={result['study_id']}"
    if result.get("scenario_id"):
        url = f"{url}&scenario_id={result['scenario_id']}"
    return url


def job_status(request, pk: int):
    job = get_object_or_404(Job, pk=pk)
    return render(
        request,
        "desk/job.html",
        {
            "job": job,
            "reload_on_success": request.GET.get("reload_on_success") == "1",
            "job_success_url": _job_success_url(job),
        },
    )


def export_run_json(request, kind: str, pk: int):
    model = {
        "scenario": OptimizationScenario,
        "factor_build": FactorBuild,
        "risk_snapshot": PortfolioRiskSnapshot,
    }.get(kind)
    if model is None:
        raise Http404
    run = get_object_or_404(model, pk=pk)
    payload = {field.name: getattr(run, field.name) for field in run._meta.fields}
    return JsonResponse(payload, json_dumps_params={"default": str})


def export_holdings_csv(request, pk: int):
    run = get_object_or_404(OptimizationScenario, pk=pk)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="optimization_{pk}_holdings.csv"'
    )
    writer = csv.writer(response)
    writer.writerow(["ticker", "weight"])
    writer.writerows(
        (item.security.ticker, item.optimized_weight)
        for item in run.holdings.select_related("security")
    )
    return response


def export_portfolio_csv(request, pk: int):
    portfolio = get_object_or_404(Portfolio, pk=pk)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="portfolio_{pk}_holdings.csv"'
    )
    writer = csv.writer(response)
    writer.writerow(["ticker", "weight", "shares"])
    writer.writerows(
        (item.security.ticker, item.weight, item.shares)
        for item in portfolio.holdings.select_related("security").order_by(
            "security__ticker"
        )
    )
    return response


@require_POST
def launch_scenario_backtest(request, pk: int):
    scenario = get_object_or_404(
        OptimizationScenario.objects.select_related("study", "portfolio"), pk=pk
    )
    if scenario.status != "succeeded":
        return HttpResponse(
            "Only succeeded variants can be compared historically", status=400
        )
    portfolio = scenario.portfolio
    selection = (
        (portfolio.configuration or {}).get("stock_selection") if portfolio else {}
    )
    if not selection:
        return HttpResponse(
            "This 24-month comparison needs a Stock Selection portfolio so each "
            "month can use that month's ranked names.",
            status=400,
        )
    lookback_months = 24
    build_id = int(
        request.POST.get("factor_build_id")
        or selection.get("factor_build_id")
        or scenario.study.factor_build_id
    )
    idempotency_key = f"scenario-backtest:{scenario.pk}:{build_id}:{lookback_months}"
    inflight = Job.objects.filter(
        kind="scenario_backtest",
        idempotency_key=idempotency_key,
        status__in=["queued", "running", "cancel_requested"],
    ).first()
    if inflight:
        job = inflight
    else:
        succeeded = BacktestRun.objects.filter(
            optimization_scenario=scenario,
            lookback_months=lookback_months,
            status="succeeded",
        ).first()
        if succeeded:
            return redirect(
                f"{reverse('desk:page', args=['optimizer'])}"
                f"?study_id={scenario.study_id}&scenario_id={scenario.pk}"
            )
        job = submit_job(
            "scenario_backtest",
            {
                "scenario_id": scenario.pk,
                "factor_build_id": build_id,
                "lookback_months": lookback_months,
            },
            idempotency_key=idempotency_key,
        )
    return render(
        request,
        "desk/job.html",
        {"job": job, "reload_on_success": True},
    )
