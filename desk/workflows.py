from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import models, transaction
from django.utils import timezone

from backtests.engine import performance_statistics, screen_backtest_statistics
from backtests.factor_strategy import monthly_event_backtest
from backtests.models import BacktestRebalance, BacktestRun
from factors.catalog import (
    PROXY_CORE_FACTORS,
    PROXY_MODEL_VERSION,
    REVERSE_EXTERNAL_NAMES,
    SECTOR_FACTORS,
    sync_factor_catalogs,
)
from factors.engine import (
    FACTOR_HORIZONS,
    factor_covariance,
    portfolio_decomposition,
    horizon_returns,
    rolling_zscores,
)
from factors.exposure_workers import run_exposure_estimate
from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorEquation,
    FactorModelCatalog,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)
from factors.proxyfactorlib import (
    STYLE_SPECS as PROXY_STYLE_SPECS,
)
from factors.proxyfactorlib import (
    THEME_BASKETS,
)
from factors.proxyfactorlib import (
    build_model as build_proxy_factor_model,
)
from factors.versioning import canonical_model_fingerprint
from market_data.models import (
    Artifact,
    DataSnapshot,
    PriceSnapshot,
    RiskFreeRate,
    Security,
)
from market_data.providers import synthetic_dataset
from market_data.rates import risk_free_rate_on
from market_data.store import load_price_dataset
from optimization.engine import (
    factor_risk_covariance,
    optimize_portfolio,
)
from optimization.models import (
    OptimizationHolding,
    OptimizationScenario,
    OptimizationStudy,
)
from selection.engine import composite_scores, preview_screen
from selection.models import (
    BreachEvent,
    Portfolio,
    PortfolioRiskSnapshot,
    PortfolioSnapshot,
    RiskLimit,
    ScreenDefinition,
    ScreenResult,
    ScreenRun,
)
from selection.stock_selection import parse_ticker_list

# Standard deviation of the prior on each annual factor premium.  Trailing
# factor means carry standard errors near 6%, so this leaves roughly a fifth of
# the weight on history and keeps premiums inside a defensible range.
PREMIUM_PRIOR_DISPERSION = 0.03
DEFAULT_COMMON_SHARPE = 0.5


def _json(value):
    if isinstance(value, pd.Series):
        return {str(key): _json(item) for key, item in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return _json(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dataset_for_build(build: FactorBuild):
    return load_price_dataset()


def _portfolio_weights(portfolio: Portfolio, dataset, as_of: date) -> pd.Series:
    holdings = list(portfolio.holdings.select_related("security"))
    explicit = pd.Series(
        {
            item.security.ticker: item.weight
            for item in holdings
            if item.weight is not None
        },
        dtype=float,
    )
    if len(explicit) == len(holdings) and explicit.sum() > 0:
        weights = explicit / explicit.sum()
        total_value = None
    else:
        price_history = dataset.prices.loc[dataset.prices.index.date <= as_of]
        if price_history.empty:
            raise ValueError(f"No prices available on or before {as_of}")
        prices = price_history.ffill().iloc[-1]
        values = pd.Series(
            {
                item.security.ticker: float(item.shares or 0)
                * float(prices.get(item.security.ticker, np.nan))
                for item in holdings
            }
        ).dropna()
        if values.sum() <= 0:
            raise ValueError(
                "Share holdings have no positive market value at the requested as-of date"
            )
        total_value = float(values.sum())
        weights = values / total_value
    PortfolioSnapshot.objects.update_or_create(
        portfolio=portfolio,
        as_of=as_of,
        defaults={
            "holdings": _json(weights.to_dict()),
            "total_value": total_value,
            "source": "shares_as_of" if total_value is not None else "weights",
        },
    )
    return weights


def sync_model_catalogs(parameters: dict, progress) -> dict:
    progress(10, "syncing factor definitions")
    requested_version = parameters.get("model_version", PROXY_MODEL_VERSION)
    if requested_version != PROXY_MODEL_VERSION:
        raise ValueError(f"Unsupported factor model version: {requested_version}")
    catalogs = sync_factor_catalogs(PROXY_MODEL_VERSION)
    progress(100, "complete")
    return {
        "catalogs": {
            slug: {
                "id": catalog.pk,
                "available": catalog.available_factor_count,
                "expected": catalog.expected_factor_count,
                "completeness": catalog.completeness,
            }
            for slug, catalog in catalogs.items()
        }
    }


def run_proxy_factor_build(parameters: dict, progress) -> dict:
    """Build the fully local V2-inspired proxy and thematic factor dataset."""
    requested_version = parameters.get("model_version", PROXY_MODEL_VERSION)
    if requested_version != PROXY_MODEL_VERSION:
        raise ValueError(f"Unsupported factor model version: {requested_version}")
    snapshot_id = parameters.get("price_snapshot_id")
    if not snapshot_id:
        raise ValueError("price_snapshot_id is required for the proxy factor build")
    snapshot = PriceSnapshot.objects.get(pk=snapshot_id)
    dataset = load_price_dataset()

    progress(5, "syncing proxy model catalogs")
    catalogs = sync_factor_catalogs(PROXY_MODEL_VERSION)
    progress(10, "building local proxy hierarchy")
    panel = build_proxy_factor_model(dataset.prices)
    panel = panel.rename(columns=REVERSE_EXTERNAL_NAMES)
    panel = panel.loc[:, ~panel.columns.duplicated()].sort_index()
    if parameters.get("start"):
        panel = panel.loc[pd.Timestamp(parameters["start"]) :]
    if parameters.get("end"):
        panel = panel.loc[: pd.Timestamp(parameters["end"])]
    panel = panel.dropna(how="all")
    if panel.empty:
        raise ValueError("Proxy factor returns do not overlap the requested date range")

    stable_configuration = {
        "source": "local_adjusted_prices",
        # Retained in the fingerprint for compatibility with the existing V2 build.
        "published_factor_returns": False,
        "weekly_window": 156,
        "volatility_window": 60,
        "annual_target": 0.10,
        "style_recipes": PROXY_STYLE_SPECS,
        "theme_baskets": THEME_BASKETS,
    }
    fingerprint = canonical_model_fingerprint(
        PROXY_MODEL_VERSION,
        mode="replication",
        configuration=stable_configuration,
    )
    build = (
        FactorBuild.objects.filter(
            definition_version=PROXY_MODEL_VERSION,
            model_fingerprint=fingerprint,
            is_canonical=True,
        )
        .order_by("-id")
        .first()
    )
    created_dataset = build is None
    if created_dataset:
        build = FactorBuild.objects.create(
            definition_version=PROXY_MODEL_VERSION,
            model_fingerprint=fingerprint,
            is_canonical=True,
            price_snapshot=snapshot,
            mode="replication",
            status="running",
            update_state="updating",
            as_of=panel.index.max().date(),
            started_at=timezone.now(),
            configuration=stable_configuration,
        )
    else:
        build.price_snapshot = snapshot
        build.status = "running"
        build.update_state = "updating"
        build.started_at = timezone.now()
        build.configuration = stable_configuration
        build.save()

    definitions = {
        item.name: item
        for item in FactorDefinition.objects.filter(
            model_version=PROXY_MODEL_VERSION,
            name__in=panel.columns,
        )
    }
    missing_definitions = sorted(set(panel) - set(definitions))
    if missing_definitions:
        raise ValueError(
            f"Missing proxy factor definitions: {', '.join(missing_definitions)}"
        )

    existing_values = {
        (row["factor_id"], row["date"]): row["scaled_return"]
        for row in FactorObservation.objects.filter(build=build).values(
            "factor_id", "date", "scaled_return"
        )
    }
    coverage = {}
    observations = []
    changed_dates = []
    horizons = FACTOR_HORIZONS
    total = len(panel.columns)
    for number, name in enumerate(panel.columns):
        values = panel[name].dropna()
        coverage[name] = {
            "observations": len(values),
            "panel_observations": len(panel),
            "ratio": float(len(values) / len(panel)),
            "start": values.index.min().date().isoformat(),
            "end": values.index.max().date().isoformat(),
        }
        cumulative = (1 + values).cumprod()
        horizon_return_series = {
            horizon: horizon_returns(values, horizon) for horizon in horizons
        }
        horizon_zscores = {
            horizon: rolling_zscores(values.to_frame(name), horizon)[name]
            for horizon in horizons
        }
        provenance = {
            "source": "local_adjusted_prices",
            "methodology": "V2 local ETF proxy",
            "price_snapshot_id": snapshot.pk,
        }
        quality_flags = (
            ["weak_proxy_liquidity"]
            if name == "Liquidity"
            else (
                ["collinear_style_pair"]
                if name in {"LowVolatility", "BetaFactor"}
                else ["fixed_thematic_membership"] if name in THEME_BASKETS else []
            )
        )
        for stamp, value in values.items():
            stamp_date = stamp.date()
            previous = existing_values.get((definitions[name].pk, stamp_date))
            if previous is None or not np.isclose(
                float(previous), float(value), equal_nan=True
            ):
                changed_dates.append(stamp_date)
            observations.append(
                FactorObservation(
                    build=build,
                    factor=definitions[name],
                    date=stamp_date,
                    scaled_return=float(value),
                    cumulative_index=float(cumulative.at[stamp]),
                    coverage=coverage[name]["ratio"],
                    quality_flags=quality_flags,
                    provenance=provenance,
                    horizons=_json(
                        {
                            str(horizon): horizon_return_series[horizon].get(stamp)
                            for horizon in horizons
                        }
                    ),
                    zscores=_json(
                        {
                            str(horizon): horizon_zscores[horizon].get(stamp)
                            for horizon in horizons
                        }
                    ),
                )
            )
        progress(15 + int(65 * (number + 1) / total), f"preparing {name}")
    FactorObservation.objects.bulk_create(
        observations,
        batch_size=2000,
        update_conflicts=True,
        unique_fields=["build", "factor", "date"],
        update_fields=[
            "scaled_return",
            "cumulative_index",
            "coverage",
            "quality_flags",
            "provenance",
            "horizons",
            "zscores",
        ],
    )

    equation_date = panel.index.max().date()
    build.equations.filter(date=equation_date, factor__name__in=THEME_BASKETS).delete()
    FactorEquation.objects.bulk_create(
        [
            FactorEquation(
                build=build,
                factor=definitions[name],
                date=equation_date,
                equation={
                    "kind": "fixed_equal_weight_basket",
                    "stripped_against": "local_core_sector_industry_country",
                    "target_volatility": 0.10,
                    "rolling_window_weeks": 156,
                },
                basket_weights={ticker: 1 / len(tickers) for ticker in tickers},
                provenance={
                    "source": "QuantLab curated fixed basket",
                    "membership_frozen": True,
                },
            )
            for name, tickers in THEME_BASKETS.items()
        ]
    )

    previous_as_of = build.as_of
    latest_date = panel.index.max().date()
    if created_dataset:
        update_mode = "full"
    elif not changed_dates:
        update_mode = "unchanged"
    elif min(changed_dates) > previous_as_of:
        update_mode = "append"
    else:
        update_mode = "correction"
    invalidated_fits = 0
    if update_mode == "append":
        stale = build.stock_fits.filter(period=latest_date.replace(day=1)).exclude(
            as_of=latest_date
        )
        invalidated_fits = stale.count()
        stale.delete()
    elif update_mode == "correction":
        stale = build.stock_fits.filter(as_of__gte=min(changed_dates))
        invalidated_fits = stale.count()
        stale.delete()

    checksum = hashlib.sha256(
        pd.util.hash_pandas_object(panel, index=True).values.tobytes()
    ).hexdigest()
    build.status = "succeeded"
    build.update_state = "idle"
    build.finished_at = timezone.now()
    build.last_factor_update_at = timezone.now()
    build.as_of = latest_date
    build.artifact_path = ""
    build.checksum = checksum
    build.input_checksum = checksum
    build.provenance = {
        "methodology": "V2 self-contained proxy model",
        "price_source": dataset.provenance.get("provider"),
        "theme_membership": "fixed",
    }
    build.coverage = coverage
    build.diagnostics = {
        "catalogs": {
            slug: {
                "available": catalog.available_factor_count,
                "expected": catalog.expected_factor_count,
                "completeness": catalog.completeness,
            }
            for slug, catalog in catalogs.items()
        },
        "warnings": [
            "LowVolatility and BetaFactor use mirror ETF legs; ElasticNet is required.",
            "Liquidity is represented by the OEF-minus-IWC ETF spread.",
            "Themes use fixed, versioned membership.",
        ],
        "latest_date": build.as_of.isoformat(),
        "last_update": {
            "mode": update_mode,
            "changed_observations": len(changed_dates),
            "earliest_changed_date": (
                min(changed_dates).isoformat() if changed_dates else None
            ),
            "latest_date": latest_date.isoformat(),
            "invalidated_stock_fits": invalidated_fits,
        },
    }
    build.save()
    Artifact.objects.filter(
        kind="factor_panel", owner_type="FactorBuild", owner_id=build.pk
    ).delete()
    Artifact.objects.create(
        kind="factor_panel",
        owner_type="FactorBuild",
        owner_id=build.pk,
        path="",
        format="sqlite",
        checksum=checksum,
        metadata={
            "columns": len(panel.columns),
            "rows": len(panel),
            "source": "local_v2_proxy",
        },
    )
    progress(100, "complete")
    return {
        "factor_build_id": build.pk,
        "observations": build.observations.count(),
        "changed_observations": len(changed_dates),
        "update_mode": update_mode,
        "factor_count": len(panel.columns),
        "latest_date": build.as_of.isoformat(),
    }


def _exposure_universe(
    metadata: pd.DataFrame,
    returns: pd.DataFrame,
    parameters: dict,
    prices: pd.DataFrame | None = None,
) -> list[str]:
    """Choose which stocks to fit. Subsets skip the automatic SPY append."""
    stocks = metadata.index[metadata["asset_type"].eq("stock")].intersection(
        returns.columns
    )
    requested = parameters.get("tickers")
    if isinstance(requested, str):
        requested = [
            item.strip().upper() for item in requested.split(",") if item.strip()
        ]
    if requested:
        requested = [
            str(item).strip().upper() for item in requested if str(item).strip()
        ]
        missing = [ticker for ticker in requested if ticker not in stocks]
        if missing:
            raise ValueError(f"Unknown stock tickers: {', '.join(missing)}")
        return list(dict.fromkeys(requested))

    names = stocks.tolist()
    top_n = parameters.get("top_n_by_market_cap")
    if top_n:
        caps = _current_market_caps(metadata.loc[names], prices)
        ranked = caps.sort_values(ascending=False, na_position="last").dropna()
        if "CIK" in metadata.columns:
            companies = pd.DataFrame(
                {"cap": ranked, "cik": metadata.reindex(ranked.index)["CIK"]}
            )
            ranked = companies.drop_duplicates("cik", keep="first")["cap"]
        names = ranked.head(int(top_n)).index.tolist()
        if not names:
            raise ValueError("No securities have market cap values to rank")
        return names

    if prices is not None and "SPY" in prices.columns and "SPY" not in names:
        names.append("SPY")
    elif prices is None and "SPY" in returns.columns and "SPY" not in names:
        names.append("SPY")
    return names


def _current_market_caps(
    metadata: pd.DataFrame, prices: pd.DataFrame | None
) -> pd.Series:
    """Prefer Yahoo market cap; fall back to current shares times last price."""
    caps = pd.Series(index=metadata.index, dtype=float)
    if "market_cap" in metadata.columns:
        caps = pd.to_numeric(metadata["market_cap"], errors="coerce")
    if caps.notna().sum() > 0:
        return caps
    if prices is None or "current_shares" not in metadata.columns:
        raise ValueError("No market cap metadata available to rank the universe")
    shares = pd.to_numeric(metadata["current_shares"], errors="coerce")
    last = prices.reindex(columns=metadata.index).ffill().iloc[-1]
    return shares * last


def _exposure_worker_count(requested=None) -> int:
    """Outer-loop thread count. SQLite writes stay on the main thread."""
    cpu = os.cpu_count() or 1
    hardware = max(1, cpu - 2)
    configured = getattr(settings, "EXPOSURE_MAX_WORKERS", None)
    default = min(hardware, int(configured) if configured else 24)
    if requested in (None, "", 0):
        return max(1, default)
    return max(1, min(int(requested), hardware))


def _level_factor_columns(catalogs, factors, levels) -> dict[str, list[str]]:
    columns_by_level = {}
    for level in levels:
        catalog = catalogs.get(level)
        if catalog:
            columns_by_level[level] = [
                membership.factor.name
                for membership in catalog.memberships.all()
                if membership.factor.name in factors.columns
            ]
        else:
            columns = list(PROXY_CORE_FACTORS)
            if level != "base":
                columns.extend(SECTOR_FACTORS)
            columns_by_level[level] = [
                name for name in columns if name in factors.columns
            ]
    return columns_by_level


def _exposure_month_ends(index, parameters: dict, build=None) -> pd.Series:
    month_ends = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    update_mode = parameters.get("update_mode", "backfill")
    if update_mode not in {"auto", "latest", "backfill"}:
        raise ValueError("Exposure update_mode must be auto, latest, or backfill")
    if update_mode in {"auto", "latest"}:
        selected = month_ends.tail(1)
        if update_mode == "auto" and build:
            last_update = build.diagnostics.get("last_update", {})
            if last_update.get("mode") == "correction" and last_update.get(
                "earliest_changed_date"
            ):
                invalid_from = pd.Timestamp(last_update["earliest_changed_date"])
                selected = month_ends[month_ends >= invalid_from]
        return selected
    if parameters.get("max_months"):
        return month_ends.tail(int(parameters["max_months"]))
    return month_ends


def build_monthly_exposures(parameters: dict, progress) -> dict:
    build = FactorBuild.objects.get(pk=parameters["factor_build_id"])
    if not build.price_snapshot_id and not build.configuration.get("data_snapshot_id"):
        raise ValueError(
            "Stock exposure models require the factor build to reference a price snapshot"
        )
    dataset = _dataset_for_build(build)
    factors = pd.DataFrame.from_records(
        build.observations.values("date", "factor__name", "scaled_return")
    ).pivot(index="date", columns="factor__name", values="scaled_return")
    factors.index = pd.to_datetime(factors.index)
    returns = dataset.prices.pct_change(fill_method=None)
    metadata = dataset.metadata.set_index("ticker")
    stock_names = _exposure_universe(metadata, returns, parameters, dataset.prices)
    update_mode = parameters.get("update_mode", "backfill")
    month_ends = _exposure_month_ends(returns.index, parameters, build)
    window = int(parameters.get("trailing_days", 756))
    minimum = int(parameters.get("minimum_observations", 252))
    levels = parameters.get("levels", ["base", "base_sector"])
    valid_levels = {"base", "base_sector", "base_sector_industry", "all_factors"}
    unknown_levels = set(levels) - valid_levels
    if unknown_levels:
        raise ValueError(f"Unknown model catalogs: {', '.join(sorted(unknown_levels))}")
    catalogs = {
        item.slug: item
        for item in FactorModelCatalog.objects.filter(
            model_version=build.definition_version,
            slug__in=levels,
        ).prefetch_related("memberships__factor")
    }
    selection_mode = parameters.get("selection_mode", "elastic_net")
    workers = _exposure_worker_count(parameters.get("workers"))
    columns_by_level = _level_factor_columns(catalogs, factors, levels)
    definitions = {
        item.name: item
        for item in FactorDefinition.objects.filter(
            model_version=build.definition_version
        )
    }
    securities = {
        item.ticker: item for item in Security.objects.filter(ticker__in=stock_names)
    }
    for ticker in stock_names:
        if ticker not in securities:
            securities[ticker], _ = Security.objects.get_or_create(
                ticker=ticker,
                defaults={"asset_type": "factor_proxy", "name": ticker},
            )
    created = 0
    latest_price_date = returns.index.max().date()
    latest_period = latest_price_date.replace(day=1)

    def _persist(as_of, results):
        persisted = 0
        with transaction.atomic():
            for result in results:
                if result is None:
                    continue
                as_of_date = result["as_of"].date()
                period = as_of.date().replace(day=1)
                estimate = result["estimate"]
                hac = estimate.inference["hac"]
                level = result["level"]
                fit, _ = StockModelFit.objects.update_or_create(
                    build=build,
                    security=securities[result["ticker"]],
                    period=period,
                    model_level=level,
                    defaults={
                        "as_of": as_of_date,
                        "is_provisional": period == latest_period,
                        "source_price_checksum": (
                            build.price_snapshot.checksum
                            if build.price_snapshot_id
                            else ""
                        ),
                        "source_factor_checksum": build.checksum,
                        "model_catalog": catalogs.get(level),
                        "alpha": float(hac.loc["const", "beta"]),
                        "adjusted_r2": estimate.adjusted_r2,
                        "residual_volatility": estimate.residual_vol,
                        "active_factor_count": len(estimate.selected),
                        "design_factor_count": result["n_columns"],
                        "observation_count": estimate.observation_count,
                        "coverage": result["coverage"],
                        "inference_method": estimate.inference_method,
                    },
                )
                fit.exposures.all().delete()
                rows = []
                for factor in estimate.selected:
                    definition = definitions.get(factor)
                    if definition is None:
                        continue
                    values = hac.loc[factor]
                    rows.append(
                        StockExposureSnapshot(
                            model_fit=fit,
                            factor=definition,
                            beta=values["beta"],
                            standard_error=values["se"],
                            t_stat=values["t_stat"],
                            p_value=values["p_value"],
                            confidence_low=values["ci_low"],
                            confidence_high=values["ci_high"],
                        )
                    )
                StockExposureSnapshot.objects.bulk_create(rows)
                persisted += 1
        return persisted

    StockModelFit.objects.filter(
        build=build,
        period__lt=latest_period,
        is_provisional=True,
    ).update(is_provisional=False)

    # A process pool rather than threads: each fit is an ElasticNetCV plus an
    # OLS/HAC re-fit, which is dominated by GIL-bound Python rather than by the
    # large BLAS calls that would let threads scale.
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for month_number, as_of in enumerate(month_ends):
            tasks = []
            for level, columns in columns_by_level.items():
                if not columns:
                    continue
                design = factors.reindex(columns=columns).loc[:as_of]
                usable_dates = design.dropna(how="any").index
                if usable_dates.empty:
                    continue
                effective_as_of = usable_dates.max()
                # Shared across tickers, so pickling it once per chunk is enough.
                level_design = design.loc[:effective_as_of].tail(window)
                for ticker in stock_names:
                    stock_slice = returns[ticker].loc[:effective_as_of].tail(window)
                    tasks.append(
                        {
                            "ticker": ticker,
                            "level": level,
                            "as_of": effective_as_of,
                            "stock": stock_slice,
                            "design": level_design,
                            "coverage": float(stock_slice.notna().mean()),
                            "minimum": minimum,
                            "selection_mode": selection_mode,
                            "n_columns": len(columns),
                        }
                    )
            if not tasks:
                progress(
                    5 + int(90 * (month_number + 1) / len(month_ends)),
                    f"month {as_of:%Y-%m}",
                )
                continue
            if pool is None:
                estimates = [run_exposure_estimate(task) for task in tasks]
            else:
                chunksize = max(1, min(64, len(tasks) // (workers * 4)))
                estimates = list(
                    pool.map(run_exposure_estimate, tasks, chunksize=chunksize)
                )
            created += _persist(as_of, estimates)
            progress(
                5 + int(90 * (month_number + 1) / len(month_ends)),
                f"month {as_of:%Y-%m}",
            )
    finally:
        if pool is not None:
            pool.shutdown()
    portfolio_ids = parameters.get("portfolio_ids", [])
    if isinstance(portfolio_ids, str):
        portfolio_ids = [int(item) for item in portfolio_ids.split(",") if item.strip()]
    for portfolio_id in portfolio_ids:
        for as_of in month_ends:
            try:
                monitor_portfolio(
                    {
                        "portfolio_id": portfolio_id,
                        "factor_build_id": build.pk,
                        "as_of": as_of.date().isoformat(),
                    },
                    lambda *_: None,
                )
            except ValueError:
                continue
    coverage = dict(build.coverage)
    stock_coverage = dict(coverage.get("stock_fits", {}))
    for level in levels:
        level_fits = StockModelFit.objects.filter(build=build, model_level=level)
        latest_fit = (
            level_fits.order_by("-as_of").values_list("as_of", flat=True).first()
        )
        latest_count = (
            level_fits.filter(as_of=latest_fit).values("security_id").distinct().count()
            if latest_fit
            else 0
        )
        months_by_stock = list(
            level_fits.values("security_id")
            .annotate(months=models.Count("period", distinct=True))
            .values_list("months", flat=True)
        )
        months_by_stock.sort()
        stock_coverage[level] = {
            "latest_as_of": latest_fit.isoformat() if latest_fit else None,
            "latest_stocks": latest_count,
            "stocks_with_history": sum(value > 1 for value in months_by_stock),
            "median_months": (
                months_by_stock[len(months_by_stock) // 2] if months_by_stock else 0
            ),
            "maximum_months": max(months_by_stock, default=0),
        }
    coverage["stock_fits"] = stock_coverage
    build.coverage = coverage
    build.save(update_fields=["coverage"])
    return {
        "model_fits": created,
        "tickers": stock_names,
        "months": len(month_ends),
        "workers": workers,
        "update_mode": update_mode,
    }


def run_canonical_update(parameters: dict, progress) -> dict:
    """Update factor returns, then the current stock decomposition period."""
    factor_result = run_proxy_factor_build(
        parameters,
        lambda value, stage="": progress(
            min(35, int(value * 0.35)), f"factors · {stage}"
        ),
    )
    exposure_parameters = {
        "factor_build_id": factor_result["factor_build_id"],
        "levels": parameters.get(
            "levels",
            [
                "base",
                "base_sector",
                "base_sector_industry",
                "all_factors",
            ],
        ),
        "trailing_days": int(parameters.get("trailing_days", 756)),
        "minimum_observations": int(parameters.get("minimum_observations", 252)),
        "update_mode": "auto",
        **({"workers": parameters["workers"]} if parameters.get("workers") else {}),
    }
    exposure_result = build_monthly_exposures(
        exposure_parameters,
        lambda value, stage="": progress(
            35 + int(value * 0.65), f"decomposition · {stage}"
        ),
    )
    progress(100, "canonical dataset current")
    return {
        **factor_result,
        "decomposition": exposure_result,
    }


def _screen_features(definition: ScreenDefinition, as_of: date):
    requested = set(definition.factor_weights) | set(definition.regime_interactions)
    model_version = (
        definition.factor_build.definition_version
        if definition.factor_build_id
        else PROXY_MODEL_VERSION
    )
    available_definitions = set(
        FactorDefinition.objects.filter(
            model_version=model_version,
            name__in=requested,
        ).values_list("name", flat=True)
    )
    unavailable = requested - available_definitions
    if unavailable:
        raise ValueError(
            f"Screen factors unavailable in this model: {', '.join(sorted(unavailable))}"
        )
    fits = StockModelFit.objects.filter(
        build=definition.factor_build,
        as_of__lte=as_of,
        model_level=definition.model_level,
        security__asset_type="stock",
    )
    latest = {}
    for fit in (
        fits.select_related("security")
        .prefetch_related("exposures__factor")
        .order_by("-as_of")
    ):
        latest.setdefault(fit.security_id, fit)
    records, sectors = {}, {}
    for fit in latest.values():
        exposure_items = list(fit.exposures.all())
        exposure_map = {item.factor.name: item for item in exposure_items}
        market_exposure = exposure_map.get("Market")
        records[fit.security.ticker] = {
            **{
                name: exposure_map[name].beta if name in exposure_map else 0.0
                for name in requested
            },
            **{f"{name}__active": int(name in exposure_map) for name in requested},
            "market_cap": fit.security.market_cap,
            "beta_significance": (
                abs(market_exposure.t_stat) if market_exposure else np.nan
            ),
            "beta_p_value": market_exposure.p_value if market_exposure else np.nan,
            "adjusted_r2": fit.adjusted_r2,
            "residual_volatility": fit.residual_volatility,
            "coverage": fit.coverage,
        }
        sectors[fit.security.ticker] = fit.security.sector
    if not records:
        raise ValueError("No point-in-time stock models are available for this screen")
    return pd.DataFrame.from_dict(records, orient="index"), pd.Series(sectors), latest


def _screen_preview(definition, features, sectors, regime=None):
    """Apply the Stock Selection UI semantics to a point-in-time feature frame."""
    filters = dict(definition.filters or {})
    engine_filters = dict(filters)
    for source, target in (
        ("minimum_coverage", "coverage"),
        ("minimum_adjusted_r2", "adjusted_r2"),
    ):
        if filters.get(source) is not None:
            engine_filters[target] = {"min": float(filters[source])}
    preview = preview_screen(
        features,
        definition.factor_weights,
        engine_filters,
        sectors,
        regime or definition.regime_interactions,
        definition.directions,
    )
    excluded = set(parse_ticker_list(filters.get("excluded_tickers", [])))
    eligible = preview["passed"] & ~preview.index.to_series().isin(excluded)
    selected = list(
        preview.loc[eligible]
        .sort_values("composite", ascending=False)
        .head(definition.top_n)
        .index
    )
    manual = parse_ticker_list(filters.get("manual_tickers", []))
    selected.extend(ticker for ticker in manual if ticker not in selected)
    return preview, selected, set(manual)


def run_screen(parameters: dict, progress) -> dict:
    definition = ScreenDefinition.objects.get(pk=parameters["screen_id"])
    idempotency_key = parameters.get("_idempotency_key")
    if idempotency_key:
        prior = ScreenRun.objects.filter(
            definition=definition, diagnostics__idempotency_key=idempotency_key
        ).first()
        if prior and prior.status == "succeeded":
            return {
                "screen_run_id": prior.pk,
                "eligible": prior.results.filter(passed=True).count(),
            }
        if prior:
            prior.delete()
    as_of = (
        definition.as_of or FactorBuild.objects.get(pk=definition.factor_build_id).as_of
    )
    features, sectors, latest = _screen_features(definition, as_of)
    preview, selected_tickers, manual_tickers = _screen_preview(
        definition, features, sectors
    )
    run = ScreenRun.objects.create(
        definition=definition,
        as_of=as_of,
        status="succeeded",
        diagnostics={
            "selected": selected_tickers,
            "top_n": definition.top_n,
            "directions": definition.directions,
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        },
    )
    rows = []
    for ticker, item in preview.iterrows():
        rows.append(
            ScreenResult(
                run=run,
                security=Security.objects.get(ticker=ticker),
                model_fit=latest[Security.objects.get(ticker=ticker).pk],
                passed=ticker in selected_tickers,
                rank=int(item["rank"]) if pd.notna(item["rank"]) else None,
                percentile=item["percentile"],
                score=item["composite"],
                components=_json(item["components"]),
                rule_results=_json(
                    {
                        **item["rules"],
                        "selection_source": (
                            "manual" if ticker in manual_tickers else "model"
                        ),
                    }
                ),
            )
        )
    existing_tickers = set(preview.index)
    for ticker in manual_tickers - existing_tickers:
        rows.append(
            ScreenResult(
                run=run,
                security=Security.objects.get(ticker=ticker),
                model_fit=None,
                passed=True,
                rank=selected_tickers.index(ticker) + 1,
                percentile=None,
                score=None,
                components={},
                rule_results={
                    "eligible": False,
                    "selection_source": "manual",
                    "confidence": 0,
                },
            )
        )
    ScreenResult.objects.bulk_create(rows)
    score_rows, forward_rows, selection_rows = {}, {}, {}
    history_fits = (
        StockModelFit.objects.filter(
            build=definition.factor_build, model_level=definition.model_level
        )
        .select_related("security")
        .prefetch_related("exposures__factor")
    )
    by_month = {}
    for fit in history_fits:
        by_month.setdefault(fit.as_of, []).append(fit)
    dataset = _dataset_for_build(definition.factor_build)
    asset_returns = dataset.prices.pct_change(fill_method=None)
    for month, month_fits in sorted(by_month.items()):
        features = pd.DataFrame.from_dict(
            {
                fit.security.ticker: {
                    item.factor.name: item.beta for item in fit.exposures.all()
                }
                for fit in month_fits
            },
            orient="index",
        )
        try:
            scores = composite_scores(
                features, definition.factor_weights, definition.directions
            )["composite"]
        except ValueError:
            continue
        next_period = pd.Period(month, freq="M") + 1
        next_rows = asset_returns[asset_returns.index.to_period("M") == next_period]
        if next_rows.empty:
            continue
        score_rows[pd.Timestamp(month)] = scores
        forward_rows[pd.Timestamp(month)] = (1 + next_rows).prod() - 1
        chosen = scores.nlargest(min(definition.top_n, len(scores))).index
        selection_rows[pd.Timestamp(month)] = pd.Series(True, index=chosen)
    if score_rows:
        score_panel = pd.DataFrame(score_rows).T
        forward_panel = pd.DataFrame(forward_rows).T
        selections = pd.DataFrame(selection_rows).T.fillna(False)
        statistics = screen_backtest_statistics(
            score_panel,
            forward_panel,
            selections,
            transaction_cost_bps=float(
                definition.filters.get("transaction_cost_bps", 25)
            ),
        )
        statistics["annualized_icir"] = (
            statistics["icir"] * np.sqrt(12) if pd.notna(statistics["icir"]) else np.nan
        )
        if statistics["observations"] < 36:
            statistics["warning"] = "Fewer than 36 monthly out-of-sample observations"
        run.diagnostics = {**run.diagnostics, "backtest_statistics": _json(statistics)}
        run.save(update_fields=["diagnostics"])
    progress(100, "complete")
    return {"screen_run_id": run.pk, "eligible": int(preview["passed"].sum())}


def _optimization_inputs(parameters: dict) -> dict:
    portfolio = Portfolio.objects.get(pk=parameters["portfolio_id"])
    build = FactorBuild.objects.get(pk=parameters["factor_build_id"])
    dataset = _dataset_for_build(build)
    holdings = list(portfolio.holdings.select_related("security"))
    if not holdings:
        raise ValueError("The portfolio has no holdings")
    names = [item.security.ticker for item in holdings]
    as_of = (
        date.fromisoformat(parameters["as_of"])
        if parameters.get("as_of")
        else build.as_of
    )
    model_level = parameters.get("model_level") or portfolio.configuration.get(
        "stock_selection", {}
    ).get("model_level", "base_sector")
    original = _portfolio_weights(portfolio, dataset, as_of).reindex(names).fillna(0)
    risk = _risk_inputs_for_tickers(
        build=build,
        names=names,
        as_of=as_of,
        parameters=parameters,
        dataset=dataset,
        model_level=model_level,
    )
    return {
        "portfolio": portfolio,
        "holdings": holdings,
        "original": original,
        **risk,
    }


def _risk_inputs_for_tickers(
    *,
    build,
    names: list[str],
    as_of,
    parameters: dict,
    dataset,
    model_level: str,
) -> dict:
    lookback = min(max(int(parameters.get("lookback", 756)), 126), 2520)
    latest = {}
    fits = (
        StockModelFit.objects.filter(
            build=build,
            security__ticker__in=names,
            model_level=model_level,
            as_of__lte=as_of,
        )
        .prefetch_related("exposures__factor")
        .order_by("security_id", "-as_of")
    )
    for fit in fits:
        latest.setdefault(fit.security.ticker, fit)
    missing = sorted(set(names) - set(latest))
    if missing:
        raise ValueError(
            f"No {model_level} stock model on or before {as_of} for: "
            + ", ".join(missing)
        )
    exposure_records = pd.DataFrame(
        {
            ticker: {item.factor.name: item.beta for item in fit.exposures.all()}
            for ticker, fit in latest.items()
        }
    ).T.reindex(names)
    catalog = (
        FactorModelCatalog.objects.filter(
            model_version=build.definition_version,
            slug=model_level,
        )
        .prefetch_related("memberships__factor")
        .first()
    )
    design_factors = (
        [membership.factor.name for membership in catalog.memberships.all()]
        if catalog
        else list(exposure_records.columns)
    )
    exposures = exposure_records.reindex(columns=design_factors).fillna(0)
    standard_errors = (
        pd.DataFrame(
            {
                ticker: {
                    item.factor.name: item.standard_error
                    for item in fit.exposures.all()
                }
                for ticker, fit in latest.items()
            }
        )
        .T.reindex(index=names, columns=design_factors)
        .fillna(1_000_000)
    )
    factor_records = FactorObservation.objects.filter(
        build=build,
        date__lte=as_of,
        factor__name__in=exposures.columns,
    ).values("date", "factor__name", "scaled_return")
    factor_returns = pd.DataFrame.from_records(factor_records)
    if factor_returns.empty:
        raise ValueError(
            "No factor return history is available through the requested date"
        )
    factor_returns = (
        factor_returns.pivot(
            index="date", columns="factor__name", values="scaled_return"
        )
        .sort_index()
        .tail(lookback)
    )
    omega = factor_covariance(factor_returns)
    specific = pd.Series(
        {name: latest[name].residual_volatility for name in names}, dtype=float
    )
    stock_returns = (
        dataset.prices.loc[: pd.Timestamp(as_of)]
        .reindex(columns=names)
        .pct_change(fill_method=None)
        .tail(lookback)
    )
    if stock_returns.dropna(how="all").empty:
        raise ValueError(
            "No stock return history is available through the requested date"
        )
    covariance = factor_risk_covariance(
        exposures,
        omega,
        specific,
        residual_correlation=stock_returns.tail(252).corr(),
        shrinkage=float(parameters.get("residual_shrinkage", 0.8)),
    )
    risk_model = "factor_model"
    spy_fit = (
        StockModelFit.objects.filter(
            build=build,
            security__ticker="SPY",
            model_level=model_level,
            as_of__lte=as_of,
        )
        .order_by("-as_of")
        .first()
    )
    spy_exposure = (
        pd.Series(
            {
                item.factor.name: item.beta
                for item in spy_fit.exposures.select_related("factor")
            },
            dtype=float,
        )
        if spy_fit
        else None
    )
    return {
        "build": build,
        "dataset": dataset,
        "names": names,
        "as_of": as_of,
        "model_level": model_level,
        "latest": latest,
        "exposures": exposures,
        "standard_errors": standard_errors,
        "factor_returns": factor_returns,
        "factor_covariance": omega,
        "specific": specific,
        "stock_returns": stock_returns,
        "covariance": covariance,
        "risk_model": risk_model,
        "spy_exposure": spy_exposure,
        "risk_free_rate": risk_free_rate_on(
            as_of, default=float(parameters.get("risk_free_rate", 0))
        ),
        "risk_free_rate_observed": RiskFreeRate.objects.filter(
            date__lte=as_of
        ).exists(),
    }


def _portfolio_ex_ante_metrics(
    weights: pd.Series,
    covariance: pd.DataFrame,
    expected: pd.Series | None = None,
) -> dict:
    aligned = weights.reindex(covariance.index).fillna(0)
    variance = float(
        aligned
        @ covariance.reindex(index=aligned.index, columns=aligned.index).fillna(0)
        @ aligned
    )
    original_return = None
    if expected is not None:
        original_return = float(aligned @ expected.reindex(aligned.index).fillna(0))
    return {
        "expected_return": original_return,
        "expected_volatility": float(np.sqrt(max(variance, 0))),
        "turnover": 0.0,
    }


def _optimization_kwargs(inputs: dict, parameters: dict) -> dict:
    return {
        "previous_weights": inputs["original"],
        "weight_constraint_mode": parameters.get("weight_constraint_mode", "absolute"),
        "min_weight": float(parameters.get("min_weight", 0)),
        "max_weight": float(parameters.get("max_weight", 0.15)),
        "max_weight_cap": parameters.get("max_weight_cap"),
        "risk_aversion": float(parameters.get("risk_aversion", 1)),
        "risk_free_rate": float(
            inputs.get("risk_free_rate", parameters.get("risk_free_rate", 0))
        ),
        "turnover_cap": parameters.get("turnover_cap"),
        "turnover_penalty": float(parameters.get("turnover_penalty", 0)),
        "transaction_cost": float(parameters.get("transaction_cost", 0)),
        "sectors": pd.Series(
            {item.security.ticker: item.security.sector for item in inputs["holdings"]}
        ),
        "sector_bounds": {
            name: tuple(bounds)
            for name, bounds in parameters.get("sector_bounds", {}).items()
        },
        "factor_exposures": inputs["exposures"],
        "factor_bounds": {
            name: tuple(bounds)
            for name, bounds in parameters.get("factor_bounds", {}).items()
        },
        "relative_factor_bounds": {
            name: tuple(bounds)
            for name, bounds in parameters.get("relative_factor_bounds", {}).items()
        },
        "benchmark_factor_exposures": inputs["spy_exposure"],
        "factor_covariance": inputs["factor_covariance"],
        "max_factor_variance": parameters.get("max_factor_variance"),
        "max_factor_components": parameters.get("max_factor_components"),
        "risk_budgets": (
            pd.Series(parameters["risk_budgets"], dtype=float)
            if parameters.get("risk_budgets")
            else None
        ),
        "tracking_error_limit": parameters.get("tracking_error_limit"),
        "benchmark_weights": inputs["original"],
        "specific_variances": inputs["specific"].pow(2),
        "covariance_method": inputs["risk_model"],
    }


def _persist_optimization_variant(
    *,
    study: OptimizationStudy,
    inputs: dict,
    parameters: dict,
    result,
    expected: pd.Series | None,
    name: str,
    variant: str,
    probability: float | None = None,
    covariance: pd.DataFrame | None = None,
    factor_covariance_matrix: pd.DataFrame | None = None,
    diagnostics: dict | None = None,
) -> OptimizationScenario:
    covariance = covariance if covariance is not None else inputs["covariance"]
    factor_covariance_matrix = (
        factor_covariance_matrix
        if factor_covariance_matrix is not None
        else inputs["factor_covariance"]
    )
    weights = result.weights
    marginal_variance = covariance @ weights
    volatility = max(float(np.sqrt(weights @ covariance @ weights)), 1e-16)
    factor_beta = inputs["exposures"].T @ weights
    factor_variance = float(
        factor_beta
        @ factor_covariance_matrix.reindex(
            index=factor_beta.index, columns=factor_beta.index
        ).fillna(0)
        @ factor_beta
    )
    specific_variance = float(
        np.sum((weights * inputs["specific"].reindex(weights.index).fillna(0)) ** 2)
    )
    scenario = OptimizationScenario.objects.create(
        study=study,
        portfolio=inputs["portfolio"],
        name=name,
        variant=variant,
        probability=probability,
        objective=parameters.get("objective", "min_variance"),
        expected_return_model=parameters.get("expected_return_model", ""),
        status="succeeded" if result.success else "infeasible",
        configuration={**parameters, "variant": variant},
        diagnostics=_json({**result.diagnostics, **(diagnostics or {})}),
        expected_return=result.diagnostics["expected_return"],
        expected_volatility=result.diagnostics["ex_ante_volatility"],
        turnover=result.diagnostics["turnover"],
        factor_variance=factor_variance,
        specific_variance=specific_variance,
    )
    original_metrics = _portfolio_ex_ante_metrics(
        inputs["original"], covariance, expected
    )
    equal = 1 / len(inputs["names"])
    securities = {item.security.ticker: item.security for item in inputs["holdings"]}
    OptimizationHolding.objects.bulk_create(
        [
            OptimizationHolding(
                scenario=scenario,
                security=securities[ticker],
                original_weight=inputs["original"][ticker],
                equal_weight=equal,
                optimized_weight=weights[ticker] if result.success else None,
                expected_return=(
                    float(expected.get(ticker, 0)) if expected is not None else None
                ),
                marginal_risk=float(marginal_variance[ticker] / volatility),
                component_risk=float(
                    weights[ticker] * marginal_variance[ticker] / volatility
                ),
            )
            for ticker in inputs["names"]
        ]
    )
    scenario.comparison = _json(
        {
            "original": {
                "weights": inputs["original"].to_dict(),
                "exposures": (inputs["exposures"].T @ inputs["original"]).to_dict(),
                **original_metrics,
            },
            "equal_weight": {"weights": {ticker: equal for ticker in inputs["names"]}},
            "optimized": {
                "weights": weights.to_dict() if result.success else None,
                "exposures": factor_beta.to_dict() if result.success else None,
                "status": scenario.status,
            },
            "SPY": {
                "exposures": (
                    inputs["spy_exposure"].to_dict()
                    if inputs["spy_exposure"] is not None
                    else {}
                )
            },
        }
    )
    scenario.save(update_fields=["comparison"])
    return scenario


def _factor_premium_forecast(inputs: dict, parameters: dict):
    from optimization.forecasts import (
        factor_implied_expected_returns,
        shrink_factor_premia,
    )

    prior = parameters.get("factor_premia", {})
    shrinkage = shrink_factor_premia(
        inputs["factor_returns"],
        prior,
        prior_dispersion=float(
            parameters.get("premium_prior_dispersion", PREMIUM_PRIOR_DISPERSION)
        ),
    )
    result = factor_implied_expected_returns(
        inputs["exposures"],
        shrinkage.premia,
        risk_free_rate=float(inputs["risk_free_rate"]),
        standard_errors=inputs["standard_errors"],
        shrink_betas=bool(parameters.get("shrink_factor_betas", False)),
    )
    premium_diagnostics = {
        factor: {
            "premium": float(premium),
            "assumption": float(shrinkage.prior_premia[factor]),
            "historical_mean": float(shrinkage.sample_premia[factor]),
            "historical_standard_error": float(shrinkage.standard_errors[factor]),
            "history_weight": float(shrinkage.shrinkage_weights[factor]),
            "observations": int(shrinkage.observations[factor]),
        }
        for factor, premium in result.factor_premia.items()
    }
    return result, premium_diagnostics


def _equal_sharpe_forecast(inputs: dict, parameters: dict):
    from optimization.forecasts import equal_sharpe_expected_returns

    return equal_sharpe_expected_returns(
        inputs["covariance"],
        risk_free_rate=float(inputs["risk_free_rate"]),
        common_sharpe=float(parameters.get("common_sharpe", DEFAULT_COMMON_SHARPE)),
    )


def _forecast_warnings(expected: pd.Series | None, objective: str) -> list[str]:
    if expected is None or objective == "min_variance":
        return []
    dispersion = float(expected.std(ddof=0))
    if dispersion <= 1e-10:
        return [
            (
                "Expected returns have no cross-sectional dispersion, so this "
                "objective is equivalent to minimum variance."
            )
        ]
    return []


def run_optimization(parameters: dict, progress) -> dict:
    idempotency_key = parameters.get("_idempotency_key")
    if idempotency_key:
        prior = OptimizationStudy.objects.filter(
            configuration__idempotency_key=idempotency_key
        ).first()
        if prior and prior.variants.exists():
            scenario = prior.variants.first()
            return {
                "study_id": prior.pk,
                "scenario_id": scenario.pk,
                "success": scenario.status == "succeeded",
            }
        if prior:
            prior.delete()
    inputs = _optimization_inputs(parameters)
    progress(20, "assembled point-in-time risk inputs")
    objective = parameters.get("objective", "min_variance")
    expected_model = parameters.get("expected_return_model", "")
    expected = None
    expected_diagnostics = {}
    if expected_model == "user_supplied":
        expected = pd.Series(parameters.get("expected_returns", {}), dtype=float)
    elif expected_model == "historical_shrinkage":
        from optimization.forecasts import shrink_annualized_expected_returns

        shrinkage = shrink_annualized_expected_returns(inputs["stock_returns"])
        expected = shrinkage.expected_returns
        expected_diagnostics["historical_shrinkage"] = {
            "prior_mean": shrinkage.prior_mean,
            "prior_variance": shrinkage.prior_variance,
            "shrinkage_weights": shrinkage.shrinkage_weights,
        }
    elif expected_model == "factor_premium":
        forecast, premium_diagnostics = _factor_premium_forecast(inputs, parameters)
        expected = forecast.expected_returns
        expected_diagnostics["factor_premia"] = premium_diagnostics
        expected_diagnostics["annualized_factor_contributions"] = (
            forecast.annualized_contributions
        )
    elif expected_model == "equal_sharpe":
        forecast = _equal_sharpe_forecast(inputs, parameters)
        expected = forecast.expected_returns
        expected_diagnostics["equal_sharpe"] = {
            "common_sharpe": forecast.common_sharpe,
            "volatilities": forecast.volatilities,
            "cancels_for_max_sharpe": objective == "max_sharpe",
        }
    elif expected_model:
        raise ValueError(f"Unknown expected return model: {expected_model}")
    if objective in {
        "max_return",
        "mean_variance",
        "max_sharpe",
        "factor_score_utility",
    } and (expected is None or not set(inputs["names"]).issubset(expected.index)):
        raise ValueError(
            f"{objective} requires an expected return model covering every holding"
        )
    warnings = _forecast_warnings(expected, objective)
    if expected_model == "equal_sharpe" and objective == "max_return":
        warnings.append(
            "Equal-Sharpe maximum return ranks names by predicted volatility, not by "
            "historical return."
        )
    if expected_model == "equal_sharpe" and objective == "max_sharpe":
        warnings.append(
            "Equal-Sharpe maximum Sharpe is the Maximum Diversification Portfolio: "
            "weights depend on covariance, not on historical means or the common "
            "Sharpe scalar."
        )
    if objective == "max_sharpe" and not inputs["risk_free_rate_observed"]:
        warnings.append(
            "No cash rate history is loaded, so the typed risk-free rate was used. "
            "Run `manage.py fetch_risk_free_rates` for point-in-time rates."
        )
    expected_diagnostics["forecast_dispersion"] = (
        float(expected.std(ddof=0)) if expected is not None else None
    )
    expected_diagnostics["risk_free_rate"] = inputs["risk_free_rate"]
    expected_diagnostics["warnings"] = warnings
    study = OptimizationStudy.objects.create(
        portfolio=inputs["portfolio"],
        factor_build=inputs["build"],
        name=parameters.get(
            "name", f"{inputs['portfolio'].name} standard optimization"
        ),
        model_level=inputs["model_level"],
        status="running",
        configuration={
            **parameters,
            "as_of": inputs["as_of"].isoformat(),
            "model_level": inputs["model_level"],
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        },
    )
    progress(50, "solving standard portfolio")
    result = optimize_portfolio(
        inputs["covariance"],
        expected,
        objective=objective,
        **_optimization_kwargs(inputs, parameters),
    )
    scenario = _persist_optimization_variant(
        study=study,
        inputs=inputs,
        parameters=parameters,
        result=result,
        expected=expected,
        name=study.name,
        variant="standard",
        diagnostics=expected_diagnostics,
    )
    study.status = scenario.status
    study.diagnostics = _json(
        {
            "as_of": inputs["as_of"],
            "risk_model": inputs["risk_model"],
            "expected_return_model": expected_model,
            "observation_count": len(inputs["stock_returns"].dropna(how="all")),
            **expected_diagnostics,
        }
    )
    study.save(update_fields=["status", "diagnostics"])
    progress(100, "complete")
    return {
        "study_id": study.pk,
        "scenario_id": scenario.pk,
        "success": result.success,
    }


def monitor_portfolio(parameters: dict, progress) -> dict:
    portfolio = Portfolio.objects.get(pk=parameters["portfolio_id"])
    build = (
        FactorBuild.objects.get(pk=parameters["factor_build_id"])
        if parameters.get("factor_build_id")
        else FactorBuild.objects.filter(
            stock_fits__security__portfolio_holdings__portfolio=portfolio,
            is_canonical=True,
        )
        .order_by("-as_of")
        .first()
    )
    if build is None:
        raise ValueError("A factor build with stock models is required")
    dataset = _dataset_for_build(build)
    requested_as_of = (
        date.fromisoformat(parameters["as_of"])
        if parameters.get("as_of")
        else build.as_of
    )
    holdings = list(portfolio.holdings.select_related("security"))
    weights = _portfolio_weights(portfolio, dataset, requested_as_of)
    model_level = parameters.get("model_level") or portfolio.configuration.get(
        "stock_selection", {}
    ).get("model_level", "base_sector")
    candidates = StockModelFit.objects.filter(
        build=build, security__ticker__in=weights.index, as_of__lte=requested_as_of
    )
    # Mixing levels would blend catalogs, so only fall back when the requested
    # level has no stored fit at all.
    at_level = candidates.filter(model_level=model_level)
    latest = {}
    for fit in (at_level if at_level.exists() else candidates).order_by("-as_of"):
        latest.setdefault(fit.security.ticker, fit)
    if not latest:
        raise ValueError(
            "No monthly stock exposure models are available for this portfolio"
        )
    exposures = pd.DataFrame(
        {
            ticker: {
                item.factor.name: item.beta
                for item in fit.exposures.select_related("factor")
            }
            for ticker, fit in latest.items()
        }
    ).T
    factor_returns = pd.DataFrame.from_records(
        FactorObservation.objects.filter(
            build=build,
            date__lte=requested_as_of,
            factor__name__in=exposures.columns,
        ).values("date", "factor__name", "scaled_return")
    ).pivot(index="date", columns="factor__name", values="scaled_return")
    covered_weights = weights.reindex(exposures.index).dropna()
    coverage = float(covered_weights.sum())
    if coverage <= 0:
        raise ValueError("Portfolio has no modeled weight coverage")
    coverage_adjusted = covered_weights / coverage
    decomposition = portfolio_decomposition(
        coverage_adjusted,
        exposures,
        factor_covariance(factor_returns),
        pd.Series({ticker: fit.residual_volatility for ticker, fit in latest.items()}),
    )
    active = decomposition["exposure"].copy()
    spy_fit = (
        StockModelFit.objects.filter(
            build=build, security__ticker="SPY", as_of__lte=requested_as_of
        )
        .order_by("-as_of")
        .first()
    )
    spy_exposure = pd.Series(
        (
            {
                item.factor.name: item.beta
                for item in spy_fit.exposures.select_related("factor")
            }
            if spy_fit
            else {}
        ),
        dtype=float,
    )
    active = active.subtract(spy_exposure.reindex(active.index).fillna(0))
    omega = (
        factor_covariance(factor_returns)
        .reindex(index=active.index, columns=active.index)
        .fillna(0)
    )
    tracking_variance = (
        float(active @ omega @ active) + decomposition["specific_variance"]
    )
    realized_volatility = None
    drawdown = None
    attribution = {}
    if dataset is not None:
        asset_returns = dataset.prices.reindex(columns=weights.index).pct_change(
            fill_method=None
        )
        portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1).dropna()
        if len(portfolio_returns) >= 20:
            realized_volatility = float(portfolio_returns.tail(60).std() * np.sqrt(252))
            equity = (1 + portfolio_returns).cumprod()
            drawdown = float((equity / equity.cummax() - 1).iloc[-1])
        as_of = max(fit.as_of for fit in latest.values())
        next_month = pd.Period(as_of, freq="M") + 1
        following = asset_returns[asset_returns.index.to_period("M") == next_month]
        if not following.empty:
            realized_assets = (1 + following).prod() - 1
            next_factors = factor_returns[
                pd.to_datetime(factor_returns.index).to_period("M") == next_month
            ]
            factor_period = (1 + next_factors).prod() - 1
            from factors.engine import return_attribution

            attribution = return_attribution(
                coverage_adjusted, exposures, factor_period, realized_assets
            )
    snapshot, _ = PortfolioRiskSnapshot.objects.update_or_create(
        portfolio=portfolio,
        as_of=max(fit.as_of for fit in latest.values()),
        defaults={
            "factor_build": next(iter(latest.values())).build if latest else None,
            "coverage": coverage,
            "exposures": _json(decomposition["exposure"].to_dict()),
            "active_exposures": _json(active.to_dict()),
            "factor_variance": decomposition["factor_variance"],
            "specific_variance": decomposition["specific_variance"],
            "total_variance": decomposition["predicted_variance"],
            "predicted_volatility": decomposition["predicted_volatility"],
            "realized_volatility": realized_volatility,
            "tracking_error": float(np.sqrt(max(tracking_variance, 0))),
            "drawdown": drawdown,
            "concentration": float((weights**2).sum()),
            "component_risk": _json(decomposition["factor_components"].to_dict()),
            "marginal_risk": _json(decomposition["factor_marginal"].to_dict()),
            "attribution": _json(attribution),
        },
    )
    for limit in RiskLimit.objects.filter(portfolio=portfolio, active=True):
        stock_max = float(weights.max())
        sectors = pd.Series(
            {item.security.ticker: item.security.sector for item in holdings}
        )
        sector_max = float(weights.groupby(sectors.reindex(weights.index)).sum().max())
        residual_share = (
            snapshot.specific_variance / snapshot.total_variance
            if snapshot.total_variance
            else 0
        )
        value = {
            "total_volatility": snapshot.predicted_volatility,
            "tracking_error": snapshot.tracking_error,
            "coverage": snapshot.coverage,
            "concentration": snapshot.concentration,
            "stock_weight": stock_max,
            "sector_weight": sector_max,
            "residual_risk_share": residual_share,
            "turnover": float(portfolio.configuration.get("latest_turnover", 0)),
        }.get(limit.metric)
        if limit.metric == "factor_beta" and limit.factor_id:
            value = snapshot.exposures.get(limit.factor.name)
        if limit.metric == "factor_risk_contribution" and limit.factor_id:
            value = snapshot.component_risk.get(limit.factor.name)
        if value is None:
            continue
        lower_slack = (
            value - limit.lower_bound if limit.lower_bound is not None else np.inf
        )
        upper_slack = (
            limit.upper_bound - value if limit.upper_bound is not None else np.inf
        )
        slack = min(lower_slack, upper_slack)
        if slack < 0:
            BreachEvent.objects.update_or_create(
                limit=limit,
                status="open",
                defaults={
                    "risk_snapshot": snapshot,
                    "observed_value": value,
                    "slack": slack,
                },
            )
        else:
            BreachEvent.objects.filter(limit=limit, status="open").update(
                status="resolved", resolved_at=timezone.now()
            )
    progress(100, "complete")
    return {"risk_snapshot_id": snapshot.pk, "coverage": coverage}


def bootstrap_normalized(parameters: dict, progress) -> dict:
    from desk.services import persist_snapshot
    from selection.services import save_portfolio

    root_key = parameters.get("_idempotency_key")
    progress(5, "creating deterministic adjusted-close fixture")
    dataset = synthetic_dataset(
        periods=int(parameters.get("periods", 520)),
        assets=int(parameters.get("assets", 16)),
        seed=int(parameters.get("seed", 7)),
    )
    dataset.provenance["profile"] = "fixture"
    persist_snapshot(dataset, "offline deterministic fixture")
    price_snapshot = PriceSnapshot.objects.latest("created_at")
    build_result = run_proxy_factor_build(
        {
            "price_snapshot_id": price_snapshot.pk,
            "model_version": PROXY_MODEL_VERSION,
            **({"_idempotency_key": f"{root_key}:factor"} if root_key else {}),
        },
        lambda value, stage="": progress(5 + value // 3, stage),
    )
    build_id = build_result["factor_build_id"]
    build_monthly_exposures(
        {
            "factor_build_id": build_id,
            "trailing_days": int(parameters.get("trailing_days", 252)),
            "minimum_observations": int(parameters.get("minimum_observations", 60)),
            "levels": ["base", "base_sector"],
            "max_months": int(parameters.get("max_months", 12)),
            "selection_mode": parameters.get("selection_mode", "fixed"),
        },
        lambda value, stage="": progress(38 + value // 2, stage),
    )
    portfolio = Portfolio.objects.filter(name="Offline Sample Portfolio").first()
    portfolio = save_portfolio(
        "Offline Sample Portfolio",
        [{"ticker": f"DEMO{i:02d}", "weight": 0.1} for i in range(1, 11)],
        portfolio=portfolio,
        equal_weight=True,
    )
    screen, _ = ScreenDefinition.objects.update_or_create(
        name="Offline Momentum and Quality",
        defaults={
            "factor_build_id": build_id,
            "as_of": price_snapshot.as_of,
            "model_level": "base_sector",
            "factor_weights": {"Momentum": 0.6, "Quality": 0.4},
            "filters": {"coverage": {"min": 0.7}},
            "top_n": 10,
        },
    )
    screen_result = run_screen(
        {
            "screen_id": screen.pk,
            **({"_idempotency_key": f"{root_key}:screen"} if root_key else {}),
        },
        lambda *_: None,
    )
    RiskLimit.objects.get_or_create(
        portfolio=portfolio,
        name="Minimum model coverage",
        defaults={"metric": "coverage", "lower_bound": 0.8},
    )
    risk_result = monitor_portfolio(
        {"portfolio_id": portfolio.pk, "factor_build_id": build_id}, lambda *_: None
    )
    optimization_result = run_optimization(
        {
            "portfolio_id": portfolio.pk,
            "factor_build_id": build_id,
            "name": "Offline Minimum Variance",
            "objective": "min_variance",
            "max_weight": 0.15,
            "residual_shrinkage": 0.8,
            **({"_idempotency_key": f"{root_key}:optimization"} if root_key else {}),
        },
        lambda *_: None,
    )
    backtest_result = run_backtest(
        {
            "mode": "screen_optimized",
            "screen_id": screen.pk,
            "scenario_id": optimization_result["scenario_id"],
            "factor_build_id": build_id,
            "name": "Offline Monthly Portfolio Backtest",
            "estimation_window": 126,
            "cost_scenarios": [10, 25, 50],
            **({"_idempotency_key": f"{root_key}:backtest"} if root_key else {}),
        },
        lambda *_: None,
    )
    progress(100, "complete")
    return {
        "price_snapshot_id": price_snapshot.pk,
        "factor_build_id": build_id,
        "portfolio_id": portfolio.pk,
        **screen_result,
        **risk_result,
        **optimization_result,
        **backtest_result,
    }


class _PointInTimeAllocator:
    def __init__(
        self, build, dataset, mode, portfolio=None, screen=None, scenario=None
    ):
        self.build = build
        self.dataset = dataset
        self.mode = mode
        self.portfolio = portfolio
        self.screen = screen
        self.scenario = scenario
        self.decisions = {}
        self.previous = None
        self.previous_date = None
        records = list(
            FactorObservation.objects.filter(build=build).values(
                "date", "factor__name", "scaled_return"
            )
        )
        self.factor_panel = (
            pd.DataFrame.from_records(records).pivot(
                index="date", columns="factor__name", values="scaled_return"
            )
            if records
            else pd.DataFrame()
        )
        if not self.factor_panel.empty:
            self.factor_panel.index = pd.to_datetime(self.factor_panel.index)

    def _drifted_previous(self, signal_date):
        if self.previous is None or self.previous_date is None:
            return None
        period = self.dataset.prices.loc[
            (self.dataset.prices.index > self.previous_date)
            & (self.dataset.prices.index <= signal_date),
            self.previous.index,
        ]
        growth = (1 + period.pct_change(fill_method=None).fillna(0)).prod()
        drifted = self.previous * growth
        return drifted / drifted.sum() if drifted.sum() else self.previous

    def _equations(self, signal_date):
        result = {}
        for equation in self.build.equations.select_related("factor"):
            coefficients = {
                stamp: values
                for stamp, values in equation.regression_betas.items()
                if date.fromisoformat(stamp) <= signal_date.date()
            }
            result[equation.factor.name] = {
                **equation.equation,
                "coefficients": coefficients,
            }
        return result

    def __call__(self, history):
        signal_stamp = history.index[-1]
        signal_date = signal_stamp.date()
        drifted = self._drifted_previous(signal_stamp)
        if self.mode == "manual":
            target = _portfolio_weights(self.portfolio, self.dataset, signal_date)
            selected = list(target.index)
            decision = {
                "status": "manual_frozen",
                "selected": selected,
                "optimizer": {"status": "not_requested"},
            }
            fits = {}
            for fit in (
                StockModelFit.objects.filter(
                    build=self.build,
                    security__ticker__in=selected,
                    as_of__lte=signal_date,
                )
                .prefetch_related("exposures__factor")
                .order_by("-as_of")
            ):
                fits.setdefault(fit.security.ticker, fit)
            omega = (
                factor_covariance(self.factor_panel.loc[:signal_stamp].tail(252))
                if not self.factor_panel.empty
                else pd.DataFrame()
            )
        else:
            features, sectors, fits_by_id = _screen_features(self.screen, signal_date)
            latest_observations = {}
            for item in (
                FactorObservation.objects.filter(
                    build=self.build, date__lte=signal_date
                )
                .select_related("factor")
                .order_by("factor_id", "-date")
            ):
                latest_observations.setdefault(item.factor.name, item)
            regime = {
                name: float(latest_observations[name].zscores.get("63") or 0)
                for name in self.screen.regime_interactions
                if name in latest_observations
            }
            preview, selected, manual_tickers = _screen_preview(
                self.screen, features, sectors, regime
            )
            unavailable_manual = sorted(
                set(manual_tickers) - set(self.dataset.prices.columns)
            )
            if unavailable_manual:
                raise ValueError(
                    "Manual additions lack backtest prices: "
                    + ", ".join(unavailable_manual)
                )
            if not selected:
                raise ValueError(f"No screen selections at {signal_date}")
            fits = {
                fit.security.ticker: fit
                for fit in fits_by_id.values()
                if fit.security.ticker in selected
            }
            exposure_frame = pd.DataFrame.from_dict(
                {
                    ticker: {
                        item.factor.name: item.beta for item in fit.exposures.all()
                    }
                    for ticker, fit in fits.items()
                },
                orient="index",
            ).fillna(0)
            factor_history = self.factor_panel.loc[:signal_stamp].tail(252)
            omega = factor_covariance(factor_history)
            if self.mode == "screen":
                target = pd.Series(1 / len(selected), index=selected)
                decision = {
                    "status": "screen_equal_weight",
                    "selected": selected,
                    "scores": preview["composite"].dropna().to_dict(),
                    "selection_sources": {
                        ticker: ("manual" if ticker in manual_tickers else "model")
                        for ticker in selected
                    },
                    "optimizer": {"status": "not_requested"},
                }
            else:
                config = self.scenario.configuration
                specific = pd.Series(
                    {ticker: fit.residual_volatility for ticker, fit in fits.items()}
                )
                returns = (
                    self.dataset.prices.loc[:signal_stamp, selected]
                    .pct_change(fill_method=None)
                    .tail(252)
                )
                covariance = factor_risk_covariance(
                    exposure_frame,
                    omega,
                    specific,
                    residual_correlation=returns.corr(),
                    shrinkage=float(config.get("residual_shrinkage", 0.8)),
                )
                expected_model = self.scenario.expected_return_model
                expected = (
                    pd.Series(config.get("expected_returns", {}), dtype=float)
                    if expected_model
                    else None
                )
                if self.scenario.objective in {
                    "max_sharpe",
                    "factor_score_utility",
                } and (expected is None or not set(selected).issubset(expected.index)):
                    raise ValueError(
                        "Dynamic optimizer lacks explicit expected returns for selected names"
                    )
                spy_fit = (
                    StockModelFit.objects.filter(
                        build=self.build,
                        security__ticker="SPY",
                        as_of__lte=signal_date,
                    )
                    .prefetch_related("exposures__factor")
                    .order_by("-as_of")
                    .first()
                )
                spy_betas = pd.Series(
                    (
                        {
                            item.factor.name: item.beta
                            for item in spy_fit.exposures.all()
                        }
                        if spy_fit
                        else {}
                    ),
                    dtype=float,
                )
                result = optimize_portfolio(
                    covariance,
                    expected,
                    objective=self.scenario.objective,
                    previous_weights=(
                        drifted.reindex(selected).fillna(0)
                        if drifted is not None
                        else None
                    ),
                    max_weight=float(config.get("max_weight", 0.15)),
                    turnover_cap=config.get("turnover_cap"),
                    turnover_penalty=float(config.get("turnover_penalty", 0)),
                    transaction_cost=float(config.get("transaction_cost", 0)),
                    sectors=sectors.reindex(selected),
                    sector_bounds={
                        name: tuple(bounds)
                        for name, bounds in config.get("sector_bounds", {}).items()
                    },
                    factor_exposures=exposure_frame,
                    factor_bounds=config.get("factor_bounds"),
                    relative_factor_bounds=config.get("relative_factor_bounds"),
                    factor_covariance=omega,
                    max_factor_variance=config.get("max_factor_variance"),
                    max_factor_components=config.get("max_factor_components"),
                    tracking_error_limit=config.get("tracking_error_limit"),
                    benchmark_factor_exposures=spy_betas,
                    specific_variances=specific.pow(2),
                )
                decision = {
                    "selected": selected,
                    "optimizer": {
                        "status": "succeeded" if result.success else "infeasible",
                        "message": result.message,
                        "diagnostics": result.diagnostics,
                        "fallback_used": False,
                    },
                }
                if not result.success:
                    decision.update(
                        {
                            "weights": drifted.to_dict() if drifted is not None else {},
                            "betas": exposure_frame.to_dict("index"),
                            "covariance": omega.to_dict(),
                            "equations": self._equations(signal_stamp),
                        }
                    )
                    self.decisions[signal_stamp] = decision
                    raise RuntimeError(f"Optimizer infeasible at {signal_date}")
                target = result.weights
        self.previous = target
        self.previous_date = signal_stamp
        betas = {
            ticker: {item.factor.name: item.beta for item in fit.exposures.all()}
            for ticker, fit in fits.items()
        }
        decision.update(
            {
                "weights": target.to_dict(),
                "betas": betas,
                "covariance": omega.to_dict(),
                "equations": self._equations(signal_stamp),
            }
        )
        self.decisions[signal_stamp] = decision
        return target


def _expected_returns_for_replay(inputs: dict, parameters: dict) -> pd.Series | None:
    from optimization.forecasts import shrink_annualized_expected_returns

    model = parameters.get("expected_return_model") or ""
    objective = parameters.get("objective", "min_variance")
    expected = None
    if model == "user_supplied":
        expected = pd.Series(parameters.get("expected_returns", {}), dtype=float)
    elif model == "factor_premium":
        expected = _factor_premium_forecast(inputs, parameters)[0].expected_returns
    elif model == "equal_sharpe":
        expected = _equal_sharpe_forecast(inputs, parameters).expected_returns
    elif model == "historical_shrinkage":
        expected = shrink_annualized_expected_returns(
            inputs["stock_returns"]
        ).expected_returns
    if objective in {
        "max_return",
        "mean_variance",
        "max_sharpe",
        "factor_score_utility",
    } and (expected is None or not set(inputs["names"]).issubset(expected.index)):
        raise ValueError(
            f"{objective} requires an expected return model covering every holding"
        )
    return expected


def _merge_optimizer_history(optimized: dict, original: dict) -> list[dict]:
    original_by_date = {row["date"]: row for row in original["chart_rows"]}
    rows = []
    for row in optimized["chart_rows"]:
        other = original_by_date.get(row["date"])
        if other is None:
            continue
        rows.append(
            {
                "date": row["date"],
                "optimized_equity": row["portfolio_equity"],
                "original_equity": other["portfolio_equity"],
                "spy_equity": row["spy_equity"],
                "optimized_drawdown": row["portfolio_drawdown"],
                "original_drawdown": other["portfolio_drawdown"],
                "spy_drawdown": row["spy_drawdown"],
            }
        )
    return rows


def run_scenario_backtest(parameters: dict, progress) -> dict:
    from selection.history import (
        _prices_for_build,
        _simulate,
        monthly_selection_decisions,
    )

    scenario = OptimizationScenario.objects.select_related(
        "study", "portfolio", "study__factor_build"
    ).get(pk=parameters["scenario_id"])
    if scenario.status != "succeeded":
        raise ValueError("Only succeeded variants can be compared historically")
    portfolio = scenario.portfolio
    if portfolio is None:
        raise ValueError("This variant is not tied to a portfolio")
    selection = (portfolio.configuration or {}).get("stock_selection") or {}
    if not selection:
        raise ValueError(
            "This 24-month comparison needs a Stock Selection portfolio so each "
            "month can use that month's ranked names."
        )
    lookback_months = int(parameters.get("lookback_months", 24))
    build = FactorBuild.objects.get(
        pk=parameters.get("factor_build_id")
        or selection.get("factor_build_id")
        or scenario.study.factor_build_id
    )
    config = {
        **selection,
        "build_id": build.pk,
        "model_level": parameters.get("model_level")
        or selection.get("model_level")
        or scenario.study.model_level,
    }
    idempotency_key = parameters.get("_idempotency_key") or (
        f"scenario-backtest:{scenario.pk}:{build.pk}:{lookback_months}"
    )
    prior = BacktestRun.objects.filter(
        optimization_scenario=scenario,
        lookback_months=lookback_months,
        status="succeeded",
    ).first()
    if prior:
        return {
            "backtest_run_id": prior.pk,
            "scenario_id": scenario.pk,
            "rebalances": prior.rebalances.count(),
        }
    BacktestRun.objects.filter(
        optimization_scenario=scenario,
        lookback_months=lookback_months,
        status__in=["pending", "failed"],
    ).delete()
    progress(10, "rebuilding monthly ranked names")
    decisions = monthly_selection_decisions(build, config, max_months=lookback_months)
    study_as_of = (scenario.study.configuration or {}).get("as_of")
    if study_as_of:
        cutoff = date.fromisoformat(str(study_as_of))
        decisions = [item for item in decisions if item["signal_date"] <= cutoff]
    if not decisions:
        raise ValueError(
            "No monthly model history is stored. Queue monthly exposures first."
        )
    dataset = _dataset_for_build(build)
    parameters_for_solve = {
        **(scenario.configuration or {}),
        "model_level": config["model_level"],
        "risk_model": "factor_model",
        "lookback": (scenario.configuration or {}).get("lookback", 756),
        "residual_shrinkage": (scenario.configuration or {}).get(
            "residual_shrinkage", 0.8
        ),
    }
    warnings = [
        (
            "Current-universe membership and replication-mode factor scaling may introduce "
            "survivorship or look-ahead bias; use a walk-forward backtest for rigorous results."
        )
    ]
    optimized_decisions = []
    rebalance_rows = []
    total = len(decisions)
    for index, decision in enumerate(decisions, start=1):
        progress(
            15 + int(60 * index / total),
            f"optimizing {decision['signal_date']}",
        )
        names = list(decision["tickers"])
        equal = pd.Series(1 / len(names), index=names)
        try:
            risk = _risk_inputs_for_tickers(
                build=build,
                names=names,
                as_of=decision["signal_date"],
                parameters=parameters_for_solve,
                dataset=dataset,
                model_level=config["model_level"],
            )
            securities = {
                item.ticker: item for item in Security.objects.filter(ticker__in=names)
            }
            holdings = [
                SimpleNamespace(security=securities[ticker])
                for ticker in names
                if ticker in securities
            ]
            inputs = {**risk, "holdings": holdings, "original": equal}
            expected = _expected_returns_for_replay(inputs, parameters_for_solve)
            result = optimize_portfolio(
                inputs["covariance"],
                expected,
                objective=parameters_for_solve.get("objective", "min_variance"),
                **_optimization_kwargs(inputs, parameters_for_solve),
            )
            if not result.success:
                raise RuntimeError(result.message)
            weights = result.weights
            status = "succeeded"
        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            warnings.append(f"{decision['signal_date']}: used equal weight ({exc})")
            weights = equal
            status = "fallback_equal"
        optimized_decisions.append({**decision, "weights": weights.to_dict()})
        rebalance_rows.append(
            {
                "signal_date": str(decision["signal_date"]),
                "tickers": names,
                "original_weights": equal.to_dict(),
                "optimized_weights": weights.to_dict(),
                "model_count": decision.get("model_count"),
                "manual_count": decision.get("manual_count"),
                "status": status,
            }
        )
    progress(80, "simulating original and optimized paths")
    prices = _prices_for_build(build)
    original_history = _simulate(prices, decisions)
    optimized_history = _simulate(prices, optimized_decisions)
    execution_by_signal = {
        str(item["signal_date"]): item for item in optimized_history["rebalances"]
    }
    for row in rebalance_rows:
        matched = execution_by_signal.get(str(row["signal_date"]))
        if matched:
            row["execution_date"] = str(matched["execution_date"])
            row["turnover"] = matched.get("turnover")
        else:
            row["execution_date"] = None
            row["turnover"] = None
    rebalance_rows = [row for row in rebalance_rows if row["execution_date"]]
    chart_rows = _merge_optimizer_history(optimized_history, original_history)
    started = timezone.now()
    run = BacktestRun.objects.create(
        name=f"{scenario.name} 24-month comparison",
        configuration={
            "scenario_id": scenario.pk,
            "study_id": scenario.study_id,
            "lookback_months": lookback_months,
            "idempotency_key": idempotency_key,
            "mode": "portfolio_optimized",
        },
        status="succeeded",
        optimization_scenario=scenario,
        portfolio=portfolio,
        lookback_months=lookback_months,
        methodology_version="optimizer-approx-24m",
        started_at=started,
        finished_at=timezone.now(),
        warnings=list(dict.fromkeys(warnings)),
        metrics=_json(
            {
                "optimized": optimized_history["metrics"],
                "original": original_history["metrics"],
                "spy": optimized_history["benchmark_metrics"],
                "chart_rows": chart_rows,
                "rebalances": rebalance_rows,
                "start": str(optimized_history["start"]),
                "end": str(optimized_history["end"]),
                "observations": optimized_history["observations"],
                "methodology": (
                    "Approximate stored-exposure replay. Each month uses that month's "
                    "ranked names. Original equal-weights those names; Optimized applies "
                    "the selected method and constraints. Both execute the next trading "
                    "day and are gross of transaction costs."
                ),
                "warnings": list(dict.fromkeys(warnings)),
            }
        ),
    )
    BacktestRebalance.objects.bulk_create(
        [
            BacktestRebalance(
                run=run,
                signal_date=row["signal_date"],
                execution_date=row["execution_date"],
                holdings=_json(row["optimized_weights"]),
                optimizer=_json(
                    {
                        "status": row["status"],
                        "original_weights": row["original_weights"],
                        "tickers": row["tickers"],
                    }
                ),
                turnover=row["turnover"] or 0,
            )
            for row in rebalance_rows
        ]
    )
    progress(100, "complete")
    return {
        "backtest_run_id": run.pk,
        "scenario_id": scenario.pk,
        "rebalances": len(rebalance_rows),
    }


def run_backtest(parameters: dict, progress) -> dict:
    build = FactorBuild.objects.get(pk=parameters["factor_build_id"])
    if build.definition_version != PROXY_MODEL_VERSION:
        raise ValueError(
            f"Backtests require a {PROXY_MODEL_VERSION} proxy factor build"
        )
    idempotency_key = parameters.get("_idempotency_key")
    if idempotency_key:
        prior = BacktestRun.objects.filter(
            configuration__idempotency_key=idempotency_key
        ).first()
        if prior and prior.status == "succeeded":
            return {"backtest_run_id": prior.pk, "rebalances": prior.rebalances.count()}
        if prior:
            prior.delete()
    mode = parameters.get("mode", "manual")
    if mode not in {"manual", "screen", "screen_optimized"}:
        raise ValueError("Backtest mode must be manual, screen, or screen_optimized")
    portfolio = (
        Portfolio.objects.get(pk=parameters["portfolio_id"])
        if parameters.get("portfolio_id")
        else None
    )
    screen = (
        ScreenDefinition.objects.get(pk=parameters["screen_id"])
        if parameters.get("screen_id")
        else None
    )
    scenario = (
        OptimizationScenario.objects.get(pk=parameters["scenario_id"])
        if parameters.get("scenario_id")
        else None
    )
    if mode == "manual" and portfolio is None:
        raise ValueError("Manual backtest requires a portfolio")
    if mode == "screen" and screen is None:
        raise ValueError("Screen backtest requires a screen")
    if mode == "screen_optimized":
        if scenario is None:
            raise ValueError("Optimized screen backtest requires a scenario")
        screen = (
            screen or scenario.screen_run.definition
            if scenario.screen_run_id
            else screen
        )
        if screen is None:
            raise ValueError("Scenario must reference a screen run or screen_id")
        if parse_ticker_list((screen.filters or {}).get("manual_tickers", [])):
            raise ValueError(
                "Optimized screen backtests do not support manual additions; "
                "use the equal-weight screen mode"
            )
    dataset = _dataset_for_build(build)
    metadata = dataset.metadata.set_index("ticker")
    stock_names = metadata.index[metadata["asset_type"].eq("stock")].intersection(
        dataset.prices.columns
    )
    if screen is not None:
        stock_names = stock_names.union(
            pd.Index(
                parse_ticker_list((screen.filters or {}).get("manual_tickers", []))
            ).intersection(dataset.prices.columns)
        )
    prices = (
        dataset.prices.reindex(
            columns=[
                item.security.ticker
                for item in portfolio.holdings.select_related("security")
            ]
        )
        if mode == "manual"
        else dataset.prices.reindex(columns=stock_names)
    ).dropna(how="all")
    estimation_window = int(parameters.get("estimation_window", 126))
    costs = parameters.get("cost_scenarios", [10, 25, 50])
    if isinstance(costs, str):
        costs = [float(item) for item in costs.split(",") if item.strip()]
    if not costs:
        raise ValueError("Provide at least one transaction-cost scenario")
    selection_config = (
        {
            "factor_build_id": screen.factor_build_id,
            "model_level": screen.model_level,
            "factor_weights": screen.factor_weights,
            "directions": screen.directions,
            "top_n": screen.top_n,
            "filters": screen.filters,
        }
        if screen
        else None
    )
    run = BacktestRun.objects.create(
        name=parameters.get(
            "name",
            f"{portfolio.name if portfolio else screen.name} {mode} monthly backtest",
        ),
        portfolio=portfolio,
        screen=screen,
        configuration={
            **parameters,
            "history_methodology": "approximate_current_universe",
            "point_in_time_universe": False,
            "survivorship_biased": True,
            **(
                {"selection_config": selection_config}
                if selection_config is not None
                else {}
            ),
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        },
        methodology_version=PROXY_MODEL_VERSION,
        status="running",
        started_at=timezone.now(),
    )
    comparisons = {}
    primary = None
    primary_allocator = None
    for position, cost in enumerate(costs):
        allocator = _PointInTimeAllocator(
            build, dataset, mode, portfolio=portfolio, screen=screen, scenario=scenario
        )
        result = monthly_event_backtest(
            prices,
            allocator,
            estimation_window=estimation_window,
            transaction_cost_bps=float(cost),
        )
        comparisons[f"{float(cost):g}bps"] = result.metrics
        if primary is None or float(cost) == 25:
            primary = result
            primary_allocator = allocator
        progress(
            15 + int(45 * (position + 1) / len(costs)), f"cost scenario {cost} bps"
        )
    spy = (
        dataset.prices["SPY"]
        .pct_change(fill_method=None)
        .reindex(primary.returns.index)
    )
    primary.metrics = performance_statistics(primary.returns, spy)
    primary.metrics.update(
        {
            "turnover": float(primary.turnover.sum()),
            "transaction_cost": float(primary.costs.sum()),
        }
    )

    def equal_selected(history):
        decision = primary_allocator.decisions.get(history.index[-1])
        if not decision or not decision.get("selected"):
            raise ValueError("No point-in-time selection")
        selected = decision["selected"]
        return pd.Series(1 / len(selected), index=selected)

    if mode == "manual":
        manual_names = list(prices.columns)
        equal = pd.Series(1 / len(manual_names), index=manual_names)
        equal_allocator = lambda history: equal
    else:
        equal_allocator = equal_selected
    equal_result = monthly_event_backtest(
        prices,
        equal_allocator,
        estimation_window=estimation_window,
        transaction_cost_bps=25,
    )
    equal_spy = spy.reindex(equal_result.returns.index)
    equal_result.metrics = performance_statistics(equal_result.returns, equal_spy)
    equal_result.metrics.update(
        {
            "turnover": float(equal_result.turnover.sum()),
            "transaction_cost": float(equal_result.costs.sum()),
        }
    )
    rng = np.random.default_rng(int(parameters.get("seed", 7)))
    candidates = metadata.index[metadata["asset_type"].eq("stock")].intersection(
        dataset.prices.columns
    )
    random_names = rng.choice(
        candidates,
        size=min(
            (
                len(prices.columns)
                if mode == "manual"
                else max(
                    (
                        len(decision.get("selected", []))
                        for decision in primary_allocator.decisions.values()
                    ),
                    default=screen.top_n,
                )
            ),
            len(candidates),
        ),
        replace=False,
    )
    random_weights = pd.Series(1 / len(random_names), index=random_names)
    random_result = monthly_event_backtest(
        dataset.prices.reindex(columns=random_names),
        lambda history: random_weights,
        estimation_window=estimation_window,
        transaction_cost_bps=25,
    )
    comparisons.update(
        {
            "portfolio": primary.metrics,
            "equal_weight": equal_result.metrics,
            "random_n": random_result.metrics,
            "SPY": (
                performance_statistics(spy.dropna()) if spy.notna().sum() >= 20 else {}
            ),
        }
    )
    selection_diagnostics = {}
    if screen is not None:
        score_rows = {}
        forward_rows = {}
        selection_rows = {}
        asset_returns = dataset.prices.pct_change(fill_method=None)
        for signal_stamp, decision in primary_allocator.decisions.items():
            scores = pd.Series(decision.get("scores", {}), dtype=float)
            if scores.empty:
                continue
            next_period = signal_stamp.to_period("M") + 1
            next_rows = asset_returns[asset_returns.index.to_period("M") == next_period]
            if next_rows.empty:
                continue
            score_rows[signal_stamp] = scores
            forward_rows[signal_stamp] = (1 + next_rows).prod() - 1
            selection_rows[signal_stamp] = pd.Series(
                True, index=decision.get("selected", [])
            )
        if score_rows:
            selection_diagnostics = screen_backtest_statistics(
                pd.DataFrame(score_rows).T,
                pd.DataFrame(forward_rows).T,
                pd.DataFrame(selection_rows).T.fillna(False),
                transaction_cost_bps=25,
            )
    for signal_stamp, decision in sorted(primary_allocator.decisions.items()):
        signal_position = prices.index.get_loc(signal_stamp)
        if signal_position + 1 >= len(prices.index):
            continue
        stamp = prices.index[signal_position + 1]
        turnover = primary.turnover.get(stamp, 0)
        betas = decision.get("betas", {})
        omega = pd.DataFrame(decision.get("covariance", {}))
        exposure_frame = pd.DataFrame.from_dict(betas, orient="index").fillna(0)
        current_fits = {}
        for fit in StockModelFit.objects.filter(
            build=build,
            security__ticker__in=exposure_frame.index,
            as_of__lte=signal_stamp.date(),
        ).order_by("-as_of"):
            current_fits.setdefault(fit.security.ticker, fit)
        current_weights = (
            primary.weights.loc[stamp].reindex(exposure_frame.index).fillna(0)
        )
        specific = pd.Series(
            {ticker: fit.residual_volatility for ticker, fit in current_fits.items()}
        )
        risk = (
            portfolio_decomposition(current_weights, exposure_frame, omega, specific)
            if not exposure_frame.empty and not omega.empty
            else {}
        )
        breaches = []
        limit_portfolio = portfolio or (scenario.portfolio if scenario else None)
        if limit_portfolio and risk:
            for limit in limit_portfolio.risk_limits.filter(active=True).select_related(
                "factor"
            ):
                observed = (
                    risk.get("exposure", {}).get(limit.factor.name)
                    if limit.factor_id
                    else risk.get(limit.metric)
                )
                if observed is None:
                    continue
                observed = float(observed)
                if limit.lower_bound is not None and observed < limit.lower_bound:
                    breaches.append(
                        {
                            "limit_id": limit.pk,
                            "name": limit.name,
                            "observed": observed,
                            "slack": observed - limit.lower_bound,
                        }
                    )
                if limit.upper_bound is not None and observed > limit.upper_bound:
                    breaches.append(
                        {
                            "limit_id": limit.pk,
                            "name": limit.name,
                            "observed": observed,
                            "slack": limit.upper_bound - observed,
                        }
                    )
        BacktestRebalance.objects.create(
            run=run,
            signal_date=signal_stamp.date(),
            execution_date=stamp.date(),
            holdings=_json(
                primary.weights.loc[stamp].loc[lambda values: values > 0].to_dict()
            ),
            equations=decision.get("equations", {}),
            exposures=betas,
            covariance=_json(omega.to_dict()),
            optimizer=_json(
                {
                    **decision.get("optimizer", {"status": "not_requested"}),
                    "selection_sources": decision.get("selection_sources", {}),
                }
            ),
            risk=_json(risk),
            costs=float(primary.costs.loc[stamp]),
            turnover=float(turnover),
            breaches=breaches,
        )
    export_frame = pd.DataFrame(
        {
            "portfolio_return": primary.returns,
            "SPY_return": spy,
            "portfolio_equity": primary.equity,
            "SPY_equity": (1 + spy.fillna(0)).cumprod(),
            "turnover": primary.turnover.reindex(primary.returns.index).fillna(0),
            "cost": primary.costs.reindex(primary.returns.index).fillna(0),
        }
    )
    export_frame["portfolio_drawdown"] = (
        export_frame["portfolio_equity"] / export_frame["portfolio_equity"].cummax() - 1
    )
    export_frame["SPY_drawdown"] = (
        export_frame["SPY_equity"] / export_frame["SPY_equity"].cummax() - 1
    )
    run.status = "succeeded"
    run.finished_at = timezone.now()
    common = export_frame[["portfolio_return", "SPY_return"]].dropna()
    run.metrics = _json(
        {
            "comparison": comparisons,
            "selection": selection_diagnostics,
            "period": {
                "start": common.index.min() if not common.empty else None,
                "end": common.index.max() if not common.empty else None,
                "observations": len(common),
            },
            "series": [
                {"date": stamp.isoformat(), **_json(row)}
                for stamp, row in export_frame.iterrows()
            ],
        }
    )
    run.warnings = [
        "The stock universe uses current constituents and is subject to survivorship bias.",
        *(
            ["Fewer than three years of common strategy and SPY observations."]
            if len(common) < 3 * 252
            else []
        ),
    ]
    run.result_path = ""
    run.save()
    progress(100, "complete")
    return {"backtest_run_id": run.pk, "rebalances": run.rebalances.count()}
