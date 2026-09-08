"""Canonical factor-dataset identity and display metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from django.db.models import Count, Max

from factors.models import FactorDefinition, FactorModelCatalog, StockModelFit

PIPELINE_SCHEMA_VERSION = "canonical-incremental-v1"


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def canonical_model_fingerprint(
    model_version: str,
    *,
    mode: str,
    configuration: dict[str, Any] | None = None,
) -> str:
    """Hash methodology inputs, never data dates or execution identifiers."""
    ignored = {
        "_idempotency_key",
        "idempotency_key",
        "price_snapshot_id",
        "snapshot_id",
        "start",
        "end",
        "workers",
        "max_months",
        "tickers",
        "top_n_by_market_cap",
    }
    stable_configuration = {
        key: value for key, value in (configuration or {}).items() if key not in ignored
    }
    definitions = list(
        FactorDefinition.objects.filter(model_version=model_version)
        .order_by("sort_order", "name")
        .values(
            "name",
            "family",
            "level",
            "external_name",
            "configuration",
            "active",
        )
    )
    catalogs = []
    for catalog in FactorModelCatalog.objects.filter(
        model_version=model_version
    ).prefetch_related("memberships__factor"):
        catalogs.append(
            {
                "slug": catalog.slug,
                "factors": [
                    membership.factor.name for membership in catalog.memberships.all()
                ],
            }
        )
    payload = {
        "pipeline_schema": PIPELINE_SCHEMA_VERSION,
        "model_version": model_version,
        "mode": mode,
        "configuration": stable_configuration,
        "definitions": definitions,
        "catalogs": sorted(catalogs, key=lambda item: item["slug"]),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_stock_coverage(build, model_level: str) -> dict[str, Any]:
    fits = StockModelFit.objects.filter(build=build, model_level=model_level)
    latest = fits.aggregate(latest_as_of=Max("as_of"))["latest_as_of"]
    latest_stocks = (
        fits.filter(as_of=latest).values("security_id").distinct().count()
        if latest
        else 0
    )
    per_stock = list(
        fits.values("security_id")
        .annotate(months=Count("period", distinct=True))
        .values_list("months", flat=True)
    )
    per_stock.sort()
    return {
        "latest_as_of": latest,
        "latest_stocks": latest_stocks,
        "stocks_with_history": sum(months > 1 for months in per_stock),
        "median_months": per_stock[len(per_stock) // 2] if per_stock else 0,
        "maximum_months": max(per_stock, default=0),
    }


def model_dataset_label(build, model_level: str | None = None) -> str:
    label = f"{build.definition_version} · {build.as_of:%b %d, %Y}"
    if model_level:
        coverage = build_stock_coverage(build, model_level)
        label += f" · {coverage['latest_stocks']} stocks"
    return label
