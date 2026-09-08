from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from market_data.models import DataSnapshot
from market_data.providers import (
    MarketDataset,
    YahooWikipediaProvider,
)
from market_data.store import persist_price_dataset


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _field_value(value):
    """Yahoo leaves NaN for proxies and metadata failures; model fields need None."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if value is not pd.NaT and pd.isna(value):
        return None
    return value


def persist_snapshot(dataset: MarketDataset, source: str) -> DataSnapshot:
    price_snapshot = persist_price_dataset(dataset, source)
    return DataSnapshot.objects.create(
        source=source,
        as_of=dataset.prices.index.max().date(),
        universe=_json_safe(dataset.metadata.to_dict("records")),
        provenance=_json_safe(dataset.provenance),
        warnings=_json_safe(dataset.provenance.get("warnings", [])),
        checksum=price_snapshot.checksum,
    )


def run_live(
    start: str, end: str, limit: int | None, progress=lambda value: None
) -> dict:
    progress(5)
    dataset = YahooWikipediaProvider().download(start, end, limit=limit)
    progress(80)
    snapshot = persist_snapshot(dataset, "yfinance+wikipedia")
    progress(100)
    return {
        "snapshot_id": snapshot.pk,
        "rows": len(dataset.prices),
        "stocks": int(dataset.metadata["asset_type"].eq("stock").sum()),
        "factor_proxies": int(dataset.metadata["asset_type"].eq("factor_proxy").sum()),
        "warnings": dataset.provenance.get("warnings", []),
    }
