from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
from django.db import transaction

from .models import PriceBar, PriceSnapshot, Security, Universe, UniverseSnapshot
from .providers import MarketDataset, normalize_ticker


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _field_value(value):
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def upsert_securities(metadata: pd.DataFrame, source: str) -> dict[str, Security]:
    rows = metadata.to_dict("records") if not metadata.empty else []
    for row in rows:
        ticker = normalize_ticker(str(row.get("ticker", "")))
        if not ticker:
            continue
        metadata_as_of = _field_value(row.get("metadata_as_of"))
        if metadata_as_of is not None:
            metadata_as_of = pd.Timestamp(metadata_as_of).date()
        Security.objects.update_or_create(
            ticker=ticker,
            defaults={
                "name": _field_value(row.get("Security", row.get("name"))) or "",
                "asset_type": _field_value(row.get("asset_type")) or "stock",
                "sector": _field_value(row.get("sector")) or "",
                "industry": (
                    _field_value(row.get("GICS Sub-Industry", row.get("industry")))
                    or ""
                ),
                "exchange": _field_value(row.get("exchange")) or "",
                "currency": _field_value(row.get("currency")) or "USD",
                "current_shares": _field_value(row.get("current_shares")),
                "market_cap": _field_value(row.get("market_cap")),
                "metadata_as_of": metadata_as_of,
                "provenance": {"source": source},
            },
        )
    tickers = {
        normalize_ticker(str(ticker))
        for ticker in metadata.get("ticker", pd.Series(dtype=str))
    }
    return {
        security.ticker: security
        for security in Security.objects.filter(ticker__in=tickers)
    }


def upsert_price_rows(rows: list[PriceBar], *, batch_size: int = 5000) -> int:
    if not rows:
        return 0
    PriceBar.objects.bulk_create(
        rows,
        batch_size=batch_size,
        update_conflicts=True,
        update_fields=["close"],
        unique_fields=["security", "date"],
    )
    clear_price_dataset_cache()
    return len(rows)


def upsert_price_frame(
    prices: pd.DataFrame, securities: dict[str, Security] | None = None
) -> int:
    normalized = prices.copy()
    normalized.columns = [normalize_ticker(str(column)) for column in normalized.columns]
    if securities is None:
        securities = {
            security.ticker: security
            for security in Security.objects.filter(ticker__in=normalized.columns)
        }
    missing = sorted(set(normalized.columns) - set(securities))
    if missing:
        Security.objects.bulk_create(
            [Security(ticker=ticker) for ticker in missing], ignore_conflicts=True
        )
        securities.update(
            {
                security.ticker: security
                for security in Security.objects.filter(ticker__in=missing)
            }
        )
    rows = [
        PriceBar(
            security=securities[ticker],
            date=pd.Timestamp(stamp).date(),
            close=float(value),
        )
        for stamp, values in normalized.sort_index().iterrows()
        for ticker, value in values.items()
        if pd.notna(value) and math.isfinite(float(value))
    ]
    return upsert_price_rows(rows)


def price_book_checksum() -> str:
    digest = hashlib.sha256()
    rows = PriceBar.objects.order_by("security__ticker", "date").values_list(
        "security__ticker", "date", "close"
    )
    for ticker, stamp, close in rows.iterator(chunk_size=5000):
        digest.update(
            f"{ticker}\t{stamp.isoformat()}\t{float(close):.17g}\n".encode("utf-8")
        )
    return digest.hexdigest()


@transaction.atomic
def persist_price_dataset(dataset: MarketDataset, source: str) -> PriceSnapshot:
    if dataset.prices.empty:
        raise ValueError("Cannot persist an empty price dataset")
    metadata = dataset.metadata.copy()
    if "ticker" not in metadata:
        metadata = pd.DataFrame({"ticker": dataset.prices.columns})
    securities = upsert_securities(metadata, source)
    upsert_price_frame(dataset.prices, securities)

    as_of = pd.Timestamp(dataset.prices.index.max()).date()
    checksum = price_book_checksum()
    universe, _ = Universe.objects.get_or_create(
        name="Current S&P 500",
        defaults={
            "description": "Current membership; historical use is survivorship-biased.",
            "methodology": "current_membership",
        },
    )
    universe_snapshot, _ = UniverseSnapshot.objects.update_or_create(
        universe=universe,
        as_of=as_of,
        source=source,
        defaults={
            "point_in_time": False,
            "survivorship_warning": True,
            "checksum": checksum,
            "provenance": _json_safe(dataset.provenance),
        },
    )
    universe_snapshot.securities.set(
        Security.objects.filter(
            ticker__in=[
                normalize_ticker(str(ticker)) for ticker in dataset.prices.columns
            ]
        )
    )
    return PriceSnapshot.objects.create(
        universe_snapshot=universe_snapshot,
        as_of=as_of,
        source=source,
        checksum=checksum,
        row_count=PriceBar.objects.count(),
        missing_flags={"tickers": dataset.provenance.get("missing_tickers", [])},
        anomaly_flags={
            "short_history": dataset.provenance.get("short_history_tickers", [])
        },
        provenance=_json_safe(dataset.provenance),
    )


def clear_price_dataset_cache() -> None:
    _cached_price_dataset.cache_clear()


def _price_cache_key() -> str:
    snapshot = (
        PriceSnapshot.objects.order_by("-created_at")
        .values("id", "checksum", "row_count")
        .first()
    )
    if snapshot:
        return f"{snapshot['id']}:{snapshot['checksum']}:{snapshot['row_count']}"
    count = PriceBar.objects.count()
    return f"bars:{count}" if count else ""


@lru_cache(maxsize=2)
def _cached_price_dataset(cache_key: str) -> MarketDataset:
    del cache_key
    rows = list(
        PriceBar.objects.order_by("date", "security__ticker").values(
            "date", "close", "security__ticker"
        )
    )
    if not rows:
        raise ValueError(
            "SQLite price store is empty; download or bootstrap a price dataset first"
        )
    frame = pd.DataFrame(rows)
    prices = frame.pivot(
        index="date", columns="security__ticker", values="close"
    ).sort_index()
    prices.index = pd.to_datetime(prices.index).as_unit("us")
    prices.index.name = None
    prices.columns.name = None
    tickers = list(prices.columns)
    metadata = pd.DataFrame.from_records(
        Security.objects.filter(ticker__in=tickers)
        .order_by("ticker")
        .values(
            "ticker",
            "name",
            "asset_type",
            "sector",
            "industry",
            "exchange",
            "currency",
            "current_shares",
            "market_cap",
            "metadata_as_of",
        )
    )
    snapshot = PriceSnapshot.objects.order_by("-created_at").first()
    provenance = dict(snapshot.provenance) if snapshot else {}
    if snapshot:
        provenance["sqlite_snapshot_id"] = snapshot.pk
        provenance["checksum"] = snapshot.checksum
    else:
        provenance["checksum"] = price_book_checksum()
    provenance["provider"] = "sqlite"
    provenance["source"] = snapshot.source if snapshot else "sqlite"
    return MarketDataset(prices=prices, metadata=metadata, provenance=provenance)


def load_price_dataset() -> MarketDataset:
    cache_key = _price_cache_key()
    if not cache_key:
        raise ValueError(
            "SQLite price store is empty; download or bootstrap a price dataset first"
        )
    dataset = _cached_price_dataset(cache_key)
    return MarketDataset(
        prices=dataset.prices.copy(),
        metadata=dataset.metadata.copy(),
        provenance=dict(dataset.provenance),
    )
