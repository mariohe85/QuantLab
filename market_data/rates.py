"""Point-in-time cash rates used to express expected returns in excess of cash."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .models import RiskFreeRate
from .providers import RISK_FREE_SOURCE, download_risk_free_rates


def store_risk_free_rates(rates: pd.Series, source: str = RISK_FREE_SOURCE) -> int:
    """Upsert an annualized decimal rate series keyed by observation date."""

    records = [
        RiskFreeRate(
            date=pd.Timestamp(stamp).date(),
            annualized_rate=float(rate),
            source=source,
        )
        for stamp, rate in rates.items()
    ]
    if not records:
        return 0
    RiskFreeRate.objects.bulk_create(
        records,
        update_conflicts=True,
        unique_fields=["date"],
        update_fields=["annualized_rate", "source"],
    )
    return len(records)


def refresh_risk_free_rates(start_year: int = 2015, end_year: int | None = None) -> int:
    end_year = end_year or datetime.now(UTC).year
    return store_risk_free_rates(download_risk_free_rates(start_year, end_year))


def risk_free_rate_on(as_of, default: float = 0.0) -> float:
    """Return the most recent cash rate observed on or before ``as_of``.

    Falls back to ``default`` when no rate history has been loaded, which keeps
    optimization runnable before the series is first fetched.
    """

    if as_of is None:
        return float(default)
    stamp = pd.Timestamp(as_of).date()
    observation = (
        RiskFreeRate.objects.filter(date__lte=stamp)
        .order_by("-date")
        .only("annualized_rate")
        .first()
    )
    return float(observation.annualized_rate) if observation else float(default)
