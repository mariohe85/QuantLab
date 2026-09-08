from __future__ import annotations

import csv
import io

from django.db import transaction

from market_data.models import Security
from market_data.store import load_price_dataset

from .models import Portfolio, PortfolioHolding, ScreenRun


def ensure_price_securities(tickers: list[str]) -> dict[str, Security]:
    """Return securities backed by the latest price snapshot, creating missing rows."""
    normalized = [ticker.strip().upper().replace(".", "-") for ticker in tickers]
    known = {
        item.ticker: item for item in Security.objects.filter(ticker__in=normalized)
    }
    missing = sorted(set(normalized) - set(known))
    if not missing:
        return known
    available = set(load_price_dataset().prices.columns)
    unavailable = sorted(set(normalized) - available)
    if unavailable:
        raise ValueError(f"No SQLite prices for: {', '.join(unavailable)}")
    if missing:
        accepted = set(missing) & available
        Security.objects.bulk_create(
            [
                Security(
                    ticker=ticker,
                    asset_type="stock",
                    provenance={
                        "warning": (
                            "Not in current S&P 500; accepted because snapshot prices exist"
                        )
                    },
                )
                for ticker in accepted
            ],
            ignore_conflicts=True,
        )
        unavailable = set(missing) - accepted
        if unavailable:
            raise ValueError(
                f"No SQLite prices for: {', '.join(sorted(unavailable))}"
            )
        known.update(
            {item.ticker: item for item in Security.objects.filter(ticker__in=accepted)}
        )
    return known


def parse_holdings_csv(content: str) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(content.strip())):
        ticker = (
            (row.get("ticker") or row.get("symbol") or "")
            .strip()
            .upper()
            .replace(".", "-")
        )
        if not ticker:
            continue
        weight = row.get("weight")
        shares = row.get("shares")
        if not weight and not shares:
            raise ValueError(f"{ticker}: provide weight or shares")
        rows.append(
            {
                "ticker": ticker,
                "weight": float(weight) if weight else None,
                "shares": float(shares) if shares else None,
            }
        )
    return rows


def validate_holdings(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("At least one holding is required")
    tickers = [row["ticker"].strip().upper().replace(".", "-") for row in rows]
    if len(tickers) != len(set(tickers)):
        raise ValueError("Duplicate tickers are not allowed")
    ensure_price_securities(tickers)
    normalized = [{**row, "ticker": ticker} for row, ticker in zip(rows, tickers)]
    weights = [row.get("weight") for row in normalized]
    if all(value is not None for value in weights):
        if any(float(value) < 0 for value in weights):
            raise ValueError("Weights must be non-negative")
        total = sum(float(value) for value in weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        for row in normalized:
            row["weight"] = float(row["weight"]) / total
    return normalized


@transaction.atomic
def save_portfolio(
    name: str,
    rows: list[dict],
    *,
    portfolio: Portfolio | None = None,
    source: str = "manual",
    equal_weight: bool = False,
) -> Portfolio:
    rows = validate_holdings(rows)
    if equal_weight:
        for row in rows:
            row["weight"] = 1 / len(rows)
            row["shares"] = None
    portfolio = portfolio or Portfolio.objects.create(name=name, source=source)
    if portfolio.name != name:
        portfolio.name = name
        portfolio.save(update_fields=["name", "updated_at"])
    portfolio.holdings.all().delete()
    securities = {
        item.ticker: item
        for item in Security.objects.filter(ticker__in=[r["ticker"] for r in rows])
    }
    PortfolioHolding.objects.bulk_create(
        [
            PortfolioHolding(
                portfolio=portfolio,
                security=securities[row["ticker"]],
                weight=row.get("weight"),
                shares=row.get("shares"),
            )
            for row in rows
        ]
    )
    warnings = [
        f"{item.ticker} is outside the current S&P 500 universe"
        for item in securities.values()
        if item.provenance.get("warning")
    ]
    portfolio.configuration = {**portfolio.configuration, "warnings": warnings}
    portfolio.save(update_fields=["configuration", "updated_at"])
    return portfolio


def clone_portfolio(portfolio: Portfolio, name: str) -> Portfolio:
    return save_portfolio(
        name,
        [
            {
                "ticker": item.security.ticker,
                "weight": item.weight,
                "shares": item.shares,
            }
            for item in portfolio.holdings.select_related("security")
        ],
        source=portfolio.source,
    )


def portfolio_from_screen(run: ScreenRun, name: str, top_n: int) -> Portfolio:
    if not 10 <= top_n <= 50:
        raise ValueError("Screen portfolios require top_n between 10 and 50")
    results = run.results.filter(passed=True).order_by("rank")[:top_n]
    return save_portfolio(
        name,
        [
            {"ticker": item.security.ticker, "weight": 1 / len(results)}
            for item in results
        ],
        source="screen",
    )


def save_selection_portfolio(name: str, preview: dict) -> Portfolio:
    rows = [
        {"ticker": row["ticker"], "weight": row["portfolio_weight"]}
        for row in preview["included_rows"]
    ]
    if not rows:
        raise ValueError("The stock selection contains no holdings")
    return save_portfolio(
        name,
        rows,
        source="screen",
        equal_weight=True,
    )
