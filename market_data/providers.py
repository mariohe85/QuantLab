from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import Protocol

import numpy as np
import pandas as pd

from factors.proxyfactorlib import ALL_TICKERS as PROXY_MODEL_TICKERS

TREASURY_BILL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_bill_rates"
    "&field_tdr_date_value={year}&page&_format=csv"
)
TREASURY_USER_AGENT = "Mozilla/5.0 (QuantLab local factor research)"
RISK_FREE_COLUMN = "13 WEEKS COUPON EQUIVALENT"
RISK_FREE_SOURCE = "US Treasury 13-week bill (coupon equivalent)"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_USER_AGENT = "QuantLab/1.0 (local factor research; python-urllib)"
SURVIVORSHIP_WARNING = "Uses the current S&P 500 membership for all dates; historical results have survivorship bias."
YAHOO_WARNING = "Yahoo Finance is an unofficial source with possible revisions, gaps, and rate limits."


@dataclass(frozen=True)
class MarketDataset:
    prices: pd.DataFrame
    metadata: pd.DataFrame
    provenance: dict


class MarketDataProvider(Protocol):
    def constituents(self) -> pd.DataFrame: ...
    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


class YahooWikipediaProvider:
    def __init__(
        self,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        *,
        verify_ssl: bool = True,
    ):
        self.retries = max(1, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.verify_ssl = verify_ssl

    def _yahoo_session(self):
        if self.verify_ssl:
            return None
        from curl_cffi import requests

        return requests.Session(impersonate="chrome", verify=False)

    def _constituents_html(self) -> str:
        request = urllib.request.Request(
            WIKIPEDIA_URL, headers={"User-Agent": WIKIPEDIA_USER_AGENT}
        )
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.read().decode("utf-8")
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise RuntimeError(
            f"Wikipedia constituent download failed after {self.retries} attempts"
        ) from error

    def constituents(self) -> pd.DataFrame:
        table = pd.read_html(StringIO(self._constituents_html()))[0]
        table["Symbol"] = table["Symbol"].map(normalize_ticker)
        return table.rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})

    def security_metadata(
        self, tickers: list[str], batch_size: int = 50
    ) -> tuple[pd.DataFrame, list[str]]:
        import yfinance as yf

        rows, failures = [], []
        for start in range(0, len(tickers), batch_size):
            batch = tickers[start : start + batch_size]
            collection = yf.Tickers(" ".join(batch), session=self._yahoo_session())
            for ticker in batch:
                try:
                    info = collection.tickers[ticker].fast_info
                    shares = info.get("shares")
                    market_cap = info.get("market_cap")
                    rows.append(
                        {
                            "ticker": ticker,
                            "current_shares": float(shares) if shares else np.nan,
                            "market_cap": float(market_cap) if market_cap else np.nan,
                            "metadata_as_of": datetime.now(UTC).date().isoformat(),
                        }
                    )
                except Exception:
                    failures.append(ticker)
        return pd.DataFrame(rows), failures

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                data = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=True,
                    timeout=20,
                    session=self._yahoo_session(),
                )
                if data.empty:
                    raise ValueError("Yahoo returned no rows")
                break
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        else:
            raise RuntimeError(
                f"Yahoo download failed after {self.retries} attempts"
            ) from error
        close = (
            data["Close"]
            if isinstance(data.columns, pd.MultiIndex)
            else data[["Close"]]
        )
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        elif not isinstance(data.columns, pd.MultiIndex) and len(tickers) == 1:
            close.columns = [tickers[0]]
        close.columns = [normalize_ticker(str(c)) for c in close.columns]
        return validate_prices(close)

    def download(self, start: str, end: str, limit: int | None = None) -> MarketDataset:
        members = self.constituents()
        if limit is not None and limit > 0:
            members = members.iloc[:limit]
        members = members.copy()
        members["asset_type"] = "stock"
        required_proxies = factor_proxy_tickers()
        proxies = pd.DataFrame(
            {
                "ticker": required_proxies,
                "sector": "Factor proxy",
                "asset_type": "factor_proxy",
            }
        )
        metadata = pd.concat([members, proxies], ignore_index=True)
        metadata = metadata.drop_duplicates("ticker", keep="first")
        tickers = metadata["ticker"].tolist()
        prices = self.prices(tickers, start, end)
        downloaded = set(prices.columns)
        missing = [
            ticker
            for ticker in tickers
            if ticker not in downloaded or prices[ticker].notna().sum() < 2
        ]
        counts = prices.notna().sum()
        maximum_history = int(counts.max()) if len(counts) else 0
        short = counts[counts < max(60, int(maximum_history * 0.6))].index.tolist()
        warnings = [SURVIVORSHIP_WARNING, YAHOO_WARNING]
        if not self.verify_ssl:
            warnings.append(
                "TLS certificate verification was disabled for Yahoo because the local "
                "certificate chain could not be validated."
            )
        if missing:
            warnings.append(f"Missing tickers: {', '.join(missing)}")
        if short:
            warnings.append(f"Short-history tickers: {', '.join(short)}")
        current, metadata_failures = self.security_metadata(members["ticker"].tolist())
        metadata = metadata.merge(current, on="ticker", how="left")
        return MarketDataset(
            prices,
            metadata,
            {
                "provider": "yfinance",
                "universe_source": WIKIPEDIA_URL,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "start": start,
                "end": end,
                "requested_stock_count": int((metadata["asset_type"] == "stock").sum()),
                "requested_factor_proxies": required_proxies,
                "downloaded_tickers": sorted(downloaded),
                "fields": ["adjusted_close"],
                "missing_tickers": missing,
                "metadata_failures": metadata_failures,
                "metadata_source": "Yahoo fast_info current shares/market cap",
                "short_history_tickers": short,
                "retry_policy": {
                    "attempts": self.retries,
                    "exponential_backoff_seconds": self.backoff_seconds,
                },
                "warnings": warnings,
            },
        )


def _treasury_bill_csv(
    year: int, retries: int = 3, backoff_seconds: float = 1.0
) -> str:
    request = urllib.request.Request(
        TREASURY_BILL_URL.format(year=year),
        headers={"User-Agent": TREASURY_USER_AGENT},
    )
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8", "replace")
        except OSError as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(f"Treasury bill rate download failed for {year}") from error


def download_risk_free_rates(start_year: int, end_year: int) -> pd.Series:
    """Download daily 13-week Treasury bill yields as decimal annualized rates.

    Treasury publishes one CSV per calendar year and quotes rates in percent.
    The coupon-equivalent column is used rather than the bank-discount column so
    the rate is comparable to the equity returns it is subtracted from.
    """

    if end_year < start_year:
        raise ValueError("end_year must not precede start_year")
    frames = []
    for year in range(start_year, end_year + 1):
        table = pd.read_csv(StringIO(_treasury_bill_csv(year)))
        if RISK_FREE_COLUMN not in table.columns:
            raise ValueError(
                f"Treasury CSV for {year} has no {RISK_FREE_COLUMN} column"
            )
        frames.append(
            pd.Series(
                pd.to_numeric(table[RISK_FREE_COLUMN], errors="coerce").to_numpy(),
                index=pd.to_datetime(table["Date"], format="%m/%d/%Y"),
            )
        )
    rates = pd.concat(frames).dropna().sort_index() / 100
    rates = rates.loc[~rates.index.duplicated()]
    if rates.empty:
        raise RuntimeError("No usable Treasury bill observations were returned")
    if (rates < -0.01).any() or (rates > 0.25).any():
        raise ValueError("Treasury bill yields fall outside a plausible range")
    rates.name = "annualized_rate"
    return rates


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.sort_index().loc[~prices.index.duplicated()].astype(float)
    result = result.where(result > 0)
    if result.empty or result.notna().sum().max() < 2:
        raise ValueError("No ticker has enough valid adjusted prices")
    return result


def factor_proxy_tickers() -> list[str]:
    return sorted(set(PROXY_MODEL_TICKERS) | {"SPY"})


def synthetic_dataset(
    periods: int = 756, assets: int = 24, seed: int = 7
) -> MarketDataset:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    latent = pd.DataFrame(
        rng.normal(0.00008, 0.005, (periods, 15)),
        index=dates,
        columns=[
            "Market",
            "Oil",
            "Gold",
            "Rates",
            "USD",
            "CreditRisk",
            "Value",
            "Quality",
            "Growth",
            "DividendYield",
            "SmallSize",
            "Momentum",
            "BetaFactor",
            "Liquidity",
            "LowVolatility",
        ],
    )
    latent["Market"] = rng.normal(0.00025, 0.009, periods)
    sectors = ["Technology", "Financials", "Health Care", "Industrials"]
    returns = {}
    rows = []
    proxy_returns = {
        ticker: 0.25 * latent["Market"].values + rng.normal(0, 0.003, periods)
        for ticker in factor_proxy_tickers()
    }
    proxy_returns["SPY"] = latent["Market"].values + rng.normal(0, 0.001, periods)
    proxy_returns["IVV"] = latent["Market"].values + rng.normal(0, 0.001, periods)
    returns.update(proxy_returns)
    for ticker in factor_proxy_tickers():
        rows.append(
            {"ticker": ticker, "sector": "Factor proxy", "asset_type": "factor_proxy"}
        )
    for i in range(assets):
        ticker = f"DEMO{i + 1:02d}"
        loadings = rng.normal(0, 0.25, len(latent.columns))
        loadings[0] = 0.6 + 0.8 * rng.random()
        returns[ticker] = latent.values @ loadings + rng.normal(0, 0.009, periods)
        rows.append(
            {
                "ticker": ticker,
                "sector": sectors[i % len(sectors)],
                "market_cap": 1e9 * (i + 1),
                "current_shares": 1e7 * (i + 1),
                "industry": f"Industry {i % 6 + 1}",
                "asset_type": "stock",
            }
        )
    prices = 100 * np.exp(pd.DataFrame(returns, index=dates).cumsum())
    return MarketDataset(
        prices,
        pd.DataFrame(rows),
        {
            "provider": "synthetic",
            "seed": seed,
            "generated_at": datetime.now(UTC).isoformat(),
            "factor_proxy_tickers": factor_proxy_tickers(),
            "warnings": ["Synthetic demonstration data; not investment evidence."],
        },
    )
