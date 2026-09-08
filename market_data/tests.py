from datetime import date
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from desk.services import persist_snapshot
from desk.workflows import _dataset_for_build

from .models import PriceBar, PriceSnapshot, RiskFreeRate
from .providers import (
    MarketDataset,
    YahooWikipediaProvider,
    download_risk_free_rates,
    factor_proxy_tickers,
    normalize_ticker,
    synthetic_dataset,
)
from .rates import risk_free_rate_on, store_risk_free_rates
from .store import (
    _cached_price_dataset,
    clear_price_dataset_cache,
    load_price_dataset,
    persist_price_dataset,
    price_book_checksum,
    upsert_price_frame,
)


class ProviderTests(SimpleTestCase):
    def test_ticker_normalization(self):
        self.assertEqual(normalize_ticker(" brk.b "), "BRK-B")

    def test_synthetic_is_reproducible(self):
        first = synthetic_dataset(periods=40, assets=5, seed=2)
        second = synthetic_dataset(periods=40, assets=5, seed=2)
        pd.testing.assert_frame_equal(first.prices, second.prices)
        pd.testing.assert_frame_equal(first.metadata, second.metadata)
        self.assertEqual(set(first.metadata["asset_type"]), {"stock", "factor_proxy"})

    @patch("yfinance.download")
    def test_yahoo_adapter_uses_adjusted_download(self, download):
        dates = pd.bdate_range("2024-01-01", periods=4)
        download.return_value = pd.DataFrame({
            "Open": [10, 11, 12, 13],
            "Close": [10, 11, 12, 13],
        }, index=dates)
        result = YahooWikipediaProvider().prices(["BRK-B"], "2024-01-01", "2024-02-01")
        self.assertEqual(list(result.columns), ["BRK-B"])
        self.assertTrue(download.call_args.kwargs["auto_adjust"])

    @patch.object(YahooWikipediaProvider, "prices")
    @patch.object(YahooWikipediaProvider, "constituents")
    @patch.object(YahooWikipediaProvider, "security_metadata")
    def test_download_includes_stocks_and_all_factor_proxies(
        self, security_metadata, constituents, prices
    ):
        constituents.return_value = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "sector": ["One", "Two"],
        })
        columns = ["AAA", "BBB", *factor_proxy_tickers()]
        prices.return_value = pd.DataFrame(
            100.0, index=pd.bdate_range("2024-01-01", periods=80), columns=columns
        )
        security_metadata.return_value = (
            pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"],
                    "market_cap": [1_000_000.0, 2_000_000.0],
                }
            ),
            [],
        )
        dataset = YahooWikipediaProvider().download("2024-01-01", "2024-06-01", limit=None)
        self.assertEqual(set(dataset.prices.columns), set(columns))
        self.assertEqual(set(dataset.metadata["asset_type"]), {"stock", "factor_proxy"})
        self.assertEqual(set(dataset.provenance["requested_factor_proxies"]), set(factor_proxy_tickers()))

    @patch("yfinance.download")
    def test_yahoo_retry_is_bounded(self, download):
        dates = pd.bdate_range("2024-01-01", periods=3)
        download.side_effect = [
            RuntimeError("temporary"),
            pd.DataFrame({"Close": [10, 11, 12]}, index=dates),
        ]
        provider = YahooWikipediaProvider(retries=2, backoff_seconds=0)
        provider.prices(["AAA"], "2024-01-01", "2024-02-01")
        self.assertEqual(download.call_count, 2)


TREASURY_CSV = (
    'Date,"13 WEEKS BANK DISCOUNT","13 WEEKS COUPON EQUIVALENT"\n'
    "01/03/2023,4.40,4.52\n"
    "01/04/2023,4.42,4.54\n"
)


class RiskFreeRateProviderTests(SimpleTestCase):
    @patch("market_data.providers.urllib.request.urlopen")
    def test_treasury_download_reads_coupon_equivalent_as_decimal(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = (
            TREASURY_CSV.encode()
        )
        rates = download_risk_free_rates(2023, 2023)
        self.assertEqual(len(rates), 2)
        self.assertAlmostEqual(rates.iloc[0], 0.0452)
        self.assertEqual(rates.index[0], pd.Timestamp("2023-01-03"))

    @patch("market_data.providers.urllib.request.urlopen")
    def test_implausible_yields_are_rejected(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = (
            b'Date,"13 WEEKS COUPON EQUIVALENT"\n01/03/2023,4520\n'
        )
        with self.assertRaisesRegex(ValueError, "plausible range"):
            download_risk_free_rates(2023, 2023)

    def test_end_year_must_not_precede_start_year(self):
        with self.assertRaisesRegex(ValueError, "end_year"):
            download_risk_free_rates(2024, 2023)


class RiskFreeRateLookupTests(TestCase):
    def test_lookup_returns_latest_rate_on_or_before_the_as_of_date(self):
        stored = store_risk_free_rates(
            pd.Series(
                [0.0005, 0.0532],
                index=pd.to_datetime(["2021-06-30", "2023-06-30"]),
            )
        )
        self.assertEqual(stored, 2)
        self.assertAlmostEqual(risk_free_rate_on(date(2021, 6, 30)), 0.0005)
        self.assertAlmostEqual(risk_free_rate_on(date(2022, 12, 31)), 0.0005)
        self.assertAlmostEqual(risk_free_rate_on(date(2026, 1, 1)), 0.0532)

    def test_lookup_falls_back_before_history_and_when_empty(self):
        self.assertAlmostEqual(risk_free_rate_on(date(2024, 1, 1), default=0.04), 0.04)
        store_risk_free_rates(
            pd.Series([0.0532], index=pd.to_datetime(["2023-06-30"]))
        )
        self.assertAlmostEqual(risk_free_rate_on(date(2020, 1, 1), default=0.04), 0.04)

    def test_storing_the_same_date_twice_updates_rather_than_duplicates(self):
        index = pd.to_datetime(["2023-06-30"])
        store_risk_free_rates(pd.Series([0.05], index=index))
        store_risk_free_rates(pd.Series([0.0532], index=index))
        self.assertEqual(RiskFreeRate.objects.count(), 1)
        self.assertAlmostEqual(risk_free_rate_on(date(2023, 6, 30)), 0.0532)


class SQLitePriceStoreTests(TestCase):
    def test_persist_is_sql_only_and_round_trips(self):
        dataset = synthetic_dataset(periods=40, assets=5, seed=4)
        record = persist_snapshot(dataset, "test")
        self.assertEqual(record.source, "test")
        self.assertEqual(PriceSnapshot.objects.count(), 1)
        replayed = load_price_dataset()
        pd.testing.assert_frame_equal(
            dataset.prices.sort_index(axis=1),
            replayed.prices.sort_index(axis=1),
            check_freq=False,
        )

    def test_second_persist_appends_and_overwrites(self):
        metadata = pd.DataFrame(
            [{"ticker": "AAA", "asset_type": "stock", "sector": "Test"}]
        )
        first = MarketDataset(
            pd.DataFrame(
                {"AAA": [10.0, 11.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            ),
            metadata,
            {"provider": "test"},
        )
        second = MarketDataset(
            pd.DataFrame(
                {"AAA": [12.0, 13.0]},
                index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
            ),
            metadata,
            {"provider": "test"},
        )
        persist_price_dataset(first, "test")
        persist_price_dataset(second, "test")
        prices = load_price_dataset().prices
        self.assertEqual(len(prices), 3)
        self.assertEqual(prices.loc["2024-01-03", "AAA"], 12.0)

    def test_checksum_changes_when_close_changes(self):
        persist_price_dataset(
            synthetic_dataset(periods=20, assets=2, seed=9), "test"
        )
        before = price_book_checksum()
        prices = load_price_dataset().prices.iloc[[-1], [0]].copy()
        prices.iloc[0, 0] += 1
        upsert_price_frame(prices)
        self.assertNotEqual(before, price_book_checksum())

    def test_price_dataset_is_cached_until_prices_change(self):
        persist_price_dataset(synthetic_dataset(periods=20, assets=2, seed=11), "test")
        clear_price_dataset_cache()
        first = load_price_dataset()
        second = load_price_dataset()
        self.assertEqual(_cached_price_dataset.cache_info().hits, 1)
        first.prices.iloc[0, 0] = -999.0
        self.assertNotEqual(load_price_dataset().prices.iloc[0, 0], -999.0)
        mutated = second.prices.iloc[[-1], [0]].copy()
        mutated.iloc[0, 0] += 1
        upsert_price_frame(mutated)
        refreshed = load_price_dataset()
        self.assertEqual(
            refreshed.prices.loc[mutated.index[0], mutated.columns[0]],
            mutated.iloc[0, 0],
        )

    def test_dataset_for_build_reads_sql_prices(self):
        dataset = synthetic_dataset(periods=20, assets=2, seed=10)
        snapshot = persist_price_dataset(dataset, "test")
        loaded = _dataset_for_build(type("Build", (), {"price_snapshot_id": snapshot.pk})())
        pd.testing.assert_frame_equal(
            dataset.prices.sort_index(axis=1),
            loaded.prices.sort_index(axis=1),
            check_freq=False,
        )
