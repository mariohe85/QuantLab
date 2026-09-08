from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase

from backtests.factor_strategy import monthly_event_backtest
from desk.workflows import run_backtest, run_proxy_factor_build
from factors.catalog import PROXY_MODEL_VERSION
from factors.models import FactorBuild
from jobs.models import Job
from jobs.runner import _handler, claim_job, request_cancellation, submit_job
from market_data.models import Security
from market_data.providers import YahooWikipediaProvider
from selection.engine import composite_scores, preview_screen
from selection.models import Portfolio, PortfolioHolding


class NumericalAuditTests(SimpleTestCase):
    def test_monthly_engine_rebalances_next_day_and_drifts(self):
        index = pd.bdate_range("2022-01-01", periods=100)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 160, len(index)), "B": 100.0}, index=index
        )
        result = monthly_event_backtest(
            prices,
            lambda history: pd.Series({"A": 0.5, "B": 0.5}),
            estimation_window=20,
            transaction_cost_bps=25,
        )
        trade_dates = result.turnover[result.turnover > 0].index
        self.assertGreaterEqual(len(trade_dates), 2)
        self.assertTrue(
            all(
                index[index.get_loc(stamp) - 1].month != stamp.month
                for stamp in trade_dates
            )
        )
        between = result.weights.loc[trade_dates[0] : trade_dates[1], "A"]
        self.assertGreater(between.max(), between.min())

    def test_weighted_screen_features_cannot_silently_disappear(self):
        features = pd.DataFrame({"Market": [0.1, 0.2]}, index=["A", "B"])
        with self.assertRaisesRegex(ValueError, "Momentum"):
            composite_scores(features, {"Market": 0.5, "Momentum": 0.5})
        with self.assertRaisesRegex(ValueError, "Momentum"):
            preview_screen(
                features,
                {"Market": 0.5, "Momentum": 0.5},
                {},
                pd.Series({"A": "Tech", "B": "Tech"}),
            )

    def test_yahoo_dataset_contains_adjusted_closes_and_metadata(self):
        provider = YahooWikipediaProvider()
        index = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame({"AAA": [10, 11, 12, 13]}, index=index)
        with (
            patch.object(
                provider,
                "constituents",
                return_value=pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]}),
            ),
            patch.object(provider, "prices", return_value=prices),
            patch.object(
                provider,
                "security_metadata",
                return_value=(
                    pd.DataFrame({"ticker": ["AAA"], "market_cap": [1_000_000]}),
                    [],
                ),
            ),
        ):
            dataset = provider.download("2024-01-01", "2024-02-01")
        pd.testing.assert_frame_equal(dataset.prices, prices, check_dtype=False)
        self.assertEqual(dataset.metadata.loc[0, "market_cap"], 1_000_000)
        self.assertEqual(dataset.provenance["fields"], ["adjusted_close"])


class V2WiringAuditTests(TestCase):
    def test_ui_job_allowlist_is_v2_only(self):
        allowed = {
            "monthly_exposures",
            "screen_run",
            "optimization",
            "risk_refresh",
            "proxy_factor_build",
            "canonical_update",
            "factor_catalog_sync",
        }
        for kind in allowed:
            response = self.client.post(f"/jobs/launch/{kind}/", {})
            self.assertNotEqual(response.status_code, 404, kind)
        for removed in (
            "factor_build",
            "factorstoday_build",
            "factor_reconciliation",
        ):
            response = self.client.post(f"/jobs/launch/{removed}/", {})
            self.assertEqual(response.status_code, 404, removed)
            self.assertFalse(Job.objects.filter(kind=removed).exists())

    def test_risk_refresh_web_launch_preserves_portfolio_and_build_ids(self):
        response = self.client.post(
            "/jobs/launch/risk_refresh/",
            {"portfolio_id": "17", "factor_build_id": "23"},
        )

        self.assertEqual(response.status_code, 200)
        job = Job.objects.get(kind="risk_refresh")
        self.assertEqual(
            job.parameters,
            {"portfolio_id": 17, "factor_build_id": 23},
        )

    def test_job_runner_has_no_removed_factor_handlers(self):
        for removed in (
            "factor_build",
            "factorstoday_build",
            "factor_reconciliation",
        ):
            with self.assertRaisesRegex(ValueError, "Unknown durable job kind"):
                _handler(removed)

    def test_proxy_build_rejects_non_v2_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "Unsupported factor model version"):
            run_proxy_factor_build({"model_version": "legacy"}, lambda *_: None)

    def test_backtest_rejects_non_v2_build(self):
        build = FactorBuild.objects.create(
            as_of="2026-09-04",
            definition_version="legacy",
            status="succeeded",
        )
        with self.assertRaisesRegex(ValueError, PROXY_MODEL_VERSION):
            run_backtest({"factor_build_id": build.pk}, lambda *_: None)

    @patch(
        "desk.workflows.monthly_event_backtest",
        side_effect=RuntimeError("stop after marker"),
    )
    def test_backtest_accepts_v2_and_marks_current_universe_approximation(self, _):
        build = FactorBuild.objects.create(
            as_of="2026-09-04",
            definition_version=PROXY_MODEL_VERSION,
            status="succeeded",
        )
        security = Security.objects.create(ticker="A", asset_type="stock")
        portfolio = Portfolio.objects.create(name="V2 backtest")
        PortfolioHolding.objects.create(
            portfolio=portfolio, security=security, weight=1
        )
        dates = pd.bdate_range("2024-01-01", periods=40)
        dataset = SimpleNamespace(
            prices=pd.DataFrame({"A": np.linspace(100, 110, 40)}, index=dates),
            metadata=pd.DataFrame(
                [{"ticker": "A", "asset_type": "stock", "sector": "Test"}]
            ),
        )
        with (
            patch("desk.workflows._dataset_for_build", return_value=dataset),
            self.assertRaisesRegex(RuntimeError, "stop after marker"),
        ):
            run_backtest(
                {
                    "factor_build_id": build.pk,
                    "portfolio_id": portfolio.pk,
                    "estimation_window": 20,
                },
                lambda *_: None,
            )
        run = portfolio.backtests.get()
        self.assertEqual(run.methodology_version, PROXY_MODEL_VERSION)
        self.assertEqual(
            run.configuration["history_methodology"],
            "approximate_current_universe",
        )
        self.assertFalse(run.configuration["point_in_time_universe"])
        self.assertTrue(run.configuration["survivorship_biased"])

    def test_generic_jobs_still_claim_and_cancel(self):
        first = submit_job("screen_run", {})
        second = submit_job("monthly_exposures", {})
        self.assertEqual(claim_job("one").pk, first.pk)
        request_cancellation(first)
        first.refresh_from_db()
        self.assertEqual(first.status, "cancelled")
        self.assertEqual(claim_job("two").pk, second.pk)
