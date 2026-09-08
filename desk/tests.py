import math
import os
import re
from datetime import date
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.core.management import call_command
from django.core.management.base import CommandError
from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase

from backtests.models import BacktestRun
from factors.catalog import PROXY_MODEL_VERSION
from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorModelCatalog,
    FactorModelCatalogMembership,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)
from jobs.models import Job
from market_data.models import DataSnapshot, PriceBar, PriceSnapshot, Security
from market_data.providers import synthetic_dataset
from market_data.rates import store_risk_free_rates
from optimization.models import OptimizationScenario, OptimizationStudy
from selection.models import (
    Portfolio,
    PortfolioHolding,
    ScreenDefinition,
)

from .templatetags.optimizer_display import sharpe_ratio, variance_volatility
from .views import _default_selection_config, _refresh_portfolio_risk
from .workflows import (
    _exposure_month_ends,
    _exposure_universe,
    _exposure_worker_count,
    _forecast_warnings,
    _optimization_inputs,
    run_optimization,
    run_scenario_backtest,
)


class DashboardIntegrationTests(TestCase):
    def test_dashboard_smoke(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Factor return monitor")
        # Factor datasets are rebuilt from the backend, not from this page.
        self.assertNotContains(response, "/jobs/launch/proxy_factor_build/")
        self.assertNotContains(response, "/jobs/launch/factorstoday_build/")
        self.assertNotContains(response, ">Data</a>")
        self.assertNotContains(response, ">Risk</a>")
        self.assertNotContains(response, ">Backtests</a>")
        self.assertNotContains(response, ">Runs</a>")
        content = response.content.decode()
        nav_labels = (
            ">Factors</a>",
            ">Signals</a>",
            ">Stock</a>",
            ">Stock Selection</a>",
            ">Portfolio</a>",
            ">Optimizer</a>",
        )
        positions = [content.index(label) for label in nav_labels]
        self.assertEqual(positions, sorted(positions))

    def test_legacy_demo_endpoint_is_not_served(self):
        response = self.client.post("/run/demo/")
        self.assertEqual(response.status_code, 404)

    def test_retired_factor_job_endpoints_are_not_served(self):
        for kind in (
            "factor_build",
            "factorstoday_build",
            "factor_reconciliation",
        ):
            response = self.client.post(f"/jobs/launch/{kind}/")
            self.assertEqual(response.status_code, 404, kind)
            self.assertFalse(Job.objects.filter(kind=kind).exists())

    def test_retired_ops_pages_are_not_served(self):
        for path in (
            "/data/",
            "/risk/",
            "/backtests/",
            "/runs/",
            "/run/live/",
            "/export/backtest/1/returns.csv",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)

    def test_legacy_snapshot_analysis_is_not_served(self):
        snapshot = DataSnapshot.objects.create(
            source="test", as_of="2024-01-01"
        )
        response = self.client.post(f"/snapshots/{snapshot.pk}/analyze/")
        self.assertEqual(response.status_code, 404)

    def test_portfolio_detail_workspace_route(self):
        portfolio = Portfolio.objects.create(name="My Portfolio")
        security = Security.objects.create(ticker="XYZ")
        PortfolioHolding.objects.create(
            portfolio=portfolio, security=security, weight=1.0
        )
        response = self.client.get(f"/portfolios/{portfolio.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Holdings decomposition")


class CollapsibleSectionTests(TestCase):
    def assertBalancedDisclosures(self, html, label):
        self.assertEqual(html.count("<details"), html.count("</details>"), label)
        self.assertEqual(html.count("<summary"), html.count("</summary>"), label)

    def test_no_workspace_tab_renders_collapse_all_or_build_context(self):
        for page in (
            "signals",
            "stock",
            "stock-selection",
            "portfolio",
            "optimizer",
            "factors",
        ):
            response = self.client.get(f"/{page}/")
            self.assertEqual(response.status_code, 200, page)
            html = response.content.decode()
            self.assertNotIn("data-collapse-all", html, page)
            self.assertNotIn("build-context", html, page)
            self.assertBalancedDisclosures(html, page)

    def test_factor_universe_drops_the_build_controls(self):
        response = self.client.get("/factors/")
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertNotIn("Model version", html)
        self.assertNotIn("Update factor dataset", html)
        self.assertBalancedDisclosures(html, "factors")

    def test_stock_page_drops_the_build_picker(self):
        # One canonical build means there is nothing to switch between; the
        # page falls back to that build on its own.
        response = self.client.get("/stock/")
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertNotIn("Model version", html)
        self.assertNotIn("Change build", html)
        self.assertBalancedDisclosures(html, "stock")

    def test_portfolio_workspace_sections_declare_collapse_keys(self):
        portfolio = Portfolio.objects.create(name="Collapsible Portfolio")
        security = Security.objects.create(ticker="XYZ")
        PortfolioHolding.objects.create(
            portfolio=portfolio, security=security, weight=1.0
        )
        html = self.client.get(f"/portfolios/{portfolio.pk}/").content.decode()
        for key in (
            "portfolio-detail-run-analysis",
            "portfolio-detail-risk-snapshot",
            "portfolio-detail-holdings",
            "portfolio-detail-optimization-history",
        ):
            self.assertIn(f'data-collapse-key="{key}"', html, key)
        self.assertBalancedDisclosures(html, "portfolio detail")

    def test_history_panels_keep_separate_state_per_workspace(self):
        history = {
            "start": date(2024, 1, 2),
            "end": date(2024, 3, 1),
            "observations": 42,
            "rebalances": [],
            "chart_rows": [],
            "metric_rows": [],
            "warnings": [],
            "methodology": "Approximate replay.",
        }
        rendered = {
            prefix: render_to_string(
                "desk/partials/history_performance.html",
                {"history": history, "collapse_prefix": prefix},
            )
            for prefix in ("selection-history", "portfolio-history")
        }
        for prefix, html in rendered.items():
            for section in ("equity", "drawdown", "statistics", "rebalances"):
                self.assertIn(f'data-collapse-key="{prefix}-{section}"', html)
            self.assertBalancedDisclosures(html, prefix)
        self.assertNotIn("selection-history", rendered["portfolio-history"])
        self.assertNotIn("portfolio-history", rendered["selection-history"])

    def test_long_history_rebalance_list_starts_collapsed(self):
        html = render_to_string(
            "desk/partials/history_performance.html",
            {
                "history": {
                    "start": date(2024, 1, 2),
                    "end": date(2024, 3, 1),
                    "observations": 42,
                    "rebalances": [],
                    "chart_rows": [],
                    "metric_rows": [],
                    "warnings": [],
                    "methodology": "Approximate replay.",
                },
            },
        )
        rebalances = html.split('data-collapse-key="history-rebalances"')[1]
        self.assertNotIn("open", rebalances.split(">")[0])
        statistics = html.split('data-collapse-key="history-statistics"')[1]
        self.assertIn("open", statistics.split(">")[0])


class ReplicateResearchCommandTests(TestCase):
    def test_refuses_to_overwrite_existing_research(self):
        security = Security.objects.create(ticker="EXISTING")
        PriceBar.objects.create(
            security=security, date=date(2024, 1, 2), close=100
        )
        with self.assertRaisesMessage(CommandError, "Research tables are not empty"):
            call_command("replicate_research", skip_rates=True)

    @patch("desk.management.commands.replicate_research.build_monthly_exposures")
    @patch("desk.management.commands.replicate_research.run_proxy_factor_build")
    @patch("desk.management.commands.replicate_research.YahooWikipediaProvider")
    def test_runs_stages_with_generated_ids(self, provider, factor_build, exposures):
        provider.return_value.download.return_value = synthetic_dataset(
            periods=20, assets=2, seed=12
        )
        factor_build.return_value = {
            "factor_build_id": 37,
            "factor_count": 56,
            "observations": 100,
        }
        exposures.return_value = {
            "model_fits": 80,
            "tickers": ["S000", "S001"],
            "months": 10,
        }

        call_command(
            "replicate_research",
            start="2024-01-01",
            end="2024-02-01",
            months=10,
            workers=3,
            skip_rates=True,
            verbosity=0,
        )

        snapshot = PriceSnapshot.objects.get()
        factor_build.assert_called_once()
        self.assertEqual(
            factor_build.call_args.args[0]["price_snapshot_id"], snapshot.pk
        )
        parameters = exposures.call_args.args[0]
        self.assertEqual(parameters["factor_build_id"], 37)
        self.assertEqual(parameters["max_months"], 10)
        self.assertEqual(parameters["workers"], 3)
        self.assertEqual(
            parameters["levels"],
            ["base", "base_sector", "base_sector_industry", "all_factors"],
        )


class CanonicalDatasetConsolidationTests(TestCase):
    def test_current_period_update_replaces_prior_provisional_date(self):
        build = FactorBuild.objects.create(
            as_of="2026-09-03",
            status="succeeded",
        )
        security = Security.objects.create(ticker="CURRENT")
        defaults = {
            "alpha": 0,
            "adjusted_r2": 0.5,
            "residual_volatility": 0.2,
            "active_factor_count": 1,
            "design_factor_count": 1,
            "observation_count": 252,
            "coverage": 1,
            "is_provisional": True,
        }
        first, _ = StockModelFit.objects.update_or_create(
            build=build,
            security=security,
            period=date(2026, 9, 1),
            model_level="base",
            defaults={**defaults, "as_of": date(2026, 9, 2)},
        )
        latest, created = StockModelFit.objects.update_or_create(
            build=build,
            security=security,
            period=date(2026, 9, 1),
            model_level="base",
            defaults={**defaults, "as_of": date(2026, 9, 3)},
        )
        self.assertFalse(created)
        self.assertEqual(latest.pk, first.pk)
        self.assertEqual(latest.as_of, date(2026, 9, 3))
        self.assertEqual(
            StockModelFit.objects.filter(
                build=build,
                security=security,
                period=date(2026, 9, 1),
                model_level="base",
            ).count(),
            1,
        )

    def test_merges_fit_history_and_repoints_consumers(self):
        target = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            mode="replication",
            definition_version=PROXY_MODEL_VERSION,
        )
        source = FactorBuild.objects.create(
            as_of="2026-09-03",
            status="succeeded",
            mode="replication",
            definition_version=PROXY_MODEL_VERSION,
        )
        security = Security.objects.create(ticker="MERGE", name="Merge Corp")
        fit = StockModelFit.objects.create(
            build=source,
            security=security,
            as_of="2026-09-03",
            model_level="base",
            alpha=0.01,
            adjusted_r2=0.7,
            residual_volatility=0.2,
            active_factor_count=0,
            design_factor_count=1,
            observation_count=252,
            coverage=1,
        )
        portfolio = Portfolio.objects.create(
            name="Merge Portfolio",
            configuration={"stock_selection": {"factor_build_id": source.pk}},
        )
        screen = ScreenDefinition.objects.create(
            name="Merge Screen",
            factor_build=source,
            model_level="base",
        )

        call_command(
            "consolidate_factor_datasets",
            canonical=target.pk,
            merge=[source.pk],
            stdout=StringIO(),
        )

        source.refresh_from_db()
        portfolio.refresh_from_db()
        screen.refresh_from_db()
        merged = StockModelFit.objects.get(
            build=target,
            security=security,
            period=fit.period,
            model_level="base",
        )
        self.assertEqual(merged.as_of.isoformat(), "2026-09-03")
        self.assertFalse(source.is_canonical)
        self.assertEqual(source.superseded_by_id, target.pk)
        self.assertEqual(screen.factor_build_id, target.pk)
        self.assertEqual(
            portfolio.configuration["stock_selection"]["factor_build_id"],
            target.pk,
        )


class CleanV2OnlyCommandTests(TestCase):
    def test_refuses_destructive_cleanup_without_a_v2_build(self):
        legacy = FactorBuild.objects.create(
            as_of="2026-09-01",
            definition_version="legacy",
            status="succeeded",
        )
        portfolio = Portfolio.objects.create(name="Keep on refusal")

        with self.assertRaisesRegex(CommandError, "refusing"):
            call_command("clean_v2_only", yes=True, stdout=StringIO())

        self.assertTrue(FactorBuild.objects.filter(pk=legacy.pk).exists())
        self.assertTrue(Portfolio.objects.filter(pk=portfolio.pk).exists())

    def test_preserves_v2_build_fits_and_backtest_but_removes_consumer_state(self):
        v2 = FactorBuild.objects.create(
            as_of="2026-09-04",
            definition_version=PROXY_MODEL_VERSION,
            status="succeeded",
        )
        legacy = FactorBuild.objects.create(
            as_of="2026-09-03",
            definition_version="legacy",
            status="succeeded",
        )
        security = Security.objects.create(ticker="CLEAN")
        fit = StockModelFit.objects.create(
            build=v2,
            security=security,
            as_of=v2.as_of,
            model_level="base",
            alpha=0,
            adjusted_r2=0.5,
            residual_volatility=0.2,
            active_factor_count=1,
            design_factor_count=1,
            observation_count=252,
            coverage=1,
        )
        portfolio = Portfolio.objects.create(name="Remove consumer")
        screen = ScreenDefinition.objects.create(name="Remove screen", factor_build=v2)
        study = OptimizationStudy.objects.create(
            portfolio=portfolio,
            factor_build=v2,
            name="Remove study",
        )
        OptimizationScenario.objects.create(name="Remove scenario", study=study)
        keep_backtest = BacktestRun.objects.create(
            name="V2 backtest",
            configuration={"factor_build_id": v2.pk},
            portfolio=portfolio,
            status="succeeded",
        )
        remove_backtest = BacktestRun.objects.create(
            name="Legacy backtest",
            configuration={"factor_build_id": legacy.pk},
            status="succeeded",
        )
        keep_job = Job.objects.create(
            kind="canonical_update",
            parameters={"model_version": PROXY_MODEL_VERSION},
        )
        remove_job = Job.objects.create(kind="screen_run")

        call_command("clean_v2_only", yes=True, stdout=StringIO())

        self.assertTrue(FactorBuild.objects.filter(pk=v2.pk).exists())
        self.assertTrue(StockModelFit.objects.filter(pk=fit.pk).exists())
        self.assertFalse(FactorBuild.objects.filter(pk=legacy.pk).exists())
        self.assertTrue(BacktestRun.objects.filter(pk=keep_backtest.pk).exists())
        self.assertFalse(BacktestRun.objects.filter(pk=remove_backtest.pk).exists())
        keep_backtest.refresh_from_db()
        self.assertIsNone(keep_backtest.portfolio_id)
        self.assertFalse(Portfolio.objects.filter(pk=portfolio.pk).exists())
        self.assertFalse(ScreenDefinition.objects.filter(pk=screen.pk).exists())
        self.assertFalse(OptimizationStudy.objects.filter(pk=study.pk).exists())
        self.assertTrue(Job.objects.filter(pk=keep_job.pk).exists())
        self.assertFalse(Job.objects.filter(pk=remove_job.pk).exists())


class FactorWorkspaceUITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.build = FactorBuild.objects.create(
            as_of="2026-08-31",
            status="succeeded",
        )
        cls.market = FactorDefinition.objects.create(
            name="Market",
            family="Core",
            provenance_badge="exact_etf",
            sort_order=1,
        )
        cls.value = FactorDefinition.objects.create(
            name="Value",
            family="Fundamental",
            provenance_badge="public_approximation",
            sort_order=2,
        )
        for factor, value in ((cls.market, 0.01), (cls.value, -0.02)):
            FactorObservation.objects.create(
                build=cls.build,
                factor=factor,
                date=cls.build.as_of,
                scaled_return=value,
                cumulative_index=1.0 + value,
                horizons={"21": value},
                zscores={"21": value * 10},
                coverage=0.9,
            )
        cls.security = Security.objects.create(ticker="XYZ", name="Example Corp")
        cls.fit = StockModelFit.objects.create(
            build=cls.build,
            security=cls.security,
            as_of=cls.build.as_of,
            model_level="base",
            alpha=0.001,
            adjusted_r2=0.62,
            residual_volatility=0.18,
            active_factor_count=1,
            design_factor_count=2,
            observation_count=252,
            coverage=0.96,
        )
        StockExposureSnapshot.objects.create(
            model_fit=cls.fit,
            factor=cls.market,
            beta=1.1,
            standard_error=0.1,
            t_stat=11.0,
            p_value=0.001,
            confidence_low=0.9,
            confidence_high=1.3,
        )
        cls.portfolio = Portfolio.objects.create(name="Factor UI Portfolio")
        PortfolioHolding.objects.create(
            portfolio=cls.portfolio,
            security=cls.security,
            weight=1.0,
        )
        cls.base_catalog = FactorModelCatalog.objects.create(
            slug="base",
            name="Base",
            available_factor_count=2,
        )
        for position, factor in enumerate((cls.market, cls.value)):
            FactorModelCatalogMembership.objects.create(
                catalog=cls.base_catalog, factor=factor, position=position
            )
        cls.unmodeled = Security.objects.create(ticker="NOFIT", name="Unmodeled Corp")
        cls.mixed_portfolio = Portfolio.objects.create(name="Mixed Coverage")
        for security in (cls.security, cls.unmodeled):
            PortfolioHolding.objects.create(
                portfolio=cls.mixed_portfolio,
                security=security,
                weight=0.5,
            )
        cls.empty_build = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            mode="replication",
        )

    def test_universe_filters_and_factor_dialog_are_rendered(self):
        response = self.client.get(
            "/factors/",
            {
                "tab": "universe",
                "build_id": self.build.id,
                "family": "Core",
                "factor": "Market",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-factor-dialog")
        self.assertContains(
            response,
            f"/factors/builds/{self.build.id}/{self.market.id}/",
        )
        self.assertContains(response, '<option value="monthly" selected>')
        self.assertNotContains(response, "Exact ETF construction")
        self.assertNotContains(response, "data-factor-provenance")
        self.assertNotContains(response, "History factor")
        self.assertNotContains(response, 'data-chart="factor-history"')
        self.assertNotContains(response, 'data-factor-name="Value"')

    def test_latest_panel_pairs_each_zscore_window_with_its_percent_return(self):
        response = self.client.get(
            "/factors/",
            {"tab": "universe", "build_id": self.build.id, "family": "Core"},
        )
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        for window in (1, 5, 21, 63, 252):
            self.assertIn(f"<th>{window}d return</th>", html)
            self.assertIn(f"<th>{window}d z</th>", html)
        # The one-day return is the stored daily return, shown as a percentage.
        self.assertIn("<td>1.00%</td>", html)
        self.assertIn('data-zscore="0.1"', html)
        # Windows without stored statistics stay blank rather than rendering zero.
        self.assertIn('data-zscore=""', html)

    def test_factor_return_detail_compounds_daily_weekly_and_monthly_series(self):
        FactorObservation.objects.create(
            build=self.build,
            factor=self.market,
            date="2026-07-31",
            scaled_return=0.10,
            cumulative_index=1.10,
        )
        FactorObservation.objects.create(
            build=self.build,
            factor=self.market,
            date="2026-08-03",
            scaled_return=0.10,
            cumulative_index=1.21,
        )
        FactorObservation.objects.create(
            build=self.build,
            factor=self.market,
            date="2026-08-04",
            scaled_return=-0.10,
            cumulative_index=1.089,
        )

        response = self.client.get(f"/factors/builds/{self.build.id}/{self.market.id}/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["factor"]["name"], "Market")
        self.assertEqual(payload["factor"]["construction"]["kind"], "direct_etf")
        self.assertEqual(payload["factor"]["construction"]["long_leg"], "IVV")
        self.assertEqual(len(payload["series"]["daily"]), 4)
        self.assertEqual(len(payload["series"]["weekly"]), 3)
        self.assertEqual(len(payload["series"]["monthly"]), 2)
        self.assertAlmostEqual(
            payload["series"]["weekly"][1]["return"],
            (1.10 * 0.90) - 1,
        )
        self.assertAlmostEqual(
            payload["series"]["monthly"][1]["return"],
            (1.10 * 0.90 * 1.01) - 1,
        )
        self.assertEqual(payload["series"]["monthly"][1]["cumulative_index"], 1.01)

    def test_factor_return_detail_returns_json_errors(self):
        missing_build = self.client.get(f"/factors/builds/999999/{self.market.id}/")
        self.assertEqual(missing_build.status_code, 404)
        self.assertEqual(missing_build.json()["error"], "Factor build not found.")

        missing_factor = self.client.get(f"/factors/builds/{self.build.id}/999999/")
        self.assertEqual(missing_factor.status_code, 404)
        self.assertEqual(missing_factor.json()["error"], "Factor not found.")

    def test_stale_family_still_appears_in_latest_panel(self):
        lagging = FactorDefinition.objects.create(
            name="Country: Japan",
            family="country",
            provenance_badge="exact_etf",
            sort_order=3,
        )
        FactorObservation.objects.create(
            build=self.build,
            factor=lagging,
            date="2026-08-28",
            scaled_return=0.003,
            cumulative_index=1.003,
            horizons={"21": 0.003},
            zscores={"21": 0.4},
            coverage=1.0,
        )
        unfiltered = self.client.get(
            "/factors/", {"tab": "universe", "build_id": self.build.id}
        )
        self.assertContains(unfiltered, "Country: Japan")
        self.assertContains(unfiltered, "Aug. 28, 2026")

        filtered = self.client.get(
            "/factors/",
            {"tab": "universe", "build_id": self.build.id, "family": "country"},
        )
        self.assertContains(filtered, "Country: Japan")
        self.assertNotContains(filtered, ">Market</a>")
        self.assertEqual(len(filtered.context["factor_panel_rows"]), 1)

    def test_stock_models_degrade_without_silent_fallback(self):
        response = self.client.get(
            "/stock/",
            {
                "build_id": self.build.id,
                "ticker": "XYZ",
                "model_level": "base",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Example Corp")
        self.assertContains(response, "Adjusted R²")
        self.assertContains(response, "HAC inference")
        self.assertContains(response, "Base + Sector")
        self.assertContains(response, "Unavailable", count=3)
        self.assertContains(response, "Monthly model history")
        self.assertContains(response, "Only the latest period is stored")

    def test_stock_beta_history_renders_across_month_ends(self):
        earlier = StockModelFit.objects.create(
            build=self.build,
            security=self.security,
            as_of="2026-07-31",
            model_level="base",
            alpha=0.002,
            adjusted_r2=0.55,
            residual_volatility=0.2,
            active_factor_count=1,
            design_factor_count=2,
            observation_count=250,
            coverage=0.95,
        )
        StockExposureSnapshot.objects.create(
            model_fit=earlier,
            factor=self.market,
            beta=0.8,
            standard_error=0.1,
            t_stat=8.0,
            p_value=0.001,
            confidence_low=0.6,
            confidence_high=1.0,
        )
        response = self.client.get(
            "/stock/",
            {
                "build_id": self.build.id,
                "ticker": "XYZ",
                "model_level": "base",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Beta over time")
        self.assertContains(response, "data-stock-beta-history")
        self.assertNotContains(response, "Only one month-end is stored")
        self.assertEqual(response.context["selected_stock_beta_factor"], "Market")
        self.assertEqual(len(response.context["stock_beta_history_rows"]), 2)

    def test_saved_and_manual_portfolio_exposures(self):
        saved = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "exposure",
                "build_id": self.build.id,
                "model_level": "base",
                "portfolio_id": self.portfolio.id,
            },
        )
        self.assertEqual(saved.status_code, 200)
        self.assertContains(saved, "Factor UI Portfolio")
        self.assertContains(saved, "1.1000")
        self.assertContains(saved, "Risk decomposition unavailable")

        manual = self.client.post(
            f"/portfolio/?portfolio_tab=exposure&build_id={self.build.id}&model_level=base",
            {"holdings": "ticker,weight\nXYZ,100"},
        )
        self.assertEqual(manual.status_code, 200)
        self.assertContains(manual, "Analyze without saving")
        self.assertContains(manual, "1.1000")

    @patch("desk.views.replay_portfolio_history")
    def test_portfolio_performance_subtab_renders_saved_history(self, replay):
        replay.return_value = {
            "metrics": {
                "annual_return": 0.12,
                "annual_volatility": 0.18,
                "sharpe": 0.8,
                "max_drawdown": -0.09,
                "alpha": 0.03,
                "beta": 0.9,
                "tracking_error": 0.08,
                "turnover": 2.0,
            },
            "benchmark_metrics": {
                "annual_return": 0.08,
                "annual_volatility": 0.15,
                "sharpe": 0.6,
                "max_drawdown": -0.12,
            },
            "chart_rows": [
                {
                    "date": "2026-01-02",
                    "portfolio_equity": 1.01,
                    "spy_equity": 1.005,
                    "portfolio_drawdown": 0,
                    "spy_drawdown": 0,
                }
            ],
            "rebalances": [],
            "warnings": [],
            "methodology": "Monthly equal-weight replay.",
            "start": date(2026, 1, 2),
            "end": date(2026, 8, 31),
            "observations": 168,
        }
        response = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "performance",
                "build_id": self.build.id,
                "model_level": "base",
                "portfolio_id": self.portfolio.id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Historical Performance")
        self.assertContains(response, "Performance statistics")
        self.assertContains(response, "data-history-series")
        replay.assert_called_once()

    def test_holdings_matrix_covers_design_factors_and_flags_missing_fits(self):
        response = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "holdings",
                "build_id": self.build.id,
                "model_level": "base",
                "portfolio_id": self.mixed_portfolio.id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["portfolio_matrix_factors"], ["Market", "Value"]
        )
        rows = {
            row["security"].ticker: row["matrix_values"]
            for row in response.context["factor_portfolio_holdings"]
        }
        self.assertEqual(
            [value["beta"] for value in rows["XYZ"]],
            [1.1, 0],
        )
        self.assertEqual(
            [value["beta"] for value in rows["NOFIT"]],
            [None, None],
        )
        self.assertContains(response, "No fit")
        self.assertEqual(response.context["portfolio_modeled_holdings"], 1)

    def test_portfolio_defaults_to_build_that_models_the_most_holdings(self):
        auto = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "holdings",
                "model_level": "base",
                "portfolio_id": self.portfolio.id,
            },
        )
        self.assertEqual(auto.context["selected_factor_build"].id, self.build.id)
        self.assertEqual(auto.context["portfolio_modeled_holdings"], 1)

        requested = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "holdings",
                "build_id": self.empty_build.id,
                "model_level": "base",
                "portfolio_id": self.portfolio.id,
            },
        )
        self.assertEqual(
            requested.context["selected_factor_build"].id, self.empty_build.id
        )
        self.assertEqual(requested.context["portfolio_modeled_holdings"], 0)
        self.assertContains(requested, f"Use {self.build.display_label}")

    def test_portfolio_holdings_show_matrix_and_all_model_levels(self):
        response = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "holdings",
                "build_id": self.build.id,
                "model_level": "base",
                "portfolio_id": self.portfolio.id,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Holding factor exposure matrix")
        self.assertNotContains(response, "Holding model details")
        self.assertContains(response, "Example Corp")
        self.assertContains(response, "1.100")
        for label in (
            "Base",
            "Base + Sector",
            "Base + Sector + Industry",
            "All Factors",
        ):
            self.assertContains(response, label)


class PortfolioRiskSnapshotTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.build = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            is_canonical=True,
            definition_version=PROXY_MODEL_VERSION,
        )
        cls.market = FactorDefinition.objects.create(
            name="Market",
            model_version=PROXY_MODEL_VERSION,
            family="market",
            level="base",
            provenance_badge="exact_etf",
            sort_order=1,
        )
        cls.value = FactorDefinition.objects.create(
            name="Value",
            model_version=PROXY_MODEL_VERSION,
            family="style",
            level="base",
            provenance_badge="exact_etf",
            sort_order=2,
        )
        for offset, stamp in enumerate(
            ("2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01")
        ):
            for factor, scale in ((cls.market, 0.004), (cls.value, 0.002)):
                FactorObservation.objects.create(
                    build=cls.build,
                    factor=factor,
                    date=stamp,
                    scaled_return=scale * (1 if offset % 2 else -1),
                    cumulative_index=1 + scale * offset,
                )
        cls.securities = [
            Security.objects.create(ticker=ticker, name=f"{ticker} Corp", asset_type="stock")
            for ticker in ("AAA", "BBB")
        ]
        # The same tickers carry a different All Factors fit so a snapshot built
        # from the wrong catalog is detectable.
        betas = {
            "base_sector": {"AAA": (1.0, 0.5), "BBB": (2.0, -0.5)},
            "all_factors": {"AAA": (9.0, 9.0), "BBB": (9.0, 9.0)},
        }
        for level, by_ticker in betas.items():
            for security in cls.securities:
                market_beta, value_beta = by_ticker[security.ticker]
                fit = StockModelFit.objects.create(
                    build=cls.build,
                    security=security,
                    as_of="2026-09-02",
                    model_level=level,
                    alpha=0.0,
                    adjusted_r2=0.5,
                    residual_volatility=0.2,
                    active_factor_count=2,
                    design_factor_count=2,
                    observation_count=252,
                    coverage=1.0,
                )
                for factor, beta in ((cls.market, market_beta), (cls.value, value_beta)):
                    StockExposureSnapshot.objects.create(
                        model_fit=fit,
                        factor=factor,
                        beta=beta,
                        standard_error=0.1,
                        t_stat=beta / 0.1,
                        p_value=0.01,
                        confidence_low=beta - 0.2,
                        confidence_high=beta + 0.2,
                    )
        cls.portfolio = Portfolio.objects.create(
            name="Screen Portfolio",
            configuration={"stock_selection": {"model_level": "base_sector"}},
        )
        for security in cls.securities:
            PortfolioHolding.objects.create(
                portfolio=cls.portfolio, security=security, weight=0.5
            )

    def dataset(self):
        dates = pd.bdate_range("2026-08-03", periods=25)
        return SimpleNamespace(
            prices=pd.DataFrame(
                {
                    "AAA": np.linspace(100, 110, len(dates)),
                    "BBB": np.linspace(50, 54, len(dates)),
                },
                index=dates,
            ),
            metadata=pd.DataFrame(
                [
                    {"ticker": "AAA", "asset_type": "stock", "sector": "Technology"},
                    {"ticker": "BBB", "asset_type": "stock", "sector": "Energy"},
                ]
            ),
        )

    @patch("desk.workflows._dataset_for_build")
    def test_refresh_stores_a_snapshot_for_the_requested_model_level(self, dataset):
        dataset.return_value = self.dataset()
        self.assertTrue(
            _refresh_portfolio_risk(self.portfolio, self.build, "base_sector")
        )
        snapshot = self.portfolio.risk_snapshots.get()
        self.assertEqual(snapshot.factor_build_id, self.build.pk)
        self.assertAlmostEqual(snapshot.exposures["Market"], 1.5)
        self.assertAlmostEqual(snapshot.exposures["Value"], 0.0)
        self.assertGreater(snapshot.predicted_volatility, 0)
        self.assertIn("Market", snapshot.component_risk)

    @patch("desk.workflows._dataset_for_build")
    def test_stored_model_level_is_not_blended_with_other_catalogs(self, dataset):
        dataset.return_value = self.dataset()
        _refresh_portfolio_risk(self.portfolio, self.build)
        snapshot = self.portfolio.risk_snapshots.get()
        self.assertAlmostEqual(snapshot.exposures["Market"], 1.5)

    def test_exposure_tab_calculates_factor_risk_without_a_snapshot(self):
        self.assertFalse(self.portfolio.risk_snapshots.exists())
        response = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "exposure",
                "build_id": self.build.pk,
                "portfolio_id": self.portfolio.pk,
                "model_level": "base_sector",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["factor_portfolio_risk"])
        self.assertNotContains(response, "Risk decomposition unavailable")
        html = response.content.decode()
        for label in ("Specific risk", "Factor risk", "Total risk"):
            card = html.split(f"<span>{label}</span>")[1].split("</article>")[0]
            self.assertNotIn("Unavailable", card, label)
        self.assertAlmostEqual(
            response.context["factor_portfolio_specific_risk_share"]
            + response.context["factor_portfolio_factor_risk_share"],
            100.0,
        )

    def test_live_factor_risk_changes_with_the_selected_model_level(self):
        base_sector = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "exposure",
                "build_id": self.build.pk,
                "portfolio_id": self.portfolio.pk,
                "model_level": "base_sector",
            },
        ).context["factor_portfolio_risk"]
        all_factors = self.client.get(
            "/portfolio/",
            {
                "portfolio_tab": "exposure",
                "build_id": self.build.pk,
                "portfolio_id": self.portfolio.pk,
                "model_level": "all_factors",
            },
        ).context["factor_portfolio_risk"]
        self.assertNotEqual(
            base_sector["factor_variance"], all_factors["factor_variance"]
        )
        self.assertNotEqual(
            base_sector["predicted_volatility"],
            all_factors["predicted_volatility"],
        )
        self.assertAlmostEqual(base_sector["exposure"]["Market"], 1.5)
        self.assertAlmostEqual(all_factors["exposure"]["Market"], 9.0)

    def test_temporary_holdings_receive_live_factor_risk(self):
        response = self.client.post(
            (
                f"/portfolio/?portfolio_tab=exposure&build_id={self.build.pk}"
                "&model_level=base_sector"
            ),
            {"holdings": "ticker,weight\nAAA,0.5\nBBB,0.5"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["factor_portfolio_risk"])
        self.assertNotContains(response, "Risk decomposition unavailable")

    def test_refresh_failure_leaves_the_portfolio_usable(self):
        empty = Portfolio.objects.create(name="No Models")
        self.assertFalse(_refresh_portfolio_risk(empty, self.build, "base_sector"))
        self.assertFalse(empty.risk_snapshots.exists())


class SignalsPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.wide_build = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            definition_version=PROXY_MODEL_VERSION,
        )
        cls.deep_build = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            definition_version=PROXY_MODEL_VERSION,
        )
        cls.market = FactorDefinition.objects.create(
            name="Market",
            model_version=PROXY_MODEL_VERSION,
            family="market",
            level="base",
            provenance_badge="exact_etf",
            sort_order=1,
        )
        cls.value = FactorDefinition.objects.create(
            name="Value",
            model_version=PROXY_MODEL_VERSION,
            family="style",
            level="base",
            provenance_badge="public_approximation",
            sort_order=2,
        )
        catalog = FactorModelCatalog.objects.create(
            slug="all_factors",
            name="All Factors",
            model_version=PROXY_MODEL_VERSION,
            available_factor_count=2,
        )
        for position, factor in enumerate((cls.market, cls.value)):
            FactorModelCatalogMembership.objects.create(
                catalog=catalog, factor=factor, position=position
            )
        base_catalog = FactorModelCatalog.objects.create(
            slug="base",
            name="Base",
            model_version=PROXY_MODEL_VERSION,
            available_factor_count=1,
        )
        FactorModelCatalogMembership.objects.create(
            catalog=base_catalog, factor=cls.market, position=0
        )
        cls.base_sector_catalog = FactorModelCatalog.objects.create(
            slug="base_sector",
            name="Base + Sector",
            model_version=PROXY_MODEL_VERSION,
            available_factor_count=2,
        )
        for position, factor in enumerate((cls.market, cls.value)):
            FactorModelCatalogMembership.objects.create(
                catalog=cls.base_sector_catalog, factor=factor, position=position
            )
        FactorObservation.objects.create(
            build=cls.wide_build,
            factor=cls.market,
            date="2026-09-02",
            scaled_return=0.01,
            cumulative_index=1.01,
            horizons={"1": 0.01, "21": 0.025},
            zscores={"1": 0.25, "21": 0.5},
        )
        cls.securities = [
            Security.objects.create(
                ticker=ticker,
                name=f"{ticker} Corp",
                asset_type="stock",
                sector=sector,
            )
            for ticker, sector in (
                ("HIGH", "Technology"),
                ("ZERO", "Technology"),
                ("LOW", "Financials"),
            )
        ]
        for security, beta in zip(cls.securities, (2.0, None, -1.0)):
            fit = StockModelFit.objects.create(
                build=cls.wide_build,
                security=security,
                as_of="2026-09-02",
                model_level="all_factors",
                alpha=0,
                adjusted_r2=0.6,
                residual_volatility=0.2,
                active_factor_count=1 if beta is not None else 0,
                design_factor_count=2,
                observation_count=756,
                coverage=1,
            )
            if beta is not None:
                StockExposureSnapshot.objects.create(
                    model_fit=fit,
                    factor=cls.market,
                    beta=beta,
                    standard_error=0.1,
                    t_stat=beta / 0.1,
                    p_value=0.01,
                    confidence_low=beta - 0.2,
                    confidence_high=beta + 0.2,
                )
        for stamp in ("2026-08-31", "2026-09-02"):
            StockModelFit.objects.create(
                build=cls.deep_build,
                security=cls.securities[0],
                as_of=stamp,
                model_level="all_factors",
                alpha=0,
                adjusted_r2=0.5,
                residual_volatility=0.2,
                active_factor_count=0,
                design_factor_count=2,
                observation_count=756,
                coverage=1,
            )

    def test_defaults_to_widest_build_and_ranks_beta_zscore(self):
        response = self.client.get("/signals/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_signal_build"], self.wide_build)
        rows = response.context["signal_ranking_rows"]
        self.assertEqual(
            [row["security"].ticker for row in rows], ["HIGH", "ZERO", "LOW"]
        )
        self.assertGreater(rows[0]["score"], rows[1]["score"])
        self.assertGreater(rows[1]["score"], rows[2]["score"])
        self.assertFalse(rows[1]["selected"])
        self.assertEqual(rows[1]["beta"], 0)
        self.assertContains(response, "Factor signals")
        self.assertContains(response, "ElasticNet-omitted exposures")
        html = response.content.decode()
        self.assertIn("<span>1d factor return</span><strong>1.00%</strong>", html)
        self.assertIn("<span>1d factor z</span><strong>0.25</strong>", html)
        self.assertIn("<span>21d factor return</span><strong>2.50%</strong>", html)
        self.assertIn("<span>21d factor z</span><strong>0.50</strong>", html)
        ranking = html.split("Latest stock ranking", 1)[1]
        self.assertNotIn(">p-value<", ranking)
        self.assertNotIn(">Contribution<", ranking)
        self.assertNotIn(">Adjusted R²<", ranking)
        self.assertNotIn(">Coverage<", ranking)

    def test_direction_and_sector_change_the_ranked_universe(self):
        response = self.client.get(
            "/signals/",
            {
                "build_id": self.wide_build.pk,
                "model_level": "all_factors",
                "factor": "Market",
                "direction": "-1",
                "sector": "Technology",
            },
        )
        rows = response.context["signal_ranking_rows"]
        self.assertEqual([row["security"].ticker for row in rows], ["ZERO", "HIGH"])

    def test_incompatible_factor_falls_back_to_model_catalog(self):
        response = self.client.get(
            "/signals/",
            {
                "build_id": self.wide_build.pk,
                "model_level": "base",
                "factor": "Value",
            },
        )
        self.assertEqual(response.context["selected_signal_factor"], self.market)
        self.assertNotContains(response, ">Value</span>")

    def test_company_detail_renders_stored_history(self):
        response = self.client.get(
            "/signals/",
            {
                "build_id": self.deep_build.pk,
                "model_level": "all_factors",
                "factor": "Market",
                "ticker": "HIGH",
            },
        )
        self.assertEqual(len(response.context["signal_history_rows"]), 2)
        self.assertContains(response, "data-signal-beta-history", count=2)
        self.assertContains(response, "data-signal-percentile-history", count=2)
        self.assertNotContains(response, "data-signal-fit-history")
        self.assertNotContains(response, "Residual vol")
        self.assertNotContains(response, "<th>Coverage</th>")
        for row in response.context["signal_history_rows"]:
            self.assertIsNotNone(row["percentile"])

    def test_company_history_job_is_deduplicated(self):
        payload = {
            "build_id": self.wide_build.pk,
            "model_level": "all_factors",
            "factor": "Market",
            "ticker": "ZERO",
        }
        first = self.client.post("/signals/company/", payload)
        self.assertEqual(first.status_code, 302)
        job = Job.objects.get(kind="monthly_exposures")
        self.assertEqual(job.parameters["tickers"], ["ZERO"])
        self.assertEqual(job.parameters["max_months"], 24)
        second = self.client.post("/signals/company/", payload)
        self.assertEqual(second.status_code, 302)
        self.assertEqual(Job.objects.filter(kind="monthly_exposures").count(), 1)

    def test_signal_job_poll_reloads_on_success(self):
        job = Job.objects.create(
            kind="monthly_exposures",
            status="succeeded",
            progress=100,
        )
        response = self.client.get(f"/jobs/{job.pk}/", {"reload_on_success": "1"})
        self.assertContains(response, "window.location.reload()")

    def test_finished_optimization_opens_its_own_study(self):
        # The optimizer only renders a study named in the URL, so reloading the
        # builder would hide a run that just finished.
        job = Job.objects.create(
            kind="optimization",
            status="succeeded",
            progress=100,
            result={"study_id": 7, "scenario_id": 9, "success": True},
        )
        response = self.client.get(f"/jobs/{job.pk}/", {"reload_on_success": "1"})
        self.assertContains(
            response,
            r'window.location.assign("/optimizer/?study_id\u003D7'
            r'\u0026scenario_id\u003D9");',
        )
        self.assertNotContains(response, "window.location.reload();")

    def stock_selection_payload(self, **overrides):
        payload = {
            "name": "Quality Leaders",
            "build_id": self.wide_build.pk,
            "model_level": "all_factors",
            "top_n": 2,
            f"factor_{self.market.pk}": "on",
            f"weight_{self.market.pk}": 1,
            f"direction_{self.market.pk}": 1,
            "minimum_coverage": 0,
            "minimum_adjusted_r2": "",
            "excluded_tickers": "HIGH",
            "manual_tickers": "LOW",
        }
        payload.update(overrides)
        return payload

    def test_stock_selection_page_and_legacy_redirects(self):
        response = self.client.get("/stock-selection/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Stock Selection")
        self.assertContains(response, "Factors and weights")
        self.assertNotContains(response, ">Portfolios</a>")
        self.assertNotContains(response, ">Screens</a>")
        self.assertRedirects(
            self.client.get("/portfolios/"),
            "/stock-selection/",
        )
        self.assertRedirects(
            self.client.get("/screens/"),
            "/stock-selection/",
        )
        self.assertContains(response, "Create manual portfolio")

    def test_stock_selection_defaults_to_base_sector_style_screen(self):
        style_factors = []
        for position, name in enumerate(
            ("Momentum", "Value", "Quality", "Growth"), start=10
        ):
            factor, _ = FactorDefinition.objects.get_or_create(
                name=name,
                model_version=PROXY_MODEL_VERSION,
                defaults={
                    "family": "style",
                    "level": "base",
                    "provenance_badge": "exact_etf",
                    "sort_order": position,
                },
            )
            style_factors.append(factor)
        for position, factor in enumerate(style_factors, start=10):
            FactorModelCatalogMembership.objects.get_or_create(
                catalog=self.base_sector_catalog,
                factor=factor,
                defaults={"position": position},
            )
        config = _default_selection_config()
        self.assertEqual(config["model_level"], "base_sector")
        self.assertEqual(config["top_n"], 20)
        self.assertEqual(config["minimum_trading_days"], 252)
        self.assertIsNone(config["minimum_coverage"])
        self.assertEqual(
            config["factor_weights"],
            {
                "Momentum": 25.0,
                "Value": 25.0,
                "Quality": 25.0,
                "Growth": 25.0,
            },
        )
        self.assertEqual(
            config["directions"],
            {
                "Momentum": 1,
                "Value": 1,
                "Quality": 1,
                "Growth": 1,
            },
        )
        page = self.client.get("/stock-selection/?new=1")
        html = page.content.decode()
        self.assertContains(page, 'value="base_sector" selected')
        self.assertContains(page, 'name="top_n"')
        self.assertIn('value="20"', html)
        for name, factor in zip(
            ("Momentum", "Value", "Quality", "Growth"), style_factors
        ):
            self.assertContains(page, f'name="factor_{factor.field_key}"')
            self.assertRegex(
                html,
                rf'name="factor_{factor.field_key}"[^>]*checked',
            )
            self.assertContains(page, f'name="weight_{factor.field_key}"')
            self.assertRegex(
                html,
                rf'name="weight_{factor.field_key}"[^>]*value="25(\.0)?"',
            )

    def test_manual_portfolio_creator_accepts_ticker_list(self):
        response = self.client.post(
            "/portfolios/create-manual/",
            {"name": "Permanent Choices", "tickers": "HIGH, LOW"},
        )
        portfolio = Portfolio.objects.get(name="Permanent Choices")
        self.assertEqual(portfolio.source, "manual")
        self.assertEqual(portfolio.holdings.count(), 2)
        self.assertTrue(
            all(
                abs(holding.weight - 0.5) < 1e-9 for holding in portfolio.holdings.all()
            )
        )
        self.assertIn(f"portfolio_id={portfolio.pk}", response["Location"])

    def test_stock_selection_preview_and_saved_screen(self):
        response = self.client.post(
            "/stock-selection/preview/",
            self.stock_selection_payload(),
        )
        self.assertRedirects(
            response,
            "/stock-selection/?draft=1",
            fetch_redirect_response=False,
        )
        page = self.client.get(response["Location"])
        self.assertEqual(page.context["selection_preview"]["final_count"], 2)
        self.assertEqual(
            {
                row["ticker"]: row["selection_source"]
                for row in page.context["selection_included_rows"]
            },
            {"ZERO": "model", "LOW": "manual"},
        )
        saved = self.client.post(
            "/stock-selection/save/",
            {"name": "Saved Quality Leaders"},
        )
        screen = ScreenDefinition.objects.get(name="Saved Quality Leaders")
        self.assertRedirects(
            saved,
            f"/stock-selection/?screen_id={screen.pk}",
            fetch_redirect_response=False,
        )
        self.assertEqual(screen.runs.get().results.count(), 3)

    def test_selection_form_keys_factors_by_name_not_definition_id(self):
        twin = FactorDefinition.objects.create(
            name="Market",
            model_version="signals-test-parallel",
            family="market",
            level="base",
            provenance_badge="exact_etf",
            sort_order=1,
        )
        self.assertNotEqual(twin.pk, self.market.pk)
        self.assertEqual(twin.field_key, self.market.field_key)
        page = self.client.get("/stock-selection/")
        self.assertContains(page, f'name="factor_{self.market.field_key}"')
        self.assertNotContains(page, f'name="factor_{self.market.pk}"')
        response = self.client.post(
            "/stock-selection/preview/",
            {
                "name": "Quality Leaders",
                "build_id": self.wide_build.pk,
                "model_level": "all_factors",
                "top_n": 2,
                f"factor_{self.market.field_key}": "on",
                f"weight_{self.market.field_key}": 1,
                f"direction_{self.market.field_key}": 1,
                "minimum_coverage": 0,
            },
        )
        self.assertRedirects(
            response,
            "/stock-selection/?draft=1",
            fetch_redirect_response=False,
        )
        self.assertEqual(
            self.client.session["stock_selection_preview"]["factor_weights"],
            {"Market": 1.0},
        )

    def test_a_weight_alone_does_not_select_an_unticked_factor(self):
        response = self.client.post(
            "/stock-selection/preview/",
            self.stock_selection_payload(
                **{f"weight_{self.value.field_key}": "20"},
            ),
        )
        self.assertRedirects(
            response,
            "/stock-selection/?draft=1",
            fetch_redirect_response=False,
        )
        self.assertEqual(
            self.client.session["stock_selection_preview"]["factor_weights"],
            {"Market": 1.0},
        )

    def test_unticked_factors_render_a_locked_weight_and_direction(self):
        html = self.client.get("/stock-selection/").content.decode()
        row = html.split(f'name="factor_{self.market.field_key}"')[1].split("</div>")[0]
        self.assertRegex(row, rf'name="weight_{self.market.field_key}"[^>]*disabled')
        self.assertRegex(row, rf'name="direction_{self.market.field_key}"[^>]*disabled')
        selected = html.split(f'name="factor_{self.value.field_key}"')[1].split(
            "</div>"
        )[0]
        self.assertIn("checked", selected)
        self.assertNotRegex(
            selected, rf'name="weight_{self.value.field_key}"[^>]*disabled'
        )

    def test_checked_factor_with_a_zero_weight_is_reported(self):
        response = self.client.post(
            "/stock-selection/preview/",
            self.stock_selection_payload(
                **{
                    f"factor_{self.value.field_key}": "on",
                    f"weight_{self.value.field_key}": "0",
                },
            ),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Value", response.content.decode())

    def test_stock_selection_creates_equal_weight_portfolio_and_handoffs(self):
        self.client.post(
            "/stock-selection/preview/",
            self.stock_selection_payload(),
        )
        response = self.client.post(
            "/stock-selection/portfolio/",
            {"name": "Selected Portfolio", "action": "factors"},
        )
        portfolio = Portfolio.objects.get(name="Selected Portfolio")
        self.assertEqual(portfolio.source, "screen")
        self.assertEqual(portfolio.holdings.count(), 2)
        for holding in portfolio.holdings.all():
            self.assertAlmostEqual(holding.weight, 0.5)
        self.assertIn("/portfolio/?", response["Location"])
        self.assertIn(f"portfolio_id={portfolio.pk}", response["Location"])
        self.assertIn("portfolio_tab=exposure", response["Location"])

        optimizer_page = self.client.get(
            "/optimizer/",
            {"portfolio_id": portfolio.pk, "factor_build_id": self.wide_build.pk},
        )
        self.assertContains(
            optimizer_page,
            f'<option value="{portfolio.pk}" selected>',
            html=False,
        )
        self.assertContains(
            optimizer_page,
            f'<option value="{self.wide_build.pk}" selected>',
            html=False,
        )


class OptimizationWorkflowTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        dates = pd.bdate_range("2023-10-02", periods=336)
        cls.dates = dates
        cls.build = FactorBuild.objects.create(
            as_of=dates[-1].date(),
            status="succeeded",
            definition_version=PROXY_MODEL_VERSION,
            configuration={"data_snapshot_id": 1},
        )
        cls.market = FactorDefinition.objects.create(
            name="Market",
            family="market",
            model_version=PROXY_MODEL_VERSION,
            sort_order=1,
        )
        cls.oil = FactorDefinition.objects.create(
            name="Oil",
            family="macro",
            model_version=PROXY_MODEL_VERSION,
            sort_order=2,
        )
        catalog = FactorModelCatalog.objects.create(
            slug="base",
            name="Base",
            model_version=PROXY_MODEL_VERSION,
            available_factor_count=2,
        )
        for position, factor in enumerate((cls.market, cls.oil)):
            FactorModelCatalogMembership.objects.create(
                catalog=catalog, factor=factor, position=position
            )
        for number, stamp in enumerate(dates):
            FactorObservation.objects.create(
                build=cls.build,
                factor=cls.market,
                date=stamp.date(),
                scaled_return=np.sin(number / 11) / 100,
            )
            FactorObservation.objects.create(
                build=cls.build,
                factor=cls.oil,
                date=stamp.date(),
                scaled_return=np.cos(number / 13) / 100,
            )
        cls.securities = {
            ticker: Security.objects.create(
                ticker=ticker,
                name=ticker,
                sector="Technology" if ticker == "AAA" else "Financials",
            )
            for ticker in ("AAA", "BBB", "SPY")
        }
        cls.portfolio = Portfolio.objects.create(name="Optimizer Input")
        for ticker in ("AAA", "BBB"):
            PortfolioHolding.objects.create(
                portfolio=cls.portfolio,
                security=cls.securities[ticker],
                weight=0.5,
            )
        for ticker, oil_beta in (("AAA", 1.0), ("BBB", -0.5), ("SPY", 0.1)):
            fit = StockModelFit.objects.create(
                build=cls.build,
                security=cls.securities[ticker],
                as_of=dates[-1].date(),
                model_level="base",
                alpha=0,
                adjusted_r2=0.6,
                residual_volatility=0.2,
                active_factor_count=2,
                design_factor_count=2,
                observation_count=336,
                coverage=1,
            )
            for factor, beta in ((cls.market, 1.0), (cls.oil, oil_beta)):
                StockExposureSnapshot.objects.create(
                    model_fit=fit,
                    factor=factor,
                    beta=beta,
                    standard_error=0.1,
                    t_stat=beta / 0.1,
                    p_value=0.01,
                    confidence_low=beta - 0.2,
                    confidence_high=beta + 0.2,
                )
        future_fit = StockModelFit.objects.create(
            build=cls.build,
            security=cls.securities["AAA"],
            as_of=(dates[-1] + pd.offsets.MonthBegin(1)).date(),
            model_level="base",
            alpha=0,
            adjusted_r2=0.9,
            residual_volatility=0.1,
            active_factor_count=1,
            design_factor_count=2,
            observation_count=336,
            coverage=1,
        )
        StockExposureSnapshot.objects.create(
            model_fit=future_fit,
            factor=cls.oil,
            beta=99,
            standard_error=0.1,
            t_stat=990,
            p_value=0,
            confidence_low=98,
            confidence_high=100,
        )
        cls.dataset = SimpleNamespace(
            prices=pd.DataFrame(
                {
                    "AAA": 100
                    * np.exp(np.cumsum(np.sin(np.arange(len(dates)) / 9) / 100)),
                    "BBB": 100
                    * np.exp(np.cumsum(np.cos(np.arange(len(dates)) / 10) / 100)),
                    "SPY": 100
                    * np.exp(np.cumsum(np.sin(np.arange(len(dates)) / 15) / 100)),
                },
                index=dates,
            )
        )

    def parameters(self, **overrides):
        values = {
            "portfolio_id": self.portfolio.pk,
            "factor_build_id": self.build.pk,
            "model_level": "base",
            "max_weight": 0.8,
            "lookback": 336,
        }
        values.update(overrides)
        return values

    @patch("desk.workflows._dataset_for_build")
    def test_inputs_use_requested_model_and_ignore_future_fit(self, dataset):
        dataset.return_value = self.dataset
        inputs = _optimization_inputs(self.parameters())
        self.assertEqual(inputs["model_level"], "base")
        self.assertEqual(inputs["latest"]["AAA"].as_of, self.build.as_of)
        self.assertEqual(inputs["exposures"].at["AAA", "Oil"], 1.0)

    @patch("desk.workflows._dataset_for_build")
    def test_standard_study_is_persisted(self, dataset):
        dataset.return_value = self.dataset
        standard = run_optimization(
            self.parameters(
                name="Robust Standard",
                objective="mean_variance",
                expected_return_model="historical_shrinkage",
            ),
            lambda *_: None,
        )
        self.assertTrue(standard["success"])
        standard_scenario = (
            Portfolio.objects.get(pk=self.portfolio.pk)
            .optimization_studies.get(name="Robust Standard")
            .variants.get()
        )
        self.assertEqual(standard_scenario.holdings.count(), 2)
        self.assertIsNotNone(standard_scenario.holdings.first().expected_return)

    @patch("desk.workflows._dataset_for_build")
    def test_factor_premium_model_prices_betas_and_persists_diagnostics(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Factor Premium",
                objective="max_sharpe",
                expected_return_model="factor_premium",
                factor_premia={"Oil": 0.05},
                risk_free_rate=0.04,
                premium_prior_dispersion=0,
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        forecasts = {
            holding.security.ticker: holding.expected_return
            for holding in scenario.holdings.select_related("security")
        }
        self.assertAlmostEqual(forecasts["AAA"], 0.09)
        self.assertAlmostEqual(forecasts["BBB"], 0.015)
        self.assertEqual(scenario.diagnostics["risk_free_rate"], 0.04)
        self.assertGreater(scenario.diagnostics["forecast_dispersion"], 0)
        self.assertIn("Oil", scenario.diagnostics["factor_premia"])

    @patch("desk.workflows._dataset_for_build")
    def test_equal_sharpe_model_prices_volatility_and_ignores_k_for_tangency(
        self, dataset
    ):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Equal Sharpe",
                objective="max_sharpe",
                expected_return_model="equal_sharpe",
                common_sharpe=0.5,
                risk_free_rate=0.04,
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        self.assertIn("equal_sharpe", scenario.diagnostics)
        self.assertEqual(scenario.diagnostics["equal_sharpe"]["common_sharpe"], 0.5)
        self.assertTrue(scenario.diagnostics["equal_sharpe"]["cancels_for_max_sharpe"])
        self.assertGreater(scenario.diagnostics["forecast_dispersion"], 0)
        self.assertIn(
            "Maximum Diversification", " ".join(scenario.diagnostics["warnings"])
        )

    @patch("desk.workflows._dataset_for_build")
    def test_premiums_blend_toward_trailing_factor_means(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Blended Premium",
                objective="max_sharpe",
                expected_return_model="factor_premium",
                factor_premia={"Oil": 0.05},
                risk_free_rate=0.04,
            ),
            lambda *_: None,
        )
        premium = OptimizationScenario.objects.get(
            pk=result["scenario_id"]
        ).diagnostics["factor_premia"]["Oil"]
        self.assertEqual(premium["assumption"], 0.05)
        self.assertGreater(premium["history_weight"], 0)
        self.assertLess(premium["history_weight"], 1)
        self.assertNotAlmostEqual(premium["premium"], 0.05)
        self.assertAlmostEqual(
            premium["premium"],
            0.05 + premium["history_weight"] * (premium["historical_mean"] - 0.05),
        )

    @patch("desk.workflows._dataset_for_build")
    def test_optimization_prefers_the_cash_rate_of_the_as_of_date(self, dataset):
        dataset.return_value = self.dataset
        store_risk_free_rates(
            pd.Series([0.0125], index=pd.to_datetime([self.dates[0]]))
        )
        result = run_optimization(
            self.parameters(
                name="Point In Time Cash",
                objective="max_sharpe",
                expected_return_model="factor_premium",
                factor_premia={"Oil": 0.05},
                risk_free_rate=0.04,
                premium_prior_dispersion=0,
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        forecasts = {
            holding.security.ticker: holding.expected_return
            for holding in scenario.holdings.select_related("security")
        }
        self.assertAlmostEqual(scenario.diagnostics["risk_free_rate"], 0.0125)
        self.assertAlmostEqual(forecasts["AAA"], 0.0625)
        self.assertNotIn(
            "No cash rate history is loaded",
            " ".join(scenario.diagnostics.get("warnings", [])),
        )

    @patch("desk.workflows._dataset_for_build")
    def test_missing_cash_rate_history_warns_on_sharpe_runs(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Missing Cash",
                objective="max_sharpe",
                expected_return_model="factor_premium",
                factor_premia={"Oil": 0.05},
                risk_free_rate=0.04,
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        self.assertIn(
            "No cash rate history is loaded",
            " ".join(scenario.diagnostics["warnings"]),
        )

    def test_flat_forecasts_warn_that_return_objective_reduces_to_min_variance(self):
        warnings = _forecast_warnings(
            pd.Series({"AAA": 0.08, "BBB": 0.08}), "max_sharpe"
        )
        self.assertIn("equivalent to minimum variance", warnings[0])

    @patch("desk.workflows._dataset_for_build")
    def test_standard_study_stores_original_ex_ante_metrics(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Original Metrics",
                objective="min_variance",
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        original = scenario.comparison["original"]
        self.assertIsNone(original["expected_return"])
        self.assertGreater(original["expected_volatility"], 0)
        self.assertEqual(original["turnover"], 0)
        self.assertNotAlmostEqual(
            original["expected_volatility"], scenario.expected_volatility
        )
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": scenario.study_id,
                "scenario_id": scenario.pk,
                "mode": "standard",
            },
        )
        self.assertContains(page, "Original versus optimized")
        self.assertContains(page, "Stock Selection portfolio")

    @patch("desk.workflows._dataset_for_build")
    def test_result_sections_are_collapsible_and_settings_start_closed(self, dataset):
        dataset.return_value = self.dataset
        empty = self.client.get("/optimizer/", {"mode": "standard"}).content.decode()
        builder = empty.split('data-collapse-key="optimizer-builder-standard"')[1]
        self.assertIn("open", builder.split(">")[0])

        result = run_optimization(
            self.parameters(
                name="Collapsible Sections",
                objective="min_variance",
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": scenario.study_id,
                "scenario_id": scenario.pk,
                "mode": "standard",
            },
        )
        html = page.content.decode()
        self.assertEqual(html.count("<details"), html.count("</details>"))
        self.assertEqual(html.count("<summary"), html.count("</summary>"))
        for key in (
            "optimizer-builder-standard",
            "optimizer-exposure",
            "optimizer-settings",
            "optimizer-holdings",
            "optimizer-history-list",
        ):
            self.assertIn(f'data-collapse-key="{key}"', html)
        # With results on screen the builder collapses so the output is reachable.
        builder = html.split('data-collapse-key="optimizer-builder-standard"')[1]
        self.assertNotIn("open", builder.split(">")[0])

    @patch("desk.workflows._dataset_for_build")
    def test_optimizer_page_survives_a_deleted_study_id(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Deleted Study",
                objective="min_variance",
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        missing_study_id = scenario.study_id + 1000
        page = self.client.get(
            "/optimizer/", {"study_id": missing_study_id, "mode": "standard"}
        )
        self.assertEqual(page.status_code, 200)
        # A study that no longer exists loads no results rather than quietly
        # swapping in a different run.
        self.assertIsNone(page.context["selected_optimization_study"])
        self.assertIsNone(page.context["selected_scenario"])
        self.assertContains(page, "Standard optimization studies")

    def test_sector_choices_follow_the_portfolio_picked_in_the_form(self):
        # Sectors come from the chosen portfolio's holdings, so the control
        # stays disabled with an explanation until a portfolio is picked.
        blank = self.client.get("/optimizer/", {"mode": "standard"})
        self.assertEqual(blank.context["optimizer_sector_options"], [])
        self.assertContains(blank, "Choose a portfolio above to list the sectors")
        chosen = self.client.get(
            "/optimizer/",
            {"mode": "standard", "portfolio_id": self.portfolio.pk},
        )
        self.assertEqual(
            chosen.context["optimizer_sector_options"], ["Financials", "Technology"]
        )
        self.assertNotContains(chosen, "Choose a portfolio above to list the sectors")

    @patch("desk.workflows._dataset_for_build")
    def test_optimizer_opens_on_the_builder_until_a_study_is_chosen(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(name="Not Auto Opened", objective="min_variance"),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        landing = self.client.get("/optimizer/", {"mode": "standard"})
        self.assertIsNone(landing.context["selected_optimization_study"])
        self.assertIsNone(landing.context["selected_scenario"])
        self.assertNotContains(landing, "Run settings and constraints")
        # The stored run is still listed, and opening it loads the results.
        self.assertContains(landing, "Not Auto Opened")
        opened = self.client.get(
            "/optimizer/", {"study_id": scenario.study_id, "mode": "standard"}
        )
        self.assertEqual(opened.context["selected_scenario"], scenario)
        self.assertContains(opened, "Run settings and constraints")

    @patch("desk.workflows._dataset_for_build")
    def test_optimizer_page_reports_settings_and_constraints(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Guardrailed Study",
                objective="mean_variance",
                expected_return_model="historical_shrinkage",
                risk_aversion=2.5,
                turnover_cap=0.35,
                factor_bounds={"Oil": [-0.2, 0.4]},
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": scenario.study_id,
                "scenario_id": scenario.pk,
                "mode": "standard",
            },
        )
        html = page.content.decode()
        # The method summary explains the objective and its risk weight.
        self.assertIn("Maximum return minus", html)
        self.assertIn("Historical shrinkage", html)
        self.assertIn("Risk weight", html)
        self.assertIn("2.5", html)
        # Guardrails are spelled out rather than left in a raw payload.
        self.assertIn("Turnover cap", html)
        self.assertIn("35.00%", html)
        self.assertIn("-0.20 to 0.40", html)
        self.assertIn("0.0% to 80.0%", html)
        # Holdings sit under the original-versus-optimized table and above
        # sector weights, with the study list remaining at the bottom.
        self.assertLess(
            html.index('data-collapse-key="optimizer-comparison"'),
            html.index('data-collapse-key="optimizer-holdings"'),
        )
        self.assertLess(
            html.index('data-collapse-key="optimizer-holdings"'),
            html.index('data-collapse-key="optimizer-sectors"'),
        )
        self.assertLess(
            html.index('data-collapse-key="optimizer-holdings"'),
            html.index('data-collapse-key="optimizer-history-list"'),
        )
        self.assertIn("Guardrailed Study", html)
        # Applying weights back to a portfolio is no longer offered anywhere.
        self.assertNotIn("Apply selected variant", html)
        self.assertEqual(
            self.client.post(f"/optimization/{scenario.pk}/apply/").status_code, 404
        )

    @patch("desk.workflows._dataset_for_build")
    def test_sector_bounds_are_reported_against_realized_weights(self, dataset):
        dataset.return_value = self.dataset
        sector = self.portfolio.holdings.select_related("security")[
            0
        ].security.sector
        result = run_optimization(
            self.parameters(
                name="Sector Bounded",
                objective="min_variance",
                sector_bounds={sector: [0.4, 0.6]},
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": scenario.study_id,
                "scenario_id": scenario.pk,
                "mode": "standard",
            },
        )
        sector_rows = page.context["optimizer_sector_rows"]
        bounded = next(row for row in sector_rows if row["sector"] == sector)
        # The solver honoured the weight bound, and the row says so explicitly.
        self.assertTrue(0.4 - 1e-6 <= bounded["optimized"] <= 0.6 + 1e-6)
        self.assertTrue(bounded["within"])
        self.assertEqual(bounded["bound"], "40.0% to 60.0%")
        # Sector weight is a different quantity from any factor beta, so an
        # unconstrained sector factor must not borrow the sector's bound.
        for row in page.context["optimizer_exposure_rows"]:
            if row["name"] == sector:
                self.assertEqual(row["bound"], "")

    @patch("desk.workflows._dataset_for_build")
    def test_risk_is_reported_as_volatility_and_variance_shares(self, dataset):
        dataset.return_value = self.dataset
        result = run_optimization(
            self.parameters(
                name="Risk Split",
                objective="min_variance",
            ),
            lambda *_: None,
        )
        scenario = OptimizationScenario.objects.get(pk=result["scenario_id"])
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": scenario.study_id,
                "scenario_id": scenario.pk,
                "mode": "standard",
            },
        )
        self.assertNotIn("optimizer_risk_rows", page.context)
        self.assertNotContains(page, "Where the risk comes from")
        self.assertNotContains(page, "Ledoit-Wolf")
        self.assertNotContains(page, 'name="risk_model"')
        # The variant chart and cards duplicated the metric grid, so only the
        # grid remains and it leads with return before risk.
        self.assertNotContains(page, "Variant risk and return")
        self.assertNotContains(page, "optimizer-risk-return")
        metrics = page.content.decode().split('class="metric-grid optimizer-metrics"')[
            1
        ]
        self.assertEqual(
            re.findall(r"<span>([^<]+)</span>", metrics)[:6],
            [
                "Expected return",
                "Expected volatility",
                "Sharpe ratio",
                "Turnover",
                "Factor risk",
                "Stock-specific risk",
            ],
        )
        # Volatility is the square root of variance, not a rescaled variance.
        self.assertEqual(
            variance_volatility(scenario.factor_variance, 2),
            f"{math.sqrt(scenario.factor_variance) * 100:.2f}%",
        )
        self.assertEqual(variance_volatility(0.0225, 2), "15.00%")
        self.assertEqual(variance_volatility(None), "—")
        self.assertEqual(variance_volatility(-0.5), "—")
        self.assertNotContains(page, "Factor variance")
        self.assertNotContains(page, "%²")

    def test_sharpe_card_prices_risk_over_the_cash_rate_of_the_run(self):
        # The solver stores the rate it actually used, so the card is an excess
        # return per unit of risk rather than a raw return-to-volatility ratio.
        scenario = SimpleNamespace(
            expected_return=0.12,
            expected_volatility=0.2,
            diagnostics={"excess_return": 0.08, "risk_free_rate": 0.04},
        )
        self.assertEqual(sharpe_ratio(scenario), "0.40")
        # Older rows without an excess figure fall back to the stored rate.
        self.assertEqual(
            sharpe_ratio(
                SimpleNamespace(
                    expected_return=0.12,
                    expected_volatility=0.2,
                    diagnostics={"risk_free_rate": 0.04},
                )
            ),
            "0.40",
        )
        self.assertEqual(
            sharpe_ratio(
                SimpleNamespace(
                    expected_return=0.12, expected_volatility=0, diagnostics={}
                )
            ),
            "—",
        )
        self.assertEqual(
            sharpe_ratio(
                SimpleNamespace(
                    expected_return=None, expected_volatility=0.2, diagnostics=None
                )
            ),
            "—",
        )

    def test_launch_scenario_backtest_requires_selection_portfolio(self):
        scenario = OptimizationScenario.objects.create(
            name="No Selection",
            variant="standard",
            status="succeeded",
            portfolio=self.portfolio,
            objective="min_variance",
        )
        response = self.client.post(f"/optimization/{scenario.pk}/backtest/")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Stock Selection", response.content.decode())

    def test_legacy_scenario_mode_renders_standard_only(self):
        study = OptimizationStudy.objects.create(
            portfolio=self.portfolio,
            factor_build=self.build,
            name="Standard Only Study",
            model_level="base",
            status="succeeded",
        )
        scenario = OptimizationScenario.objects.create(
            study=study,
            name="Standard Only Variant",
            variant="standard",
            status="succeeded",
            portfolio=self.portfolio,
            objective="min_variance",
        )
        page = self.client.get(
            "/optimizer/",
            {
                "mode": "scenario",
                "study_id": study.pk,
                "scenario_id": scenario.pk,
            },
        )
        self.assertContains(page, "Standard Only Study")
        self.assertContains(page, "Historical simulation")
        self.assertNotContains(page, "Macro scenario")
        self.assertNotContains(page, "data-macro-dialog")
        self.assertEqual(page.context["selected_optimization_study"], study)
        self.assertEqual(page.context["selected_scenario"], scenario)
        self.assertEqual(
            self.client.get(
                f"/optimization/studies/{study.pk}/macro/AAA/"
            ).status_code,
            404,
        )

    def _add_monthly_selection_history(self):
        self.portfolio.configuration = {
            "stock_selection": {
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "factor_weights": {"Oil": 1.0},
                "directions": {"Oil": 1},
                "top_n": 1,
                "manual_tickers": [],
                "excluded_tickers": [],
            }
        }
        self.portfolio.save(update_fields=["configuration"])
        first = self.dates[40].date()
        second = self.dates[80].date()
        for as_of, oil_betas in (
            (first, {"AAA": 2.0, "BBB": -1.0, "SPY": 0.1}),
            (second, {"AAA": -1.0, "BBB": 3.0, "SPY": 0.1}),
        ):
            for ticker, oil_beta in oil_betas.items():
                fit = StockModelFit.objects.create(
                    build=self.build,
                    security=self.securities[ticker],
                    as_of=as_of,
                    model_level="base",
                    alpha=0,
                    adjusted_r2=0.6,
                    residual_volatility=0.2,
                    active_factor_count=2,
                    design_factor_count=2,
                    observation_count=252,
                    coverage=1,
                )
                for factor, beta in ((self.market, 1.0), (self.oil, oil_beta)):
                    StockExposureSnapshot.objects.create(
                        model_fit=fit,
                        factor=factor,
                        beta=beta,
                        standard_error=0.1,
                        t_stat=beta / 0.1,
                        p_value=0.01,
                        confidence_low=beta - 0.2,
                        confidence_high=beta + 0.2,
                    )
        return first, second

    @patch("selection.history._prices_for_build")
    @patch("desk.workflows._dataset_for_build")
    def test_scenario_backtest_uses_that_month_ranked_names(self, dataset, prices):
        dataset.return_value = self.dataset
        prices.return_value = self.dataset.prices
        first, second = self._add_monthly_selection_history()
        created = run_optimization(
            self.parameters(
                name="Replay Source",
                objective="min_variance",
                max_weight=0.8,
            ),
            lambda *_: None,
        )
        result = run_scenario_backtest(
            {
                "scenario_id": created["scenario_id"],
                "factor_build_id": self.build.pk,
                "lookback_months": 24,
            },
            lambda *_: None,
        )
        run = BacktestRun.objects.get(pk=result["backtest_run_id"])
        months = {
            row["signal_date"]: row["tickers"] for row in run.metrics["rebalances"]
        }
        self.assertEqual(months[str(first)], ["AAA"])
        self.assertEqual(months[str(second)], ["BBB"])
        self.assertIn("original", run.metrics)
        self.assertIn("optimized", run.metrics)
        self.assertIn("spy", run.metrics)
        self.assertEqual(result["rebalances"], run.rebalances.count())
        self.assertEqual(len(run.metrics["rebalances"]), run.rebalances.count())
        self.assertTrue(
            all(
                item.execution_date > item.signal_date
                for item in run.rebalances.all()
            )
        )
        page = self.client.get(
            "/optimizer/",
            {
                "study_id": created["study_id"],
                "scenario_id": created["scenario_id"],
                "mode": "standard",
            },
        )
        self.assertContains(page, "data-optimizer-history-series")
        self.assertContains(page, "Approximate", status_code=200)

    @patch("desk.views.submit_job")
    def test_launch_scenario_backtest_queues_job(self, submit):
        submit.return_value = Job.objects.create(kind="scenario_backtest")
        self.portfolio.configuration = {
            "stock_selection": {
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "factor_weights": {"Oil": 1.0},
                "top_n": 1,
            }
        }
        self.portfolio.save(update_fields=["configuration"])
        scenario = OptimizationScenario.objects.create(
            name="Queued Compare",
            variant="standard",
            status="succeeded",
            portfolio=self.portfolio,
            study=OptimizationStudy.objects.create(
                portfolio=self.portfolio,
                factor_build=self.build,
                name="Queued Compare Study",
                model_level="base",
                status="succeeded",
            ),
            objective="min_variance",
        )
        response = self.client.post(f"/optimization/{scenario.pk}/backtest/")
        self.assertEqual(response.status_code, 200)
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[0], "scenario_backtest")
        self.assertEqual(submit.call_args.args[1]["scenario_id"], scenario.pk)

    def test_optimizer_page_exposes_standard_controls_only(self):
        response = self.client.get(
            "/optimizer/",
            {
                "mode": "scenario",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Standard optimization")
        self.assertNotContains(response, "Macro scenario")
        self.assertNotContains(response, "data-add-scenario")
        self.assertContains(response, "data-add-factor")
        self.assertContains(response, "data-add-sector")
        self.assertContains(response, f'value="{self.oil.field_key}"')
        self.assertContains(response, "Minimum variance")
        self.assertContains(response, "Maximum return minus ½ risk")

    def test_optimizer_page_prefills_market_and_selection_factor_premiums(self):
        self.portfolio.configuration = {
            "stock_selection": {"factor_weights": {"Oil": 25.0}}
        }
        self.portfolio.save(update_fields=["configuration"])
        response = self.client.get(
            "/optimizer/",
            {
                "mode": "standard",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
            },
        )
        self.assertContains(response, "Annual factor premiums")
        self.assertContains(
            response,
            f'name="factor_premium_{self.market.field_key}" step=".1" value="5"',
        )
        self.assertContains(
            response, f'name="factor_premium_{self.oil.field_key}" step=".1" value="2"'
        )

    @patch("desk.views.submit_job")
    def test_standard_form_serializes_structured_constraints(self, submit):
        submit.return_value = Job.objects.create(kind="optimization")
        response = self.client.post(
            "/jobs/launch/optimization/",
            {
                "name": "Parsed Standard",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "objective": "mean_variance",
                f"factor_min_{self.oil.pk}": "-0.2",
                f"factor_max_{self.oil.pk}": "0.4",
                "sector_name_1": "Technology",
                "sector_max_1": "70",
                "max_weight": "80",
                "min_weight": "2",
                "weight_constraint_mode": "relative",
                "max_weight_cap": "90",
                "tracking_error_limit": "5",
            },
        )
        self.assertEqual(response.status_code, 200)
        parameters = submit.call_args.args[1]
        self.assertEqual(parameters["factor_bounds"]["Oil"], [-0.2, 0.4])
        self.assertEqual(parameters["sector_bounds"]["Technology"], [0, 0.7])
        self.assertEqual(parameters["weight_constraint_mode"], "relative")
        self.assertEqual(parameters["min_weight"], 0.02)
        self.assertEqual(parameters["max_weight"], 0.8)
        self.assertEqual(parameters["max_weight_cap"], 0.9)
        self.assertEqual(parameters["tracking_error_limit"], 0.05)

    @patch("desk.views.submit_job")
    def test_post_cannot_activate_macro_mode(self, submit):
        submit.return_value = Job.objects.create(kind="optimization")
        response = self.client.post(
            "/jobs/launch/optimization/",
            {
                "optimization_mode": "scenario",
                "scenario_covariance": "regime_conditioned",
                f"scenario_factor_{self.oil.pk}": "on",
                f"scenario_weight_{self.oil.pk}": "4",
                "expected_return_model": "historical_shrinkage",
                "name": "Standard Only",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "max_weight": "80",
            },
        )
        self.assertEqual(response.status_code, 200)
        parameters = submit.call_args.args[1]
        self.assertNotIn("optimization_mode", parameters)
        self.assertNotIn("scenario_covariance", parameters)
        self.assertNotIn("scenario_factors", parameters)
        self.assertEqual(parameters["expected_return_model"], "historical_shrinkage")

    @patch("desk.views.submit_job")
    def test_standard_form_converts_user_return_percentages(self, submit):
        submit.return_value = Job.objects.create(kind="optimization")
        response = self.client.post(
            "/jobs/launch/optimization/",
            {
                "name": "Parsed Standard",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "objective": "max_return",
                "expected_return_model": "user_supplied",
                "expected_returns": '{"AAA": 8, "BBB": 6}',
                "weight_constraint_mode": "absolute",
                "min_weight": "10",
                "max_weight": "80",
                "turnover_cap": "25",
                "residual_shrinkage": "80",
                "risk_model": "ledoit_wolf",
            },
        )
        self.assertEqual(response.status_code, 200)
        parameters = submit.call_args.args[1]
        self.assertEqual(parameters["expected_returns"], {"AAA": 0.08, "BBB": 0.06})
        self.assertEqual(parameters["min_weight"], 0.1)
        self.assertEqual(parameters["max_weight"], 0.8)
        self.assertEqual(parameters["turnover_cap"], 0.25)
        self.assertEqual(parameters["residual_shrinkage"], 0.8)
        self.assertEqual(parameters["risk_model"], "factor_model")

    @patch("desk.views.submit_job")
    def test_standard_form_converts_factor_premia_and_risk_free_rate(self, submit):
        submit.return_value = Job.objects.create(kind="optimization")
        response = self.client.post(
            "/jobs/launch/optimization/",
            {
                "name": "Factor Premium",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "objective": "max_sharpe",
                "expected_return_model": "factor_premium",
                f"factor_premium_{self.market.pk}": "5",
                f"factor_premium_{self.oil.pk}": "2",
                "risk_free_rate": "4",
                "shrink_factor_betas": "on",
                "weight_constraint_mode": "absolute",
                "min_weight": "0",
                "max_weight": "80",
            },
        )
        self.assertEqual(response.status_code, 200)
        parameters = submit.call_args.args[1]
        self.assertEqual(parameters["factor_premia"], {"Market": 0.05, "Oil": 0.02})
        self.assertEqual(parameters["risk_free_rate"], 0.04)
        self.assertTrue(parameters["shrink_factor_betas"])

    @patch("desk.views.submit_job")
    def test_standard_form_parses_equal_sharpe(self, submit):
        submit.return_value = Job.objects.create(kind="optimization")
        response = self.client.post(
            "/jobs/launch/optimization/",
            {
                "name": "Equal Sharpe",
                "portfolio_id": self.portfolio.pk,
                "factor_build_id": self.build.pk,
                "model_level": "base",
                "objective": "max_sharpe",
                "expected_return_model": "equal_sharpe",
                "common_sharpe": "0.5",
                "risk_free_rate": "4",
                "weight_constraint_mode": "absolute",
                "min_weight": "0",
                "max_weight": "80",
            },
        )
        self.assertEqual(response.status_code, 200)
        parameters = submit.call_args.args[1]
        self.assertEqual(parameters["expected_return_model"], "equal_sharpe")
        self.assertEqual(parameters["common_sharpe"], 0.5)


class ExposureUniverseTests(SimpleTestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "ticker": ["SMALL", "MEGA", "MID", "GOOGL", "GOOG", "SPY"],
                "asset_type": [
                    "stock",
                    "stock",
                    "stock",
                    "stock",
                    "stock",
                    "factor_proxy",
                ],
                "market_cap": [1e9, 3e12, 5e10, 2e12, 1.9e12, None],
                "CIK": [1, 2, 3, 4, 4, None],
            }
        ).set_index("ticker")
        self.returns = pd.DataFrame(
            columns=["SMALL", "MEGA", "MID", "GOOGL", "GOOG", "SPY"]
        )

    def test_top_n_by_market_cap_excludes_spy(self):
        names = _exposure_universe(
            self.metadata, self.returns, {"top_n_by_market_cap": 2}
        )
        self.assertEqual(names, ["MEGA", "GOOGL"])

    def test_dual_class_keeps_one_company(self):
        names = _exposure_universe(
            self.metadata, self.returns, {"top_n_by_market_cap": 3}
        )
        self.assertEqual(names, ["MEGA", "GOOGL", "MID"])
        self.assertNotIn("GOOG", names)

    def test_explicit_tickers(self):
        names = _exposure_universe(
            self.metadata, self.returns, {"tickers": "mid, small"}
        )
        self.assertEqual(names, ["MID", "SMALL"])


class ExposureWorkerCountTests(SimpleTestCase):
    def test_caps_below_cpu_count(self):
        cpu = os.cpu_count() or 1
        self.assertLessEqual(_exposure_worker_count(), max(1, cpu - 2))
        self.assertLessEqual(_exposure_worker_count(10_000), max(1, cpu - 2))
        self.assertEqual(_exposure_worker_count(1), 1)

    def test_current_month_uses_only_latest_available_date(self):
        index = pd.to_datetime(["2026-08-31", "2026-09-02", "2026-09-03"])
        dates = _exposure_month_ends(index, {"update_mode": "backfill"})
        self.assertEqual(
            [stamp.date().isoformat() for stamp in dates],
            ["2026-08-31", "2026-09-03"],
        )
        latest = _exposure_month_ends(index, {"update_mode": "auto"})
        self.assertEqual(
            [stamp.date().isoformat() for stamp in latest],
            ["2026-09-03"],
        )
