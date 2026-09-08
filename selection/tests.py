from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase

from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorModelCatalog,
    FactorModelCatalogMembership,
    StockExposureSnapshot,
    StockModelFit,
)
from market_data.models import Security

from .engine import composite_scores, rank_ic, select_top_n
from .history import replay_screen_history
from .services import save_selection_portfolio
from .stock_selection import build_selection_preview


class SelectionTests(SimpleTestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            {
                "quality": [1.0, 2.0, 3.0, 4.0],
                "risk": [4.0, 3.0, 2.0, 1.0],
            },
            index=list("ABCD"),
        )

    def test_directions_and_weights(self):
        scores = composite_scores(
            self.features, {"quality": 1.0, "risk": 1.0}, {"quality": 1, "risk": -1}
        )
        self.assertEqual(scores.index[0], "D")

    def test_filters_and_top_n(self):
        result = select_top_n(
            self.features,
            {"quality": 1.0},
            top_n=2,
            sectors=pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"}),
            allowed_sectors=["Y"],
            minimums={"quality": 3.0},
        )
        self.assertEqual(list(result.index), ["D", "C"])

    def test_rank_ic(self):
        value = rank_ic(pd.Series(np.arange(8)), pd.Series(np.arange(8)))
        self.assertAlmostEqual(value, 1.0)


class StockSelectionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.build = FactorBuild.objects.create(
            as_of="2026-09-02",
            status="succeeded",
            definition_version="stock-selection-test",
        )
        cls.quality = FactorDefinition.objects.create(
            name="Quality",
            family="style",
            model_version="stock-selection-test",
            sort_order=1,
        )
        cls.value = FactorDefinition.objects.create(
            name="Value",
            family="style",
            model_version="stock-selection-test",
            sort_order=2,
        )
        catalog = FactorModelCatalog.objects.create(
            slug="all_factors",
            name="All Factors",
            model_version="stock-selection-test",
            available_factor_count=2,
        )
        for position, factor in enumerate((cls.quality, cls.value)):
            FactorModelCatalogMembership.objects.create(
                catalog=catalog,
                factor=factor,
                position=position,
            )
        cls.securities = {}
        betas = {
            "AAA": (4.0, 0.0),
            "BBB": (2.0, 3.0),
            "CCC": (0.0, 2.0),
            "DDD": (-2.0, -1.0),
        }
        for ticker, values in betas.items():
            security = Security.objects.create(
                ticker=ticker,
                name=f"{ticker} Corp",
                sector="Technology" if ticker != "DDD" else "Financials",
            )
            cls.securities[ticker] = security
            fit = StockModelFit.objects.create(
                build=cls.build,
                security=security,
                as_of="2026-09-02",
                model_level="all_factors",
                alpha=0,
                adjusted_r2=0.6,
                residual_volatility=0.2,
                active_factor_count=2,
                design_factor_count=2,
                observation_count=756,
                coverage=1,
            )
            for factor, beta in zip((cls.quality, cls.value), values):
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

    def preview(self, **overrides):
        parameters = {
            "build": self.build,
            "model_level": "all_factors",
            "factor_weights": {"Quality": 3, "Value": 1},
            "directions": {"Quality": 1, "Value": 1},
            "top_n": 2,
        }
        parameters.update(overrides)
        return build_selection_preview(**parameters)

    def test_factor_weights_are_normalized(self):
        first = self.preview()
        second = self.preview(factor_weights={"Quality": 75, "Value": 25})
        self.assertEqual(first["final_tickers"], second["final_tickers"])
        first_scores = {row["ticker"]: row["score"] for row in first["rows"]}
        second_scores = {row["ticker"]: row["score"] for row in second["rows"]}
        for ticker, score in first_scores.items():
            self.assertAlmostEqual(score, second_scores[ticker])

    def test_one_year_of_history_is_enough_even_if_coverage_is_below_80_percent(self):
        # Sandisk-style: high score, ~half of the 756-day window filled. The
        # old 80% coverage default rejected it; one year of trading (252 days)
        # is the FactorsToday eligibility floor.
        young = Security.objects.create(
            ticker="SNDK", name="Sandisk", sector="Information Technology"
        )
        fit = StockModelFit.objects.create(
            build=self.build,
            security=young,
            as_of="2026-09-02",
            model_level="all_factors",
            alpha=0,
            adjusted_r2=0.4,
            residual_volatility=0.3,
            active_factor_count=2,
            design_factor_count=2,
            observation_count=391,
            coverage=0.517,
        )
        for factor, beta in ((self.quality, 6.0), (self.value, 4.0)):
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
        included = self.preview()
        self.assertIn("SNDK", included["final_tickers"])
        rejected = self.preview(minimum_coverage=0.8, minimum_trading_days=0)
        self.assertNotIn("SNDK", rejected["final_tickers"])

    def test_top_x_exclusion_and_manual_addition(self):
        preview = self.preview(
            excluded_tickers=["AAA"],
            manual_tickers=["DDD"],
        )
        self.assertEqual(preview["model_selected_count"], 2)
        self.assertEqual(preview["final_count"], 3)
        self.assertNotIn("AAA", preview["final_tickers"])
        manual = next(row for row in preview["included_rows"] if row["ticker"] == "DDD")
        self.assertEqual(manual["selection_source"], "manual")

    def test_equal_weight_portfolio_includes_manual_names_beyond_x(self):
        preview = self.preview(manual_tickers=["DDD"])
        portfolio = save_selection_portfolio("Weighted Selection", preview)
        holdings = list(portfolio.holdings.order_by("security__ticker"))
        self.assertEqual(len(holdings), preview["final_count"])
        for holding in holdings:
            self.assertAlmostEqual(holding.weight, 1 / preview["final_count"])

    def test_manual_addition_does_not_require_model_fit(self):
        Security.objects.create(ticker="MANUAL", name="Manual Choice")
        preview = self.preview(manual_tickers=["MANUAL"])
        manual = next(
            row for row in preview["included_rows"] if row["ticker"] == "MANUAL"
        )
        self.assertIsNone(manual["fit"])
        self.assertIsNone(manual["score"])
        self.assertEqual(manual["selection_source"], "manual")
        self.assertEqual(preview["final_count"], 3)

    def test_manual_and_excluded_conflict_is_rejected(self):
        with self.assertRaisesMessage(ValueError, "both manual and excluded"):
            self.preview(manual_tickers=["DDD"], excluded_tickers=["DDD"])


class SelectionHistoryTests(TestCase):
    def setUp(self):
        self.build = FactorBuild.objects.create(
            as_of="2025-03-25",
            status="succeeded",
            definition_version="history-test",
            mode="replication",
        )
        self.factor = FactorDefinition.objects.create(
            name="Quality",
            family="style",
            model_version="history-test",
            sort_order=1,
        )
        self.securities = {
            ticker: Security.objects.create(ticker=ticker, name=ticker)
            for ticker in ("AAA", "BBB")
        }
        for period, as_of, betas in (
            ("2025-01-01", "2025-01-31", {"AAA": 2.0, "BBB": -1.0}),
            ("2025-02-01", "2025-02-28", {"AAA": -1.0, "BBB": 2.0}),
        ):
            for ticker, beta in betas.items():
                fit = StockModelFit.objects.create(
                    build=self.build,
                    security=self.securities[ticker],
                    period=period,
                    as_of=as_of,
                    model_level="all_factors",
                    alpha=0,
                    adjusted_r2=0.6,
                    residual_volatility=0.2,
                    active_factor_count=1,
                    design_factor_count=1,
                    observation_count=252,
                    coverage=1,
                )
                StockExposureSnapshot.objects.create(
                    model_fit=fit,
                    factor=self.factor,
                    beta=beta,
                    standard_error=0.1,
                    t_stat=beta / 0.1,
                    p_value=0.01,
                    confidence_low=beta - 0.2,
                    confidence_high=beta + 0.2,
                )

    def test_replay_reselects_top_stock_each_month_and_compares_spy(self):
        dates = pd.bdate_range("2025-01-01", periods=60)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100, 112, len(dates)),
                "BBB": np.linspace(100, 106, len(dates)),
                "SPY": np.linspace(100, 108, len(dates)),
            },
            index=dates,
        )
        with patch("selection.history._prices_for_build", return_value=prices):
            result = replay_screen_history(
                self.build,
                {
                    "model_level": "all_factors",
                    "factor_weights": {"Quality": 1},
                    "directions": {"Quality": 1},
                    "top_n": 1,
                },
            )

        self.assertEqual(result["rebalances"][0]["tickers"], ["AAA"])
        self.assertEqual(result["rebalances"][1]["tickers"], ["BBB"])
        self.assertGreater(
            result["rebalances"][0]["execution_date"],
            result["rebalances"][0]["signal_date"],
        )
        self.assertIn("alpha", result["metrics"])
        self.assertEqual(result["observations"], len(result["chart_rows"]))
