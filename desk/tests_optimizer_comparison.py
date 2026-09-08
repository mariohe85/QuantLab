from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.db.models import Model
from django.test import SimpleTestCase

from optimization.engine import OptimizationResult
from selection.history import _simulate

from .optimizer_comparison import (
    METHODS,
    compare_portfolio_optimizers,
    expected_returns_for_method,
    optimized_decisions,
)


class OptimizerComparisonTests(SimpleTestCase):
    def setUp(self):
        self.names = ["AAA", "BBB", "CCC"]
        self.covariance = pd.DataFrame(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.06, 0.008],
                [0.005, 0.008, 0.09],
            ],
            index=self.names,
            columns=self.names,
        )
        dates = pd.bdate_range("2020-01-01", periods=300)
        self.inputs = {
            "names": self.names,
            "stock_returns": pd.DataFrame(
                {
                    "AAA": np.linspace(-0.01, 0.012, len(dates)),
                    "BBB": np.linspace(-0.008, 0.009, len(dates)),
                    "CCC": np.linspace(-0.006, 0.007, len(dates)),
                },
                index=dates,
            ),
            "factor_returns": pd.DataFrame(
                {"Growth": np.linspace(-0.01, 0.011, len(dates))},
                index=dates,
            ),
            "exposures": pd.DataFrame({"Growth": [0.5, 1.0, 1.5]}, index=self.names),
            "standard_errors": pd.DataFrame(
                {"Growth": [0.1, 0.1, 0.1]}, index=self.names
            ),
            "covariance": self.covariance,
            "risk_free_rate": 0.03,
        }

    def test_all_forecast_methods_cover_every_holding(self):
        for method in METHODS[1:]:
            with self.subTest(method=method.key):
                expected = expected_returns_for_method(
                    method, self.inputs, {"factor_premia": {"Growth": 0.02}}
                )
                if method.objective == "min_variance":
                    self.assertIsNone(expected)
                else:
                    self.assertEqual(list(expected.index), self.names)
                    self.assertTrue(np.isfinite(expected).all())

    @patch("desk.optimizer_comparison.optimize_portfolio")
    @patch("desk.optimizer_comparison._risk_inputs_for_tickers")
    def test_each_rebalance_uses_only_its_signal_date(self, risk_inputs, optimize):
        dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
        decisions = [{"signal_date": stamp, "tickers": self.names} for stamp in dates]
        risk_inputs.return_value = {
            **self.inputs,
            "latest": {},
            "original": pd.Series(1 / 3, index=self.names),
            "spy_exposure": None,
            "factor_covariance": pd.DataFrame(
                [[0.04]], index=["Growth"], columns=["Growth"]
            ),
            "specific": pd.Series(0.1, index=self.names),
            "risk_model": "factor_model",
        }
        optimize.return_value = OptimizationResult(
            weights=pd.Series(1 / 3, index=self.names),
            success=True,
            message="ok",
            diagnostics={},
        )

        paths, _warnings = optimized_decisions(
            build=SimpleNamespace(),
            dataset=SimpleNamespace(),
            decisions=decisions,
            model_level="all_factors",
            parameters={},
        )

        self.assertEqual(
            [call.kwargs["as_of"] for call in risk_inputs.call_args_list], dates
        )
        self.assertEqual(len(paths["raw_mean_max_sharpe"]), 2)

    def test_equal_weight_executes_next_day(self):
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = pd.DataFrame(
            {
                "AAA": 100 * np.cumprod(np.full(len(dates), 1.01)),
                "BBB": np.full(len(dates), 100.0),
                "SPY": 100 * np.cumprod(np.full(len(dates), 1.002)),
            },
            index=dates,
        )
        result = _simulate(
            prices,
            [{"signal_date": dates[0], "tickers": ["AAA", "BBB"]}],
        )

        self.assertEqual(result["rebalances"][0]["execution_date"], dates[1].date())
        self.assertIsNone(result["rebalances"][0]["weights"])
        self.assertAlmostEqual(
            result["chart_rows"][0]["portfolio_equity"], 1.005, places=8
        )

    @patch.object(Model, "save")
    @patch("desk.optimizer_comparison.optimized_decisions")
    @patch("desk.optimizer_comparison._prices_for_build")
    @patch("desk.optimizer_comparison._dataset_for_build")
    @patch("desk.optimizer_comparison.monthly_selection_decisions")
    def test_comparison_does_not_save_models(
        self,
        monthly_decisions,
        dataset_for_build,
        prices_for_build,
        make_paths,
        model_save,
    ):
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = pd.DataFrame(
            {
                "AAA": 100 * np.cumprod(np.full(len(dates), 1.001)),
                "BBB": 100 * np.cumprod(np.full(len(dates), 1.0005)),
                "SPY": 100 * np.cumprod(np.full(len(dates), 1.0008)),
            },
            index=dates,
        )
        decisions = [{"signal_date": dates[0], "tickers": ["AAA", "BBB"]}]
        monthly_decisions.return_value = decisions
        dataset_for_build.return_value = SimpleNamespace(prices=prices)
        prices_for_build.return_value = prices
        make_paths.return_value = (
            {method.key: list(decisions) for method in METHODS},
            {method.key: [] for method in METHODS},
        )
        portfolio = SimpleNamespace(
            name="Strong growth Portfolio",
            configuration={"stock_selection": {"model_level": "all_factors"}},
        )

        result = compare_portfolio_optimizers(
            portfolio, SimpleNamespace(pk=1), months=24
        )

        self.assertEqual(len(result["rows"]), len(METHODS))
        model_save.assert_not_called()
