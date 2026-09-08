import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .engine import optimize_portfolio
from .forecasts import (
    equal_sharpe_expected_returns,
    factor_implied_expected_returns,
    shrink_annualized_expected_returns,
    shrink_betas_to_zero,
    shrink_factor_premia,
)


class OptimizerTests(SimpleTestCase):
    def setUp(self):
        self.names = ["A", "B", "C", "D"]
        self.covariance = pd.DataFrame(
            np.diag([0.04, 0.05, 0.06, 0.08]), index=self.names, columns=self.names
        )
        self.exposures = pd.DataFrame(
            {"Market": [0.5, 0.9, 1.2, 1.5]}, index=self.names
        )
        self.sectors = pd.Series(["One", "One", "Two", "Two"], index=self.names)

    def test_constraints_and_factor_beta(self):
        result = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            max_weight=0.4,
            sectors=self.sectors,
            sector_bounds={"One": (0.3, 0.7), "Two": (0.3, 0.7)},
            factor_exposures=self.exposures,
            factor_bounds={"Market": (0.8, 1.1)},
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.weights.sum(), 1)
        beta = float(result.weights @ self.exposures["Market"])
        self.assertGreaterEqual(beta, 0.8 - 1e-5)
        self.assertLessEqual(beta, 1.1 + 1e-5)

    def test_all_objectives_produce_fully_invested_portfolios(self):
        expected = pd.Series([0.10, 0.08, 0.12, 0.09], index=self.names)
        for objective in [
            "equal_weight",
            "min_variance",
            "max_return",
            "mean_variance",
            "max_sharpe",
            "risk_parity",
        ]:
            with self.subTest(objective=objective):
                result = optimize_portfolio(
                    self.covariance, expected, objective, max_weight=0.6
                )
                self.assertTrue(result.success, result.message)
                self.assertAlmostEqual(result.weights.sum(), 1, places=5)

    def test_max_return_uses_the_highest_forecasts_up_to_bounds(self):
        expected = pd.Series([0.20, 0.10, 0.05, 0.01], index=self.names)
        result = optimize_portfolio(
            self.covariance,
            expected,
            "max_return",
            max_weight=0.4,
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.weights["A"], 0.4, places=5)
        self.assertAlmostEqual(result.weights["B"], 0.4, places=5)

    def test_absolute_and_relative_stock_weight_bounds(self):
        previous = pd.Series([0.4, 0.3, 0.2, 0.1], index=self.names)
        absolute = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            previous_weights=previous,
            min_weight=0.1,
            max_weight=0.4,
        )
        self.assertTrue(absolute.success, absolute.message)
        self.assertTrue(absolute.weights.between(0.1 - 1e-7, 0.4 + 1e-7).all())

        relative = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            previous_weights=previous,
            weight_constraint_mode="relative",
            min_weight=0.05,
            max_weight=0.08,
            max_weight_cap=0.42,
        )
        self.assertTrue(relative.success, relative.message)
        lower = (previous - 0.05).clip(lower=0)
        upper = (previous + 0.08).clip(upper=0.42)
        self.assertTrue((relative.weights >= lower - 1e-7).all())
        self.assertTrue((relative.weights <= upper + 1e-7).all())
        self.assertEqual(relative.diagnostics["weight_constraint_mode"], "relative")

    def test_infeasible_stock_bounds_are_rejected_before_solving(self):
        with self.assertRaisesRegex(ValueError, "fully invested"):
            optimize_portfolio(
                self.covariance,
                objective="min_variance",
                min_weight=0,
                max_weight=0.2,
            )

    def test_return_objectives_require_expected_returns(self):
        for objective in ("max_return", "max_sharpe", "mean_variance"):
            with self.subTest(objective=objective), self.assertRaisesRegex(
                ValueError, "expected return"
            ):
                optimize_portfolio(
                    self.covariance,
                    objective=objective,
                    max_weight=0.6,
                )

    def test_mean_variance_default_is_return_minus_half_variance(self):
        expected = pd.Series([0.20, 0.10, 0.05, 0.01], index=self.names)
        default = optimize_portfolio(
            self.covariance,
            expected,
            objective="mean_variance",
            max_weight=0.6,
        )
        heavier_risk = optimize_portfolio(
            self.covariance,
            expected,
            objective="mean_variance",
            max_weight=0.6,
            risk_aversion=3,
        )
        self.assertTrue(default.success, default.message)
        self.assertTrue(heavier_risk.success, heavier_risk.message)
        self.assertGreater(default.weights @ expected, heavier_risk.weights @ expected)
        self.assertGreater(
            default.diagnostics["ex_ante_volatility"],
            heavier_risk.diagnostics["ex_ante_volatility"],
        )

    def test_max_sharpe_uses_excess_return(self):
        expected = pd.Series([0.04, 0.06, 0.08, 0.10], index=self.names)
        result = optimize_portfolio(
            self.covariance,
            expected,
            objective="max_sharpe",
            max_weight=0.6,
            risk_free_rate=0.04,
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.diagnostics["risk_free_rate"], 0.04)
        self.assertAlmostEqual(
            result.diagnostics["excess_return"],
            result.diagnostics["expected_return"] - 0.04,
        )

    def test_equal_sharpe_max_sharpe_matches_maximum_diversification(self):
        covariance = pd.DataFrame(
            [[0.04, 0.008, 0.002], [0.008, 0.09, 0.012], [0.002, 0.012, 0.16]],
            index=self.names[:3],
            columns=self.names[:3],
        )
        forecast = equal_sharpe_expected_returns(
            covariance, risk_free_rate=0.04, common_sharpe=0.5
        )
        other = equal_sharpe_expected_returns(
            covariance, risk_free_rate=0.01, common_sharpe=2.0
        )
        result = optimize_portfolio(
            covariance,
            forecast.expected_returns,
            objective="max_sharpe",
            min_weight=0,
            max_weight=1,
            risk_free_rate=0.04,
        )
        other_result = optimize_portfolio(
            covariance,
            other.expected_returns,
            objective="max_sharpe",
            min_weight=0,
            max_weight=1,
            risk_free_rate=0.01,
        )
        self.assertTrue(result.success, result.message)
        sigma = np.sqrt(np.diag(covariance.to_numpy()))
        target = np.linalg.inv(covariance.to_numpy()) @ sigma
        target = pd.Series(target / target.sum(), index=covariance.index)
        pd.testing.assert_series_equal(
            result.weights, target, atol=1e-5, check_names=False
        )
        pd.testing.assert_series_equal(
            result.weights, other_result.weights, atol=1e-5, check_names=False
        )

    def test_turnover_cap(self):
        previous = pd.Series([0.4, 0.3, 0.2, 0.1], index=self.names)
        result = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            previous_weights=previous,
            turnover_cap=0.10,
            max_weight=0.6,
        )
        self.assertTrue(result.success, result.message)
        self.assertLessEqual((result.weights - previous).abs().sum(), 0.10001)

    def test_relative_beta_constraint(self):
        benchmark = pd.Series(0.25, index=self.names)
        result = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            max_weight=0.6,
            factor_exposures=self.exposures,
            benchmark_weights=benchmark,
            relative_factor_bounds={"Market": (-0.02, 0.02)},
        )
        self.assertTrue(result.success, result.message)
        difference = float((result.weights - benchmark) @ self.exposures["Market"])
        self.assertLessEqual(abs(difference), 0.02001)

    def test_factor_variance_constraint(self):
        factor_covariance = pd.DataFrame([[0.04]], index=["Market"], columns=["Market"])
        result = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            max_weight=0.6,
            factor_exposures=self.exposures,
            factor_covariance=factor_covariance,
            max_factor_variance=0.04,
        )
        self.assertTrue(result.success, result.message)
        beta = float(result.weights @ self.exposures["Market"])
        self.assertLessEqual(beta**2 * 0.04, 0.040001)

    def test_named_constraint_slack_is_reported(self):
        result = optimize_portfolio(
            self.covariance,
            objective="min_variance",
            max_weight=0.6,
            factor_exposures=self.exposures,
            factor_bounds={"Market": (0.8, 1.1)},
        )
        self.assertIn("factor:Market:maximum", result.diagnostics["constraint_slack"])
        self.assertGreaterEqual(
            result.diagnostics["constraint_slack"]["factor:Market:maximum"],
            -1e-5,
        )


class ForecastEstimatorTests(SimpleTestCase):
    def test_factor_implied_returns_add_cash_and_factor_contributions(self):
        betas = pd.DataFrame(
            {"Market": [0.5, 1.2], "Value": [-0.2, 0.4]},
            index=["A", "B"],
        )
        result = factor_implied_expected_returns(
            betas,
            {"Market": 0.05, "Value": 0.02},
            risk_free_rate=0.04,
        )
        self.assertAlmostEqual(result.expected_returns["A"], 0.061)
        self.assertAlmostEqual(result.expected_returns["B"], 0.108)
        pd.testing.assert_series_equal(
            result.annualized_contributions.sum(axis=1) + 0.04,
            result.expected_returns,
        )
        self.assertGreater(result.dispersion, 0)

    def _premium_returns(self, precise_volatility=0.001, noisy_volatility=0.02):
        generator = np.random.default_rng(3)
        periods = 756
        return pd.DataFrame(
            {
                "Precise": generator.normal(0.0004, precise_volatility, periods),
                "Noisy": generator.normal(0.0004, noisy_volatility, periods),
            },
            index=pd.bdate_range("2021-01-04", periods=periods),
        )

    def test_factor_premia_move_toward_history_in_proportion_to_precision(self):
        result = shrink_factor_premia(
            self._premium_returns(),
            {"Precise": 0.02, "Noisy": 0.02},
            prior_dispersion=0.03,
        )
        self.assertGreater(
            result.shrinkage_weights["Precise"], result.shrinkage_weights["Noisy"]
        )
        pd.testing.assert_series_equal(
            result.premia,
            0.02 + result.shrinkage_weights * (result.sample_premia - 0.02),
        )
        # The imprecise factor cannot earn its way away from the assumption.
        self.assertAlmostEqual(result.premia["Noisy"], 0.02, places=2)
        self.assertGreater(result.premia["Precise"], 0.05)

    def test_zero_prior_dispersion_pins_premia_to_assumptions(self):
        result = shrink_factor_premia(
            self._premium_returns(),
            {"Precise": 0.02, "Noisy": 0.03},
            prior_dispersion=0,
        )
        self.assertEqual(result.premia["Precise"], 0.02)
        self.assertEqual(result.premia["Noisy"], 0.03)

    def test_factor_premia_reject_unknown_factors_and_thin_history(self):
        returns = self._premium_returns()
        with self.assertRaisesRegex(ValueError, "Unknown factor"):
            shrink_factor_premia(returns, {"Missing": 0.02})
        with self.assertRaisesRegex(ValueError, "Insufficient factor history"):
            shrink_factor_premia(returns.head(120), {"Precise": 0.02})
        with self.assertRaisesRegex(ValueError, "prior_dispersion"):
            shrink_factor_premia(returns, {"Precise": 0.02}, prior_dispersion=-1)

    def test_factor_implied_returns_reject_unknown_or_empty_premia(self):
        betas = pd.DataFrame({"Market": [0.5, 1.2]}, index=["A", "B"])
        with self.assertRaisesRegex(ValueError, "Unknown factor"):
            factor_implied_expected_returns(betas, {"Missing": 0.05})
        with self.assertRaisesRegex(ValueError, "non-zero"):
            factor_implied_expected_returns(betas, {"Market": 0.0})

    def test_equal_sharpe_returns_are_vol_proportional(self):
        covariance = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"]
        )
        result = equal_sharpe_expected_returns(
            covariance, risk_free_rate=0.04, common_sharpe=0.5
        )
        self.assertAlmostEqual(result.expected_returns["A"], 0.14)
        self.assertAlmostEqual(result.expected_returns["B"], 0.19)
        with self.assertRaisesRegex(ValueError, "positive"):
            equal_sharpe_expected_returns(covariance, common_sharpe=0)

    def test_expected_returns_and_betas_are_shrunk(self):
        index = pd.bdate_range("2020-01-01", periods=120)
        returns = pd.DataFrame(
            {
                "A": np.linspace(-0.01, 0.012, len(index)),
                "B": np.linspace(-0.009, 0.011, len(index)),
                "C": np.sin(np.arange(len(index))) * 0.03,
            },
            index=index,
        )
        estimate = shrink_annualized_expected_returns(returns)
        for ticker in returns:
            self.assertLessEqual(estimate.shrinkage_weights[ticker], 1)
            self.assertGreaterEqual(estimate.shrinkage_weights[ticker], 0)
        betas = pd.DataFrame({"Oil": [1.0, 0.5]}, index=["A", "B"])
        errors = pd.DataFrame({"Oil": [0.1, 1.0]}, index=["A", "B"])
        shrunk = shrink_betas_to_zero(betas, errors)
        self.assertGreater(
            abs(shrunk.betas.at["A", "Oil"]), abs(shrunk.betas.at["B", "Oil"])
        )

    def test_unidentified_betas_do_not_zero_the_cross_section(self):
        betas = pd.DataFrame({"Oil": [0.9, -0.8, 0.0]}, index=["A", "B", "C"])
        errors = pd.DataFrame({"Oil": [0.1, 0.12, 1_000_000.0]}, index=["A", "B", "C"])
        shrunk = shrink_betas_to_zero(betas, errors)
        self.assertEqual(shrunk.informative_counts["Oil"], 2)
        self.assertGreater(shrunk.prior_variance["Oil"], 0)
        self.assertGreater(abs(shrunk.betas.at["A", "Oil"]), 0.5)
        self.assertGreater(abs(shrunk.betas.at["B", "Oil"]), 0.5)
        self.assertEqual(shrunk.betas.at["C", "Oil"], 0)
