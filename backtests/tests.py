import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .engine import performance_statistics, selection_statistics, walk_forward
from .factor_strategy import monthly_event_backtest


def momentum_allocator(history):
    scores = history.pct_change(fill_method=None).iloc[-20:].mean()
    winner = scores.idxmax()
    return pd.Series({winner: 1.0})


class BacktestTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.dates = pd.bdate_range("2018-01-01", periods=420)
        returns = pd.DataFrame(
            rng.normal(0.0003, 0.01, (420, 4)), index=self.dates, columns=list("ABCD")
        )
        self.prices = 100 * (1 + returns).cumprod()

    def test_no_future_data_changes_prior_results(self):
        original = walk_forward(
            self.prices, momentum_allocator, estimation_window=80, rebalance_every=20
        )
        changed = self.prices.copy()
        cutoff = 300
        changed.iloc[cutoff:] *= np.linspace(1, 50, len(changed) - cutoff)[:, None]
        modified = walk_forward(
            changed, momentum_allocator, estimation_window=80, rebalance_every=20
        )
        pd.testing.assert_series_equal(
            original.returns.iloc[: cutoff - 80], modified.returns.iloc[: cutoff - 80]
        )

    def test_execution_lag_and_costs(self):
        result = walk_forward(
            self.prices,
            momentum_allocator,
            estimation_window=80,
            rebalance_every=20,
            execution_lag=1,
            transaction_cost_bps=10,
        )
        self.assertGreater(result.turnover.sum(), 0)
        self.assertAlmostEqual(result.costs.sum(), result.turnover.sum() * 0.001)

    def test_execution_applies_once_without_rewriting_weights(self):
        def fixed(_history):
            return pd.Series({"A": 0.6, "B": 0.4})

        result = walk_forward(
            self.prices, fixed, estimation_window=80, rebalance_every=100
        )
        expected = pd.Series(
            {"A": 0.6, "B": 0.4, "C": 0.0, "D": 0.0}, name=self.dates[80]
        )
        pd.testing.assert_series_equal(result.weights.iloc[80], expected)

    def test_robust_metrics(self):
        metrics = performance_statistics(
            self.prices["A"].pct_change().dropna(), bootstrap_samples=50
        )
        for key in [
            "max_drawdown",
            "daily_var_95_loss",
            "daily_cvar_95_loss",
            "hac_mean_p_value",
            "bootstrap_annual_mean_ci",
            "probabilistic_sharpe",
        ]:
            self.assertIn(key, metrics)
        self.assertGreater(metrics["daily_var_95_loss"], 0)

    def test_benchmark_relative_metrics(self):
        returns = self.prices["A"].pct_change().dropna()
        benchmark = self.prices["B"].pct_change().dropna()
        metrics = performance_statistics(returns, benchmark, bootstrap_samples=25)
        for key in ("alpha", "beta", "tracking_error", "information_ratio"):
            self.assertIn(key, metrics)

    def test_selection_ic_and_quantile_spread(self):
        scores = pd.DataFrame([np.arange(20)] * 10, index=self.dates[:10])
        forward = scores / 100
        stats = selection_statistics(scores, forward)
        self.assertAlmostEqual(stats["rank_ic"], 1.0)
        self.assertGreater(stats["top_minus_bottom"], 0)

    def test_zero_execution_lag_is_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward(self.prices, momentum_allocator, execution_lag=0)

    def test_monthly_event_backtest_executes_after_signal_date(self):
        result = monthly_event_backtest(
            self.prices,
            lambda history: pd.Series({"A": 0.5, "B": 0.5}),
            estimation_window=80,
            transaction_cost_bps=10,
        )
        trade_dates = result.turnover[result.turnover > 0].index
        self.assertGreater(len(trade_dates), 1)
        self.assertTrue(
            all(
                self.dates[self.dates.get_loc(stamp) - 1].month != stamp.month
                for stamp in trade_dates
            )
        )
