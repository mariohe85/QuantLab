from datetime import timedelta

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from factors.engine import (
    factor_covariance,
    portfolio_decomposition,
    return_attribution,
)
from jobs.models import Job
from jobs.runner import claim_job, recover_stale_jobs, request_cancellation, submit_job
from market_data.models import Security
from optimization.engine import factor_risk_covariance, optimize_portfolio
from selection.services import parse_holdings_csv, save_portfolio


class GenericNumericalTests(SimpleTestCase):
    def test_decomposition_and_attribution_identities(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        exposures = pd.DataFrame({"F": [1.0, 0.5]}, index=weights.index)
        omega = pd.DataFrame([[0.04]], index=["F"], columns=["F"])
        risk = portfolio_decomposition(
            weights, exposures, omega, pd.Series({"A": 0.1, "B": 0.2})
        )
        self.assertAlmostEqual(
            risk["predicted_variance"],
            risk["factor_variance"] + risk["specific_variance"],
        )
        attribution = return_attribution(
            weights,
            exposures,
            pd.Series({"F": 0.02}),
            pd.Series({"A": 0.03, "B": 0.01}),
        )
        self.assertAlmostEqual(
            attribution["realized_return"],
            attribution["factor_return"] + attribution["residual_contribution"],
        )

    def test_factor_covariance_and_asset_risk_are_psd(self):
        rng = np.random.default_rng(8)
        returns = pd.DataFrame(rng.normal(size=(100, 2)) / 100, columns=["F1", "F2"])
        omega = factor_covariance(returns)
        exposures = pd.DataFrame(
            {"F1": [1.0, -0.2], "F2": [0.1, 0.8]}, index=["A", "B"]
        )
        covariance = factor_risk_covariance(
            exposures, omega, pd.Series({"A": 0.1, "B": 0.2})
        )
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_max_sharpe_requires_explicit_expected_returns(self):
        covariance = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
        with self.assertRaisesRegex(ValueError, "explicit expected return"):
            optimize_portfolio(covariance, objective="max_sharpe")


class PersistenceWorkflowTests(TestCase):
    def setUp(self):
        for ticker in ("A", "B", "C"):
            Security.objects.create(ticker=ticker)

    def test_portfolio_csv_create_and_equal_weight(self):
        rows = parse_holdings_csv("ticker,weight\nA,2\nB,1\n")
        portfolio = save_portfolio("Test", rows)
        weights = list(portfolio.holdings.values_list("weight", flat=True))
        self.assertAlmostEqual(sum(weights), 1)

    def test_duplicate_portfolio_tickers_rejected(self):
        rows = parse_holdings_csv("ticker,weight\nA,1\nA,1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            save_portfolio("Bad", rows)

    def test_durable_claim_cancel_and_stale_recovery(self):
        job = submit_job("screen_run", {})
        claimed = claim_job("test-worker")
        self.assertEqual(claimed.pk, job.pk)
        request_cancellation(claimed)
        claimed.refresh_from_db()
        self.assertEqual(claimed.status, "cancel_requested")
        stale = Job.objects.create(
            kind="screen_run",
            status="running",
            started_at=timezone.now() - timedelta(hours=1),
            heartbeat_at=timezone.now() - timedelta(hours=1),
            attempts=1,
            max_attempts=3,
        )
        self.assertEqual(recover_stale_jobs(), 1)
        stale.refresh_from_db()
        self.assertEqual(stale.status, "queued")
