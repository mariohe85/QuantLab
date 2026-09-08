from django.contrib import admin
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from backtests.models import BacktestArtifact, BacktestRebalance, BacktestRun
from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorEquation,
    FactorModelCatalog,
    FactorModelCatalogMembership,
    FactorModelRun,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)
from jobs.models import Job
from market_data.models import (
    Artifact,
    DataSnapshot,
    PriceBar,
    PriceSnapshot,
    RiskFreeRate,
    Security,
    Universe,
    UniverseSnapshot,
)
from optimization.models import (
    OptimizationHolding,
    OptimizationRun,
    OptimizationScenario,
    OptimizationStudy,
)
from selection.models import (
    BreachEvent,
    Portfolio,
    PortfolioHolding,
    PortfolioRiskSnapshot,
    PortfolioSnapshot,
    RiskLimit,
    ScreenDefinition,
    ScreenResult,
    ScreenRun,
    StrategyDefinition,
)

RESEARCH_MODELS = (
    Artifact,
    BacktestArtifact,
    BacktestRebalance,
    BacktestRun,
    BreachEvent,
    DataSnapshot,
    FactorBuild,
    FactorDefinition,
    FactorEquation,
    FactorModelCatalog,
    FactorModelCatalogMembership,
    FactorModelRun,
    FactorObservation,
    Job,
    OptimizationHolding,
    OptimizationRun,
    OptimizationScenario,
    OptimizationStudy,
    Portfolio,
    PortfolioHolding,
    PortfolioRiskSnapshot,
    PortfolioSnapshot,
    PriceBar,
    PriceSnapshot,
    RiskFreeRate,
    RiskLimit,
    ScreenDefinition,
    ScreenResult,
    ScreenRun,
    Security,
    StockExposureSnapshot,
    StockModelFit,
    StrategyDefinition,
    Universe,
    UniverseSnapshot,
)


class AdminAccessTests(TestCase):
    def test_research_models_are_registered(self):
        registered = set(admin.site._registry)
        missing = [model.__name__ for model in RESEARCH_MODELS if model not in registered]
        self.assertEqual(missing, [])

    def test_superuser_can_open_admin_and_large_changelists(self):
        user = get_user_model().objects.create_superuser("admin", password="quantlab")
        self.client.force_login(user)
        index = self.client.get("/admin/")
        self.assertEqual(index.status_code, 200)
        self.assertContains(index, "Price bars")
        self.assertContains(index, "Stock exposure snapshots")
        for model in (Security, PriceBar, FactorObservation, StockExposureSnapshot):
            opts = model._meta
            url = reverse(f"admin:{opts.app_label}_{opts.model_name}_changelist")
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, url)

    def test_ensure_local_admin_creates_staff_login(self):
        call_command("ensure_local_admin", verbosity=0)
        user = get_user_model().objects.get(username="admin")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.check_password("quantlab"))
        logged_in = self.client.login(username="admin", password="quantlab")
        self.assertTrue(logged_in)
