import numpy as np
import pandas as pd
from django.test import SimpleTestCase, TestCase

from .catalog import (
    CATALOG_SPECS,
    EXTERNAL_NAMES,
    PROXY_CORE_FACTORS,
    PROXY_MODEL_VERSION,
    REVERSE_EXTERNAL_NAMES,
    SECTOR_FACTORS,
    factor_construction_details,
    sync_factor_catalogs,
)
from .engine import (
    construct_spreads,
    estimate_exposure,
    portfolio_decomposition,
    return_attribution,
    rolling_purify,
    volatility_scale,
)
from .models import FactorDefinition, FactorModelCatalog


class FactorEngineTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.index = pd.bdate_range("2020-01-01", periods=320)
        self.market = pd.Series(rng.normal(0, 0.01, 320), index=self.index)

    def test_spread_construction(self):
        returns = pd.DataFrame(
            {"LONG": self.market + 0.001, "SHORT": self.market}, index=self.index
        )
        prices = 100 * (1 + returns).cumprod()
        spread = construct_spreads(
            prices, {"Test": {"long": ["LONG"], "short": ["SHORT"]}}
        )
        self.assertAlmostEqual(spread["Test"].mean(), 0.001, places=5)

    def test_rolling_purification_removes_parent_loading(self):
        child = 1.8 * self.market + np.random.default_rng(5).normal(
            0, 0.002, len(self.market)
        )
        frame = pd.DataFrame({"Market": self.market, "Child": child})
        purified = rolling_purify(
            frame, {"Child": ["Market"]}, window=126, min_periods=63
        )
        self.assertLess(abs(purified["Child"].dropna().corr(self.market)), 0.15)

    def test_volatility_scaling_is_lagged(self):
        frame = pd.DataFrame(
            {"factor": np.r_[np.repeat(0.001, 80), 0.2]}, index=self.index[:81]
        )
        scaled = volatility_scale(frame, window=40)
        prior = frame.iloc[:-1].std() * np.sqrt(252)
        expected_scale = min(5.0, 0.10 / prior["factor"])
        self.assertAlmostEqual(scaled.iloc[-1, 0], 0.2 * expected_scale)

    def test_elastic_net_then_hac_refit(self):
        noise = np.random.default_rng(9).normal(0, 0.003, len(self.market))
        factors = pd.DataFrame(
            {"Market": self.market, "Noise": noise}, index=self.index
        )
        stock = pd.Series(1.2 * self.market + noise, index=self.index, name="ABC")
        result = estimate_exposure(stock, factors, alpha=1e-6)
        self.assertIn("se", result.inference["hac"])
        self.assertAlmostEqual(result.beta.loc["ABC", "Market"], 1.2, delta=0.1)

    def test_risk_decomposition_identity(self):
        weights = pd.Series({"A": 0.6, "B": 0.4})
        exposures = pd.DataFrame({"F": [1.0, 0.5]}, index=weights.index)
        covariance = pd.DataFrame([[0.04]], index=["F"], columns=["F"])
        result = portfolio_decomposition(
            weights, exposures, covariance, pd.Series({"A": 0.1, "B": 0.2})
        )
        self.assertAlmostEqual(
            result["predicted_variance"],
            result["factor_variance"] + result["specific_variance"],
        )

    def test_return_attribution_identity(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        exposures = pd.DataFrame({"F": [1.0, 0.0]}, index=weights.index)
        result = return_attribution(
            weights,
            exposures,
            pd.Series({"F": 0.02}),
            pd.Series({"A": 0.03, "B": 0.01}),
        )
        self.assertAlmostEqual(
            result["realized_return"],
            result["factor_return"] + result["residual_contribution"],
        )


class V2CatalogTests(TestCase):
    def test_default_sync_builds_complete_v2_catalogs(self):
        catalogs = sync_factor_catalogs()
        self.assertEqual(set(catalogs), set(CATALOG_SPECS))
        self.assertTrue(
            all(
                catalog.model_version == PROXY_MODEL_VERSION
                for catalog in catalogs.values()
            )
        )
        self.assertEqual(
            FactorDefinition.objects.filter(model_version=PROXY_MODEL_VERSION).count(),
            CATALOG_SPECS["all_factors"].expected_count,
        )

    def test_catalog_sync_is_idempotent(self):
        sync_factor_catalogs()
        sync_factor_catalogs()
        self.assertEqual(
            FactorModelCatalog.objects.filter(
                model_version=PROXY_MODEL_VERSION
            ).count(),
            len(CATALOG_SPECS),
        )
        self.assertEqual(
            FactorDefinition.objects.filter(model_version=PROXY_MODEL_VERSION).count(),
            CATALOG_SPECS["all_factors"].expected_count,
        )

    def test_catalog_memberships_are_nested_in_declared_order(self):
        catalogs = sync_factor_catalogs()
        memberships = {
            slug: tuple(
                catalog.memberships.order_by("position").values_list(
                    "factor__name", flat=True
                )
            )
            for slug, catalog in catalogs.items()
        }
        self.assertEqual(memberships["base"], PROXY_CORE_FACTORS)
        self.assertEqual(
            memberships["base_sector"], (*PROXY_CORE_FACTORS, *SECTOR_FACTORS)
        )
        self.assertTrue(
            set(memberships["base_sector"]).issubset(
                memberships["base_sector_industry"]
            )
        )
        self.assertTrue(
            set(memberships["base_sector_industry"]).issubset(
                memberships["all_factors"]
            )
        )

    def test_external_names_are_reversible(self):
        for local, external in EXTERNAL_NAMES.items():
            self.assertEqual(REVERSE_EXTERNAL_NAMES[external], local)

    def test_non_v2_catalog_sync_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported factor model version"):
            sync_factor_catalogs("legacy")

    def test_construction_metadata_describes_each_factor_pattern(self):
        market = factor_construction_details("Market")
        self.assertEqual(market["kind"], "direct_etf")
        self.assertEqual(market["long_leg"], "IVV")
        self.assertIsNone(market["volatility_target"])

        credit = factor_construction_details("CreditRisk")
        self.assertEqual(credit["kind"], "long_short")
        self.assertEqual((credit["long_leg"], credit["short_leg"]), ("JNK", "IEF"))

        value = factor_construction_details("Value")
        self.assertEqual(value["kind"], "residualized_style")
        self.assertEqual(value["purified_against"], ["RPG"])
        self.assertEqual(value["rolling_window"], "156 weeks")

        sector = factor_construction_details("InformationTechnology")
        self.assertEqual(sector["kind"], "residualized_sector")
        self.assertEqual(sector["long_leg"], "XLK")

        theme = factor_construction_details("Theme: AI Capex & Data Centers")
        self.assertEqual(theme["kind"], "residualized_thematic_basket")
        self.assertIn("NVDA", theme["basket"])

    def test_catalog_sync_persists_detailed_construction(self):
        sync_factor_catalogs()
        value = FactorDefinition.objects.get(
            name="Value", model_version=PROXY_MODEL_VERSION
        )
        self.assertIn("RPV", value.description)
        self.assertEqual(
            value.configuration["construction"]["kind"], "residualized_style"
        )
