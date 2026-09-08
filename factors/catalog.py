from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction

from .models import (
    FactorDefinition,
    FactorModelCatalog,
    FactorModelCatalogMembership,
)
from .proxyfactorlib import (
    COUNTRY_ETFS,
    INDUSTRY_ETFS,
    MACRO_SPECS,
    PROXY_MODEL_VERSION,
    SECTOR_ETFS,
    STYLE_SPECS,
    THEMATIC_FACTORS,
    THEME_BASKETS,
)

EXTERNAL_NAMES = {
    "Oil": "OilPrice",
    "Gold": "GoldPrice",
    "Rates": "InterestRate",
    "USD": "USDollar",
    "InformationTechnology": "Sector: Technology",
    "HealthCare": "Sector: Health Care",
    "ConsumerDiscretionary": "Sector: Consumer Discretionary",
    "ConsumerStaples": "Sector: Consumer Staples",
    "CommunicationServices": "Sector: Communication Services",
    "RealEstate": "Sector: Real Estate",
    "Energy": "Sector: Energy",
    "Financials": "Sector: Financials",
    "Industrials": "Sector: Industrials",
    "Materials": "Sector: Materials",
    "Utilities": "Sector: Utilities",
}
REVERSE_EXTERNAL_NAMES = {external: local for local, external in EXTERNAL_NAMES.items()}
SECTOR_FACTORS = (
    "CommunicationServices",
    "ConsumerDiscretionary",
    "ConsumerStaples",
    "Energy",
    "Financials",
    "HealthCare",
    "Industrials",
    "InformationTechnology",
    "Materials",
    "RealEstate",
    "Utilities",
)
INDUSTRY_FACTORS = (
    "Industry: Semiconductors",
    "Industry: Software",
    "Industry: Biotechnology",
    "Industry: Medical Devices",
    "Industry: Regional Banks",
    "Industry: Aerospace & Defense",
    "Industry: Home Construction",
    "Industry: Retail",
    "Industry: Gold Miners",
    "Industry: Clean Energy",
)
COUNTRY_FACTORS = (
    "Country: Japan",
    "Country: Germany",
    "Country: United Kingdom",
    "Country: China",
    "Country: Brazil",
    "Country: Canada",
    "Country: India",
    "Country: South Korea",
    "Country: Taiwan",
    "Country: Australia",
)
PROXY_CORE_FACTORS = (
    "Market",
    "Rates",
    "Oil",
    "Gold",
    "USD",
    "CreditRisk",
    "SmallSize",
    "Momentum",
    "Value",
    "LowVolatility",
    "Quality",
    "Growth",
    "BetaFactor",
    "DividendYield",
    "Liquidity",
)
PROXY_AVAILABLE_FACTORS = (
    *PROXY_CORE_FACTORS,
    *SECTOR_FACTORS,
    *INDUSTRY_FACTORS,
    *COUNTRY_FACTORS,
    *THEMATIC_FACTORS,
)
PROXY_FUNDAMENTALS = {
    "SmallSize",
    "Momentum",
    "LowVolatility",
    "BetaFactor",
    "Liquidity",
}


@dataclass(frozen=True)
class CatalogSpec:
    name: str
    factors: tuple[str, ...]
    expected_count: int
    complete: bool
    missing: str


CATALOG_SPECS = {
    "base": CatalogSpec("Base", PROXY_CORE_FACTORS, 15, True, ""),
    "base_sector": CatalogSpec(
        "Base + Sector",
        (*PROXY_CORE_FACTORS, *SECTOR_FACTORS),
        26,
        True,
        "",
    ),
    "base_sector_industry": CatalogSpec(
        "Base + Sector + Industry",
        (*PROXY_CORE_FACTORS, *SECTOR_FACTORS, *INDUSTRY_FACTORS),
        36,
        True,
        "",
    ),
    "all_factors": CatalogSpec(
        "All Factors + Themes",
        PROXY_AVAILABLE_FACTORS,
        56,
        True,
        "Self-contained proxy hierarchy with ten fixed thematic baskets.",
    ),
}


def factor_construction_details(name: str) -> dict:
    """Return user-facing construction metadata from the executable V2 specs."""
    external_name = EXTERNAL_NAMES.get(name, name)
    methodology = {
        "return_source": "local adjusted prices",
        "estimation_frequency": None,
        "rolling_window": None,
    }
    if external_name in MACRO_SPECS:
        spec = MACRO_SPECS[external_name]
        long_leg = spec["long"]
        short_leg = spec.get("short")
        inverted = spec.get("sign") == -1
        if short_leg:
            kind = "long_short"
            summary = (
                f"Volatility-scaled return spread: long {long_leg}, short {short_leg}."
            )
        elif inverted:
            kind = "inverted_etf"
            summary = (
                f"Inverse {long_leg} ETF return, so positive values represent falling "
                "Treasury prices and rising rates."
            )
        else:
            kind = "direct_etf"
            summary = f"Direct {long_leg} ETF return proxy."
        volatility_target = None if spec["kind"] == "raw" else 0.10
        if volatility_target:
            summary += " Scaled to 10% annualized volatility using lagged estimates."
        return {
            **methodology,
            "kind": kind,
            "summary": summary,
            "long_leg": long_leg,
            "short_leg": short_leg,
            "inverted": inverted,
            "purified_against": [],
            "basket": [],
            "volatility_target": volatility_target,
        }

    if external_name in STYLE_SPECS:
        long_leg, hedge = STYLE_SPECS[external_name]
        return {
            **methodology,
            "estimation_frequency": "Friday-compounded weekly returns",
            "rolling_window": "156 weeks",
            "kind": "residualized_style",
            "summary": (
                f"Residual return of {long_leg} after removing its rolling exposure "
                f"to {hedge}; this is not a simple static spread. Scaled to 10% "
                "annualized volatility."
            ),
            "long_leg": long_leg,
            "short_leg": None,
            "inverted": False,
            "purified_against": [hedge],
            "basket": [],
            "volatility_target": 0.10,
        }

    layer_specs = (
        (SECTOR_ETFS, "core factors", "residualized_sector"),
        (INDUSTRY_ETFS, "core and sector factors", "residualized_industry"),
        (
            COUNTRY_ETFS,
            "core, sector, and industry factors",
            "residualized_country",
        ),
    )
    for specifications, stack_label, kind in layer_specs:
        if external_name in specifications:
            etf = specifications[external_name]
            return {
                **methodology,
                "estimation_frequency": "Friday-compounded weekly returns",
                "rolling_window": "156 weeks",
                "kind": kind,
                "summary": (
                    f"Residual return of the {etf} ETF after removing rolling exposure "
                    f"to the locally built {stack_label}. Scaled to 10% annualized "
                    "volatility."
                ),
                "long_leg": etf,
                "short_leg": None,
                "inverted": False,
                "purified_against": [stack_label],
                "basket": [],
                "volatility_target": 0.10,
            }

    if external_name in THEME_BASKETS:
        basket = list(THEME_BASKETS[external_name])
        return {
            **methodology,
            "estimation_frequency": "Friday-compounded weekly returns",
            "rolling_window": "156 weeks",
            "kind": "residualized_thematic_basket",
            "summary": (
                "Fixed equal-weight basket residualized against the complete local "
                "core, sector, industry, and country factor stack, then scaled to "
                "10% annualized volatility."
            ),
            "long_leg": None,
            "short_leg": None,
            "inverted": False,
            "purified_against": ["core, sector, industry, and country factors"],
            "basket": basket,
            "volatility_target": 0.10,
        }

    return {
        **methodology,
        "kind": "unknown",
        "summary": "Construction metadata is unavailable for this factor.",
        "long_leg": None,
        "short_leg": None,
        "inverted": False,
        "purified_against": [],
        "basket": [],
        "volatility_target": None,
    }


def _definition_defaults(
    name: str,
    position: int,
    model_version: str = PROXY_MODEL_VERSION,
) -> dict:
    if model_version != PROXY_MODEL_VERSION:
        raise ValueError(f"Unsupported factor model version: {model_version}")
    if name in PROXY_CORE_FACTORS:
        family = (
            "market" if name == "Market" else ("macro" if position < 6 else "style")
        )
        level = "base"
    elif name in SECTOR_FACTORS:
        family, level = "sector", "sector"
    elif name in INDUSTRY_FACTORS:
        family, level = "industry", "industry"
    elif name in COUNTRY_FACTORS:
        family, level = "country", "country"
    else:
        family, level = "thematic", "thematic"
    proxy_approximation = name in PROXY_FUNDAMENTALS or name in THEMATIC_FACTORS
    provenance_badge = "public_approximation" if proxy_approximation else "exact_etf"
    construction = factor_construction_details(name)
    description = construction["summary"]
    return {
        "external_name": EXTERNAL_NAMES.get(name, name),
        "family": family,
        "level": level,
        "sort_order": position,
        "provenance_badge": provenance_badge,
        "coverage_status": "available",
        "description": description,
        "configuration": {
            "return_source": "local_adjusted_prices",
            "fixed_membership": name in THEMATIC_FACTORS,
            "construction": construction,
        },
    }


@transaction.atomic
def sync_factor_catalogs(
    model_version: str = PROXY_MODEL_VERSION,
) -> dict[str, FactorModelCatalog]:
    if model_version != PROXY_MODEL_VERSION:
        raise ValueError(f"Unsupported factor model version: {model_version}")
    definitions = {}
    for position, name in enumerate(PROXY_AVAILABLE_FACTORS):
        definitions[name], _ = FactorDefinition.objects.update_or_create(
            name=name,
            model_version=model_version,
            defaults=_definition_defaults(name, position, model_version),
        )

    catalogs = {}
    for slug, spec in CATALOG_SPECS.items():
        catalog, _ = FactorModelCatalog.objects.update_or_create(
            slug=slug,
            model_version=model_version,
            defaults={
                "name": spec.name,
                "description": spec.missing,
                "completeness": "complete" if spec.complete else "partial",
                "expected_factor_count": spec.expected_count,
                "available_factor_count": len(spec.factors),
                "coverage": {
                    "available": len(spec.factors),
                    "expected": spec.expected_count,
                    "missing_definition_count": spec.expected_count - len(spec.factors),
                    "disclosure": spec.missing,
                },
            },
        )
        catalog.memberships.all().delete()
        FactorModelCatalogMembership.objects.bulk_create(
            [
                FactorModelCatalogMembership(
                    catalog=catalog, factor=definitions[name], position=position
                )
                for position, name in enumerate(spec.factors)
            ]
        )
        catalogs[slug] = catalog
    return catalogs
