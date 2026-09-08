"""Self-contained V2 ETF-proxy and fixed-theme factor model.

The input is a panel of adjusted closes. The output is a 56-factor hierarchy
using Friday compounding, 156-week rolling OLS stripping, and lagged 10%
volatility targeting. No external factor-return series is consumed.

LowVolatility and BetaFactor intentionally use opposite SPLV/SPHB legs and are
therefore highly collinear; stock decomposition relies on ElasticNet selection.
Liquidity is an OEF-versus-IWC tradable proxy rather than stock-level price
impact. Theme membership is fixed and versioned with this model.
"""

from __future__ import annotations

import pandas as pd

from .ft_engine import build_factor, vol_scale

PROXY_MODEL_VERSION = "FactorsToday-V2-Proxy-Thematic"

STRIP_STYLE_SEQUENTIALLY = False

# ---------------------------------------------------------------- specs ----
MACRO_SPECS = {
    "Market": {"kind": "raw", "long": "IVV"},
    "InterestRate": {"kind": "vol", "long": "TLT", "sign": -1},
    "OilPrice": {"kind": "vol", "long": "USO"},
    "GoldPrice": {"kind": "vol", "long": "GLD"},
    "USDollar": {"kind": "vol", "long": "UUP", "short": "BIL"},
    "CreditRisk": {"kind": "vol", "long": "JNK", "short": "IEF"},
}

# order matters when STRIP_STYLE_SEQUENTIALLY=True
STYLE_SPECS = {
    "SmallSize": ("IJR", "IVV"),
    "Momentum": ("MTUM", "IVV"),
    "Value": ("RPV", "RPG"),  # exact
    "LowVolatility": ("SPLV", "SPHB"),
    "Quality": ("SPHQ", "IVV"),  # exact
    "Growth": ("IVW", "IVE"),  # exact
    "BetaFactor": ("SPHB", "SPLV"),
    "DividendYield": ("VYM", "VUG"),  # exact
    "Liquidity": ("OEF", "IWC"),
}

SECTOR_ETFS = {
    "Sector: Technology": "XLK",
    "Sector: Financials": "XLF",
    "Sector: Health Care": "XLV",
    "Sector: Consumer Discretionary": "XLY",
    "Sector: Consumer Staples": "XLP",
    "Sector: Energy": "XLE",
    "Sector: Materials": "XLB",
    "Sector: Industrials": "XLI",
    "Sector: Utilities": "XLU",
    "Sector: Real Estate": "XLRE",
    "Sector: Communication Services": "XLC",
}

INDUSTRY_ETFS = {
    "Industry: Semiconductors": "SOXX",
    "Industry: Software": "IGV",
    "Industry: Biotechnology": "IBB",
    "Industry: Medical Devices": "IHI",
    "Industry: Regional Banks": "KRE",
    "Industry: Aerospace & Defense": "ITA",
    "Industry: Home Construction": "ITB",
    "Industry: Retail": "XRT",
    "Industry: Gold Miners": "GDX",
    "Industry: Clean Energy": "ICLN",
}

COUNTRY_ETFS = {
    "Country: Japan": "EWJ",
    "Country: Germany": "EWG",
    "Country: United Kingdom": "EWU",
    "Country: China": "FXI",
    "Country: Brazil": "EWZ",
    "Country: Canada": "EWC",
    "Country: India": "INDA",
    "Country: South Korea": "EWY",
    "Country: Taiwan": "EWT",
    "Country: Australia": "EWA",
}

# Membership is deliberately fixed and versioned with the model.  Constituents
# favor liquid securities with price history by September 2020 so the themes
# support the requested three-year decomposition history without look-ahead
# membership changes.
THEME_BASKETS = {
    "Theme: AI Capex & Data Centers": (
        "NVDA",
        "AMD",
        "AVGO",
        "ANET",
        "VRT",
        "DELL",
        "SMCI",
        "EQIX",
        "DLR",
        "MU",
        "MRVL",
    ),
    "Theme: Space Economy": (
        "LMT",
        "NOC",
        "RTX",
        "BA",
        "GD",
        "LHX",
        "TDY",
        "HEI",
        "IRDM",
        "VSAT",
    ),
    "Theme: Quantum Computing": (
        "IBM",
        "GOOGL",
        "MSFT",
        "HON",
        "INTC",
        "NVDA",
        "AMAT",
        "MU",
        "ACN",
    ),
    "Theme: Electrification & Power Infrastructure": (
        "ETN",
        "PWR",
        "HUBB",
        "AME",
        "GE",
        "VRT",
        "EME",
        "NVT",
        "PH",
        "ROK",
    ),
    "Theme: Semiconductor Supply Chain": (
        "ASML",
        "AMAT",
        "LRCX",
        "KLAC",
        "TER",
        "MCHP",
        "ON",
        "NXPI",
        "ADI",
        "SWKS",
        "QCOM",
    ),
    "Theme: Cybersecurity": (
        "PANW",
        "FTNT",
        "CHKP",
        "CRWD",
        "ZS",
        "OKTA",
        "AKAM",
        "GEN",
        "TENB",
        "RPD",
    ),
    "Theme: Robotics & Automation": (
        "ROK",
        "AME",
        "TER",
        "ISRG",
        "SYK",
        "DE",
        "CAT",
        "HON",
        "EMR",
        "ZBRA",
    ),
    "Theme: Nuclear & Uranium": (
        "CCJ",
        "BWXT",
        "LEU",
        "UEC",
        "NXE",
        "DNN",
        "UUUU",
        "EXC",
        "DUK",
        "VST",
    ),
    "Theme: Defense & Autonomous Systems": (
        "LMT",
        "NOC",
        "RTX",
        "GD",
        "LHX",
        "AVAV",
        "KTOS",
        "LDOS",
        "HII",
        "TXT",
    ),
    "Theme: Digital Assets Infrastructure": (
        "MSTR",
        "MARA",
        "RIOT",
        "CLSK",
        "HUT",
        "XYZ",
        "PYPL",
        "NVDA",
        "AMD",
        "CME",
    ),
}

CORE_FACTORS = (*MACRO_SPECS, *STYLE_SPECS)
THEMATIC_FACTORS = tuple(THEME_BASKETS)

ALL_TICKERS = sorted(
    {v for s in MACRO_SPECS.values() for v in (s.get("long"), s.get("short")) if v}
    | {t for pair in STYLE_SPECS.values() for t in pair}
    | set(SECTOR_ETFS.values())
    | set(INDUSTRY_ETFS.values())
    | set(COUNTRY_ETFS.values())
    | {ticker for basket in THEME_BASKETS.values() for ticker in basket}
)


# ---------------------------------------------------------------- build ----
def build_core(prices: pd.DataFrame) -> pd.DataFrame:
    """Market + 5 macro + 9 style factors from ETF prices only."""
    r = prices.pct_change(fill_method=None)
    out = {}
    for name, s in MACRO_SPECS.items():
        raw = r[s["long"]] * s.get("sign", 1)
        if s.get("short"):
            raw = raw - r[s["short"]]
        out[name] = raw.dropna() if s["kind"] == "raw" else vol_scale(raw)

    style_so_far: list[str] = []
    for name, (long, hedge) in STYLE_SPECS.items():
        X = r[[hedge]]
        if STRIP_STYLE_SEQUENTIALLY and style_so_far:
            X = pd.concat(
                [X] + [out[f].rename(f) for f in style_so_far], axis=1, sort=True
            )
        out[name] = build_factor(r[long], X)
        style_so_far.append(name)
    return pd.DataFrame(out)


def build_sectors(prices: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    """Sector factors stripped against this model's own core factors."""
    r = prices.pct_change(fill_method=None)
    X = core.dropna()
    return pd.DataFrame(
        {
            name: build_factor(r[etf], X)
            for name, etf in SECTOR_ETFS.items()
            if etf in r.columns
        }
    )


def _build_layer(
    prices: pd.DataFrame,
    stack: pd.DataFrame,
    specifications: dict[str, str],
) -> pd.DataFrame:
    """Build an ETF layer stripped against this model's preceding stack."""
    returns = prices.pct_change(fill_method=None)
    missing = sorted(set(specifications.values()) - set(returns))
    if missing:
        raise ValueError(f"Missing proxy prices: {', '.join(missing)}")
    regressors = stack.dropna()
    return pd.DataFrame(
        {
            name: build_factor(returns[etf], regressors)
            for name, etf in specifications.items()
        }
    )


def build_industries(prices: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    """Industry ETFs stripped against the locally built core and sectors."""
    return _build_layer(prices, stack, INDUSTRY_ETFS)


def build_countries(prices: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    """Country ETFs stripped against the locally built preceding layers."""
    return _build_layer(prices, stack, COUNTRY_ETFS)


def make_thematic_factor(
    prices: pd.DataFrame,
    stack: pd.DataFrame,
    tickers: list[str] | None = None,
    etf: str | None = None,
    name: str = "Thematic",
) -> pd.Series:
    """Build a purified thematic factor from an equal-weight basket of tickers
    (FactorsToday-style custom group) or a single thematic ETF, stripped
    against `stack` (e.g. core + sectors) and vol-scaled to 10%."""
    r = prices.pct_change(fill_method=None)
    if tickers:
        missing = sorted(set(tickers) - set(r))
        if missing:
            raise ValueError(f"Missing thematic prices: {', '.join(missing)}")
        # Requiring every fixed constituent prevents implicit membership drift.
        raw = r[tickers].mean(axis=1, skipna=False)
    elif etf:
        raw = r[etf]
    else:
        raise ValueError("give tickers=[...] or etf='XYZ'")
    return build_factor(raw.dropna(), stack.dropna()).rename(name)


def build_themes(prices: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    """Build all fixed themes against the complete local pre-theme stack."""
    return pd.DataFrame(
        {
            name: make_thematic_factor(prices, stack, tickers=list(tickers), name=name)
            for name, tickers in THEME_BASKETS.items()
        }
    )


def build_model(
    prices: pd.DataFrame,
    with_sectors: bool = True,
    with_industries: bool = True,
    with_countries: bool = True,
    with_themes: bool = True,
) -> pd.DataFrame:
    """Build the self-contained hierarchical proxy factor panel."""
    core = build_core(prices)
    if not with_sectors:
        return core
    sec = build_sectors(prices, core)
    stack = pd.concat([core, sec], axis=1)
    if not with_industries:
        return stack
    industries = build_industries(prices, stack)
    stack = pd.concat([stack, industries], axis=1)
    if not with_countries:
        return stack
    countries = build_countries(prices, stack)
    stack = pd.concat([stack, countries], axis=1)
    if not with_themes:
        return stack
    themes = build_themes(prices, stack)
    return pd.concat([stack, themes], axis=1)
