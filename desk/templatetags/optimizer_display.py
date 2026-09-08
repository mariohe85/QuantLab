import math

from django import template

register = template.Library()

@register.filter
def sharpe_ratio(scenario):
    """Ex-ante Sharpe: expected excess return per unit of expected volatility.

    The excess is measured against the cash rate the run actually solved with,
    which the solver stores in diagnostics, so the ratio matches the objective
    rather than an assumed zero rate.
    """
    diagnostics = getattr(scenario, "diagnostics", None) or {}
    try:
        volatility = float(scenario.expected_volatility)
    except (AttributeError, TypeError, ValueError):
        return "—"
    if volatility <= 0:
        return "—"
    try:
        excess = float(diagnostics["excess_return"])
    except (KeyError, TypeError, ValueError):
        try:
            expected = float(scenario.expected_return)
            excess = expected - float(diagnostics.get("risk_free_rate") or 0)
        except (AttributeError, TypeError, ValueError):
            return "—"
    return f"{excess / volatility:.2f}"


@register.filter
def percentage(value, decimals=2):
    if value is None or value == "":
        return "—"
    try:
        precision = int(decimals)
        return f"{float(value) * 100:.{precision}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter
def signed_percentage(value, decimals=2):
    if value is None or value == "":
        return "—"
    try:
        numeric = float(value)
        precision = int(decimals)
        return f"{numeric * 100:+.{precision}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter
def signed_bps(value):
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value) * 10000:+.0f} bps"
    except (TypeError, ValueError):
        return "—"


@register.filter
def variance_volatility(value, decimals=2):
    """Show a decimal variance as the annualized volatility it implies.

    Variance in squared percentage points reads as a large, unintuitive number
    (a 19.6% risk is 383 %²), so risk is reported on the volatility scale.
    """
    if value is None or value == "":
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if numeric < 0:
        return "—"
    return f"{math.sqrt(numeric) * 100:.{int(decimals)}f}%"


@register.filter
def percentage_squared(value, decimals=2):
    """Format decimal variance in squared percentage-point units."""
    if value is None or value == "":
        return "—"
    try:
        precision = int(decimals)
        return f"{float(value) * 10000:.{precision}f} %²"
    except (TypeError, ValueError):
        return "—"
