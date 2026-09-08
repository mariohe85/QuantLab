from django.db import migrations


def _rescale(values, divisor, factor):
    return {
        name: value * factor / divisor if isinstance(value, (int, float)) else value
        for name, value in (values or {}).items()
    }


def to_volatility_units(apps, schema_editor):
    snapshot_model = apps.get_model("selection", "PortfolioRiskSnapshot")
    updated = []
    for snapshot in snapshot_model.objects.all():
        volatility = snapshot.predicted_volatility
        if not volatility:
            continue
        snapshot.component_risk = _rescale(snapshot.component_risk, volatility, 1)
        snapshot.marginal_risk = _rescale(snapshot.marginal_risk, volatility, 1)
        updated.append(snapshot)
    snapshot_model.objects.bulk_update(
        updated, ["component_risk", "marginal_risk"], batch_size=500
    )


def to_variance_units(apps, schema_editor):
    snapshot_model = apps.get_model("selection", "PortfolioRiskSnapshot")
    updated = []
    for snapshot in snapshot_model.objects.all():
        volatility = snapshot.predicted_volatility
        if not volatility:
            continue
        snapshot.component_risk = _rescale(snapshot.component_risk, 1, volatility)
        snapshot.marginal_risk = _rescale(snapshot.marginal_risk, 1, volatility)
        updated.append(snapshot)
    snapshot_model.objects.bulk_update(
        updated, ["component_risk", "marginal_risk"], batch_size=500
    )


class Migration(migrations.Migration):
    """Restate stored factor contributions as risk rather than variance.

    ``component_risk`` and ``marginal_risk`` held the Euler variance split,
    which is what the optimizer already divided by volatility before saving the
    identically named holding fields. Dividing by the portfolio volatility makes
    both sides of the app agree and makes the factor contributions sum to the
    portfolio volatility.
    """

    dependencies = [
        ("selection", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(to_volatility_units, to_variance_units),
    ]
