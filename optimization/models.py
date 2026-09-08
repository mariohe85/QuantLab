from django.db import models

from factors.models import FactorBuild
from market_data.models import Security
from selection.models import Portfolio, ScreenRun


class OptimizationRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    strategy_id = models.PositiveBigIntegerField(null=True, blank=True)
    objective = models.CharField(max_length=40)
    configuration = models.JSONField(default=dict)
    holdings = models.JSONField(default=dict)
    factor_exposures = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    diagnostics = models.JSONField(default=dict)
    status = models.CharField(max_length=24, default="pending")


class OptimizationStudy(models.Model):
    portfolio = models.ForeignKey(
        Portfolio, on_delete=models.CASCADE, related_name="optimization_studies"
    )
    factor_build = models.ForeignKey(
        FactorBuild, on_delete=models.PROTECT, related_name="optimization_studies"
    )
    name = models.CharField(max_length=120)
    model_level = models.CharField(max_length=24, default="base_sector")
    status = models.CharField(max_length=24, default="pending")
    configuration = models.JSONField(default=dict)
    diagnostics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class OptimizationScenario(models.Model):
    study = models.ForeignKey(
        OptimizationStudy,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="variants",
    )
    portfolio = models.ForeignKey(
        Portfolio, null=True, blank=True, on_delete=models.CASCADE, related_name="optimization_scenarios"
    )
    screen_run = models.ForeignKey(
        ScreenRun, null=True, blank=True, on_delete=models.CASCADE, related_name="optimization_scenarios"
    )
    name = models.CharField(max_length=120)
    variant = models.CharField(max_length=40, default="standard")
    probability = models.FloatField(null=True, blank=True)
    objective = models.CharField(max_length=40, default="min_variance")
    expected_return_model = models.CharField(max_length=40, blank=True)
    status = models.CharField(max_length=24, default="pending")
    configuration = models.JSONField(default=dict)
    diagnostics = models.JSONField(default=dict)
    comparison = models.JSONField(default=dict)
    expected_return = models.FloatField(null=True, blank=True)
    expected_volatility = models.FloatField(null=True, blank=True)
    factor_variance = models.FloatField(null=True, blank=True)
    specific_variance = models.FloatField(null=True, blank=True)
    turnover = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class OptimizationHolding(models.Model):
    scenario = models.ForeignKey(OptimizationScenario, on_delete=models.CASCADE, related_name="holdings")
    security = models.ForeignKey(Security, on_delete=models.PROTECT)
    original_weight = models.FloatField(default=0)
    equal_weight = models.FloatField(default=0)
    optimized_weight = models.FloatField(null=True, blank=True)
    expected_return = models.FloatField(null=True, blank=True)
    marginal_risk = models.FloatField(null=True, blank=True)
    component_risk = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["scenario", "security"], name="uq_optimization_holding")
        ]
