from django.db import models

from factors.proxyfactorlib import PROXY_MODEL_VERSION
from selection.models import Portfolio, ScreenDefinition


class BacktestRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=120)
    configuration = models.JSONField(default=dict)
    data_snapshot_id = models.PositiveBigIntegerField(null=True, blank=True)
    status = models.CharField(max_length=24, default="pending")
    metrics = models.JSONField(default=dict)
    holdings = models.JSONField(default=list)
    factor_exposures = models.JSONField(default=list)
    warnings = models.JSONField(default=list)
    result_path = models.CharField(max_length=500, blank=True)
    portfolio = models.ForeignKey(
        Portfolio,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="backtests",
    )
    screen = models.ForeignKey(
        ScreenDefinition,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="backtests",
    )
    optimization_scenario = models.ForeignKey(
        "optimization.OptimizationScenario",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="backtests",
    )
    lookback_months = models.PositiveSmallIntegerField(default=24)
    methodology_version = models.CharField(max_length=40, default=PROXY_MODEL_VERSION)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name


class BacktestRebalance(models.Model):
    run = models.ForeignKey(
        BacktestRun, on_delete=models.CASCADE, related_name="rebalances"
    )
    signal_date = models.DateField()
    execution_date = models.DateField()
    holdings = models.JSONField(default=dict)
    equations = models.JSONField(default=dict)
    exposures = models.JSONField(default=dict)
    covariance = models.JSONField(default=dict)
    optimizer = models.JSONField(default=dict)
    risk = models.JSONField(default=dict)
    costs = models.FloatField(default=0)
    turnover = models.FloatField(default=0)
    breaches = models.JSONField(default=list)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "signal_date"], name="uq_backtest_rebalance"
            )
        ]


class BacktestArtifact(models.Model):
    run = models.ForeignKey(
        BacktestRun, on_delete=models.CASCADE, related_name="artifacts"
    )
    kind = models.CharField(max_length=60)
    path = models.CharField(max_length=500)
    format = models.CharField(max_length=20)
    checksum = models.CharField(max_length=64)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
