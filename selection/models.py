from django.db import models

from factors.models import FactorBuild, FactorDefinition, StockModelFit
from market_data.models import Security


class StrategyDefinition(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    name = models.CharField(max_length=120, unique=True)
    configuration = models.JSONField(default=dict)
    version = models.PositiveIntegerField(default=1)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class Portfolio(models.Model):
    SOURCES = [("manual", "Manual"), ("screen", "Screen"), ("optimized", "Optimized")]
    name = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    source = models.CharField(max_length=20, choices=SOURCES, default="manual")
    benchmark = models.ForeignKey(
        Security, null=True, blank=True, on_delete=models.SET_NULL, related_name="benchmark_portfolios"
    )
    archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    configuration = models.JSONField(default=dict)

    def __str__(self):
        return self.name


class PortfolioHolding(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="holdings")
    security = models.ForeignKey(Security, on_delete=models.PROTECT, related_name="portfolio_holdings")
    weight = models.FloatField(null=True, blank=True)
    shares = models.FloatField(null=True, blank=True)
    cost_basis = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["portfolio", "security"], name="uq_portfolio_holding")
        ]


class PortfolioSnapshot(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="snapshots")
    as_of = models.DateField()
    holdings = models.JSONField(default=dict)
    total_value = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=40, default="manual")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["portfolio", "as_of"], name="uq_portfolio_snapshot")
        ]


class PortfolioRiskSnapshot(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="risk_snapshots")
    factor_build = models.ForeignKey(
        FactorBuild, null=True, blank=True, on_delete=models.SET_NULL, related_name="portfolio_risk"
    )
    as_of = models.DateField()
    coverage = models.FloatField()
    exposures = models.JSONField(default=dict)
    active_exposures = models.JSONField(default=dict)
    factor_variance = models.FloatField()
    specific_variance = models.FloatField()
    total_variance = models.FloatField()
    predicted_volatility = models.FloatField()
    realized_volatility = models.FloatField(null=True, blank=True)
    tracking_error = models.FloatField(null=True, blank=True)
    drawdown = models.FloatField(null=True, blank=True)
    concentration = models.FloatField(null=True, blank=True)
    component_risk = models.JSONField(default=dict)
    marginal_risk = models.JSONField(default=dict)
    attribution = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["portfolio", "as_of"], name="uq_portfolio_risk_snapshot")
        ]


class RiskLimit(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="risk_limits")
    name = models.CharField(max_length=120)
    metric = models.CharField(max_length=60)
    lower_bound = models.FloatField(null=True, blank=True)
    upper_bound = models.FloatField(null=True, blank=True)
    factor = models.ForeignKey(FactorDefinition, null=True, blank=True, on_delete=models.CASCADE)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class BreachEvent(models.Model):
    STATUSES = [("open", "Open"), ("acknowledged", "Acknowledged"), ("resolved", "Resolved")]
    limit = models.ForeignKey(RiskLimit, on_delete=models.CASCADE, related_name="breaches")
    risk_snapshot = models.ForeignKey(
        PortfolioRiskSnapshot, on_delete=models.CASCADE, related_name="breaches"
    )
    opened_at = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUSES, default="open")
    observed_value = models.FloatField()
    slack = models.FloatField()
    details = models.JSONField(default=dict)


class ScreenDefinition(models.Model):
    name = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    factor_build = models.ForeignKey(
        FactorBuild, null=True, blank=True, on_delete=models.SET_NULL, related_name="screens"
    )
    as_of = models.DateField(null=True, blank=True)
    model_level = models.CharField(max_length=24, default="base_sector")
    factor_weights = models.JSONField(default=dict)
    directions = models.JSONField(default=dict)
    regime_interactions = models.JSONField(default=dict)
    filters = models.JSONField(default=dict)
    top_n = models.PositiveSmallIntegerField(default=20)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class ScreenRun(models.Model):
    definition = models.ForeignKey(ScreenDefinition, on_delete=models.CASCADE, related_name="runs")
    as_of = models.DateField()
    status = models.CharField(max_length=24, default="pending")
    diagnostics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.definition_id} {self.as_of}"


class ScreenResult(models.Model):
    run = models.ForeignKey(ScreenRun, on_delete=models.CASCADE, related_name="results")
    security = models.ForeignKey(Security, on_delete=models.CASCADE, related_name="screen_results")
    model_fit = models.ForeignKey(StockModelFit, null=True, blank=True, on_delete=models.SET_NULL)
    passed = models.BooleanField()
    rank = models.PositiveIntegerField(null=True, blank=True)
    percentile = models.FloatField(null=True, blank=True)
    score = models.FloatField(null=True, blank=True)
    components = models.JSONField(default=dict)
    rule_results = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "security"], name="uq_screen_result")
        ]
