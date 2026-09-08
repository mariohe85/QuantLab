from django.db import models


class Universe(models.Model):
    name = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    methodology = models.CharField(max_length=80, default="current_membership")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Security(models.Model):
    ticker = models.CharField(max_length=24, unique=True)
    name = models.CharField(max_length=200, blank=True)
    asset_type = models.CharField(max_length=32, default="stock")
    sector = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=120, blank=True)
    exchange = models.CharField(max_length=40, blank=True)
    currency = models.CharField(max_length=12, default="USD")
    current_shares = models.FloatField(null=True, blank=True)
    market_cap = models.FloatField(null=True, blank=True)
    metadata_as_of = models.DateField(null=True, blank=True)
    provenance = models.JSONField(default=dict)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.ticker


class PriceBar(models.Model):
    security = models.ForeignKey(
        Security, on_delete=models.CASCADE, related_name="price_bars"
    )
    date = models.DateField()
    close = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["security", "date"], name="uq_price_bar"
            )
        ]
        indexes = [models.Index(fields=["date"])]

    def __str__(self):
        return f"{self.security_id} {self.date}"


class UniverseSnapshot(models.Model):
    universe = models.ForeignKey(Universe, on_delete=models.CASCADE, related_name="snapshots")
    as_of = models.DateField()
    effective_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=120)
    point_in_time = models.BooleanField(default=False)
    survivorship_warning = models.BooleanField(default=True)
    securities = models.ManyToManyField(Security, related_name="universe_snapshots")
    checksum = models.CharField(max_length=64, blank=True)
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["universe", "as_of", "source"], name="uq_universe_snapshot")
        ]

    def __str__(self):
        return f"{self.universe_id} {self.as_of}"


class DataSnapshot(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=80)
    as_of = models.DateField()
    universe = models.JSONField(default=list)
    provenance = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
    checksum = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]


class PriceSnapshot(models.Model):
    universe_snapshot = models.ForeignKey(
        UniverseSnapshot, null=True, blank=True, on_delete=models.SET_NULL, related_name="prices"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    as_of = models.DateField()
    source = models.CharField(max_length=120)
    checksum = models.CharField(max_length=64)
    row_count = models.PositiveIntegerField(default=0)
    missing_flags = models.JSONField(default=dict)
    anomaly_flags = models.JSONField(default=dict)
    provenance = models.JSONField(default=dict)


class RiskFreeRate(models.Model):
    date = models.DateField(unique=True)
    annualized_rate = models.FloatField()
    source = models.CharField(max_length=60, blank=True)

    def __str__(self):
        return str(self.date)


class Artifact(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    kind = models.CharField(max_length=60)
    owner_type = models.CharField(max_length=60)
    owner_id = models.PositiveBigIntegerField()
    path = models.CharField(max_length=500)
    format = models.CharField(max_length=20)
    checksum = models.CharField(max_length=64)
    metadata = models.JSONField(default=dict)
