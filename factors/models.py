import re
from datetime import date

from django.db import models

from market_data.models import PriceSnapshot, Security

from .proxyfactorlib import PROXY_MODEL_VERSION


class FactorDefinition(models.Model):
    PROVENANCE = [
        ("exact_etf", "Exact ETF construction"),
        ("public_approximation", "Public approximation"),
        ("unsupported", "Unsupported"),
    ]
    COVERAGE = [
        ("available", "Available"),
        ("partial", "Partial"),
        ("unsupported", "Unsupported"),
    ]
    name = models.CharField(max_length=80)
    external_name = models.CharField(max_length=120, blank=True)
    model_version = models.CharField(max_length=40, default=PROXY_MODEL_VERSION)
    family = models.CharField(max_length=32)
    level = models.CharField(max_length=24, default="base")
    sort_order = models.PositiveSmallIntegerField(default=0)
    provenance_badge = models.CharField(max_length=32, choices=PROVENANCE)
    description = models.TextField(blank=True)
    configuration = models.JSONField(default=dict)
    coverage_status = models.CharField(
        max_length=16, choices=COVERAGE, default="available"
    )
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    @property
    def field_key(self) -> str:
        """Form-field suffix that survives a model-version switch.

        Primary keys are per model version, so a form rendered for one version
        cannot be submitted against another. Names are unique within a version.
        """
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["name", "model_version"], name="uq_factor_definition_version"
            )
        ]


class FactorModelCatalog(models.Model):
    COMPLETENESS = [
        ("complete", "Complete"),
        ("partial", "Partial"),
        ("unavailable", "Unavailable"),
    ]
    slug = models.CharField(max_length=32)
    name = models.CharField(max_length=80)
    model_version = models.CharField(max_length=40, default=PROXY_MODEL_VERSION)
    description = models.TextField(blank=True)
    completeness = models.CharField(
        max_length=16, choices=COMPLETENESS, default="partial"
    )
    expected_factor_count = models.PositiveSmallIntegerField(null=True, blank=True)
    available_factor_count = models.PositiveSmallIntegerField(default=0)
    coverage = models.JSONField(default=dict)
    factors = models.ManyToManyField(
        FactorDefinition,
        through="FactorModelCatalogMembership",
        related_name="model_catalogs",
    )

    class Meta:
        ordering = ["id"]
        constraints = [
            models.UniqueConstraint(
                fields=["slug", "model_version"], name="uq_factor_model_catalog_version"
            )
        ]

    def __str__(self):
        return self.name


class FactorModelCatalogMembership(models.Model):
    catalog = models.ForeignKey(
        FactorModelCatalog, on_delete=models.CASCADE, related_name="memberships"
    )
    factor = models.ForeignKey(
        FactorDefinition, on_delete=models.CASCADE, related_name="catalog_memberships"
    )
    position = models.PositiveSmallIntegerField()

    class Meta:
        ordering = ["position"]
        constraints = [
            models.UniqueConstraint(
                fields=["catalog", "factor"], name="uq_factor_model_catalog_member"
            ),
            models.UniqueConstraint(
                fields=["catalog", "position"], name="uq_factor_model_catalog_position"
            ),
        ]

    def __str__(self):
        return f"{self.catalog_id}:{self.position}"


class FactorBuild(models.Model):
    MODES = [("replication", "V2 proxy")]
    UPDATE_STATES = [
        ("idle", "Idle"),
        ("updating", "Updating"),
        ("failed", "Failed"),
    ]
    definition_version = models.CharField(max_length=40, default=PROXY_MODEL_VERSION)
    model_fingerprint = models.CharField(max_length=64, blank=True, db_index=True)
    is_canonical = models.BooleanField(default=True, db_index=True)
    superseded_by = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="superseded_builds",
    )
    price_snapshot = models.ForeignKey(
        PriceSnapshot,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="factor_builds",
    )
    mode = models.CharField(max_length=20, choices=MODES, default="replication")
    status = models.CharField(max_length=24, default="pending")
    as_of = models.DateField()
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    configuration = models.JSONField(default=dict)
    diagnostics = models.JSONField(default=dict)
    provenance = models.JSONField(default=dict)
    coverage = models.JSONField(default=dict)
    artifact_path = models.CharField(max_length=500, blank=True)
    checksum = models.CharField(max_length=64, blank=True)
    input_checksum = models.CharField(max_length=64, blank=True)
    update_state = models.CharField(
        max_length=16, choices=UPDATE_STATES, default="idle"
    )
    last_factor_update_at = models.DateTimeField(null=True, blank=True)

    @property
    def display_label(self):
        stamp = (
            date.fromisoformat(self.as_of)
            if isinstance(self.as_of, str)
            else self.as_of
        )
        release = (
            f"{self.definition_version} [{self.model_fingerprint[:8]}]"
            if self.model_fingerprint
            else self.definition_version
        )
        return f"{release} · {stamp:%b %d, %Y}"

    def __str__(self):
        return self.display_label

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["model_fingerprint"],
                condition=models.Q(is_canonical=True) & ~models.Q(model_fingerprint=""),
                name="uq_canonical_factor_fingerprint",
            )
        ]


class FactorObservation(models.Model):
    build = models.ForeignKey(
        FactorBuild, on_delete=models.CASCADE, related_name="observations"
    )
    factor = models.ForeignKey(
        FactorDefinition, on_delete=models.PROTECT, related_name="observations"
    )
    date = models.DateField()
    raw_return = models.FloatField(null=True)
    pure_return = models.FloatField(null=True)
    scaled_return = models.FloatField(null=True)
    cumulative_index = models.FloatField(null=True)
    coverage = models.FloatField(default=1.0)
    quality_flags = models.JSONField(default=list)
    provenance = models.JSONField(default=dict)
    horizons = models.JSONField(default=dict)
    zscores = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["build", "factor", "date"], name="uq_factor_observation"
            )
        ]
        indexes = [models.Index(fields=["factor", "date"])]

    def __str__(self):
        return f"{self.factor_id} {self.date}"


class FactorEquation(models.Model):
    build = models.ForeignKey(
        FactorBuild, on_delete=models.CASCADE, related_name="equations"
    )
    factor = models.ForeignKey(
        FactorDefinition, on_delete=models.PROTECT, related_name="equations"
    )
    date = models.DateField()
    equation = models.JSONField(default=dict)
    basket_weights = models.JSONField(default=dict)
    regression_betas = models.JSONField(default=dict)
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["build", "factor", "date"], name="uq_factor_equation"
            )
        ]


class StockModelFit(models.Model):
    build = models.ForeignKey(
        FactorBuild, on_delete=models.CASCADE, related_name="stock_fits"
    )
    security = models.ForeignKey(
        Security, on_delete=models.CASCADE, related_name="model_fits"
    )
    as_of = models.DateField()
    period = models.DateField()
    is_provisional = models.BooleanField(default=False)
    source_price_checksum = models.CharField(max_length=64, blank=True)
    source_factor_checksum = models.CharField(max_length=64, blank=True)
    model_level = models.CharField(max_length=24)
    model_catalog = models.ForeignKey(
        FactorModelCatalog,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="stock_fits",
    )
    alpha = models.FloatField()
    adjusted_r2 = models.FloatField()
    residual_volatility = models.FloatField()
    active_factor_count = models.PositiveSmallIntegerField()
    design_factor_count = models.PositiveSmallIntegerField(default=0)
    observation_count = models.PositiveIntegerField()
    coverage = models.FloatField()
    inference_method = models.CharField(
        max_length=80, default="ElasticNetCV selection; conditional OLS/HAC"
    )

    def save(self, *args, **kwargs):
        if self.as_of and not self.period:
            stamp = (
                date.fromisoformat(self.as_of)
                if isinstance(self.as_of, str)
                else self.as_of
            )
            self.period = stamp.replace(day=1)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.security_id} {self.as_of} {self.model_level}"

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["build", "security", "period", "model_level"],
                name="uq_stock_model_fit_period",
            )
        ]


class StockExposureSnapshot(models.Model):
    model_fit = models.ForeignKey(
        StockModelFit, on_delete=models.CASCADE, related_name="exposures"
    )
    factor = models.ForeignKey(
        FactorDefinition, on_delete=models.PROTECT, related_name="stock_exposures"
    )
    beta = models.FloatField()
    standard_error = models.FloatField()
    t_stat = models.FloatField()
    p_value = models.FloatField()
    confidence_low = models.FloatField()
    confidence_high = models.FloatField()
    selected = models.BooleanField(default=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["model_fit", "factor"], name="uq_stock_factor_exposure"
            )
        ]

    def __str__(self):
        return f"{self.model_fit_id} {self.factor_id}"


class FactorModelRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=120)
    config_version = models.CharField(max_length=24, blank=True)
    configuration = models.JSONField(default=dict)
    data_snapshot_id = models.PositiveBigIntegerField(null=True, blank=True)
    status = models.CharField(max_length=24, default="pending")
    metrics = models.JSONField(default=dict)
    exposures = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
