from django.contrib import admin

from desk.admin_mixins import HugeTableAdmin, QuantLabModelAdmin

from .models import (
    FactorBuild,
    FactorDefinition,
    FactorEquation,
    FactorModelCatalog,
    FactorModelCatalogMembership,
    FactorModelRun,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)


@admin.register(FactorDefinition)
class FactorDefinitionAdmin(QuantLabModelAdmin):
    list_display = (
        "name",
        "family",
        "level",
        "model_version",
        "provenance_badge",
        "coverage_status",
        "active",
    )
    list_filter = ("family", "level", "provenance_badge", "coverage_status", "active")
    search_fields = ("name", "external_name")
    ordering = ("sort_order", "name")


class FactorModelCatalogMembershipInline(admin.TabularInline):
    model = FactorModelCatalogMembership
    extra = 0
    autocomplete_fields = ("factor",)
    ordering = ("position",)


@admin.register(FactorModelCatalog)
class FactorModelCatalogAdmin(QuantLabModelAdmin):
    list_display = (
        "slug",
        "name",
        "model_version",
        "completeness",
        "available_factor_count",
        "expected_factor_count",
    )
    list_filter = ("completeness", "model_version")
    search_fields = ("slug", "name")
    inlines = (FactorModelCatalogMembershipInline,)


@admin.register(FactorModelCatalogMembership)
class FactorModelCatalogMembershipAdmin(QuantLabModelAdmin):
    list_display = ("catalog", "position", "factor")
    list_select_related = ("catalog", "factor")
    autocomplete_fields = ("catalog", "factor")


@admin.register(FactorBuild)
class FactorBuildAdmin(QuantLabModelAdmin):
    list_display = (
        "id",
        "definition_version",
        "as_of",
        "status",
        "is_canonical",
        "update_state",
        "checksum",
    )
    list_filter = ("status", "is_canonical", "update_state", "mode")
    search_fields = ("model_fingerprint", "checksum")
    list_select_related = ("price_snapshot",)


@admin.register(FactorObservation)
class FactorObservationAdmin(HugeTableAdmin):
    list_display = (
        "date",
        "factor",
        "build",
        "scaled_return",
        "pure_return",
        "coverage",
    )
    list_select_related = ("factor", "build")
    search_fields = ("factor__name",)
    autocomplete_fields = ("build", "factor")
    ordering = ("-date", "-id")


@admin.register(FactorEquation)
class FactorEquationAdmin(QuantLabModelAdmin):
    list_display = ("date", "factor", "build")
    list_select_related = ("factor", "build")
    search_fields = ("factor__name",)
    autocomplete_fields = ("build", "factor")


@admin.register(StockModelFit)
class StockModelFitAdmin(HugeTableAdmin):
    list_display = (
        "as_of",
        "security",
        "model_level",
        "adjusted_r2",
        "alpha",
        "active_factor_count",
        "coverage",
        "is_provisional",
    )
    list_select_related = ("security", "build", "model_catalog")
    search_fields = ("security__ticker",)
    autocomplete_fields = ("build", "security", "model_catalog")
    ordering = ("-as_of", "-id")


@admin.register(StockExposureSnapshot)
class StockExposureSnapshotAdmin(HugeTableAdmin):
    list_display = (
        "model_fit",
        "factor",
        "beta",
        "t_stat",
        "p_value",
        "selected",
    )
    list_select_related = ("model_fit__security", "factor")
    search_fields = ("model_fit__security__ticker", "factor__name")
    autocomplete_fields = ("model_fit", "factor")


@admin.register(FactorModelRun)
class FactorModelRunAdmin(QuantLabModelAdmin):
    list_display = ("created_at", "name", "status", "config_version")
    list_filter = ("status",)
    search_fields = ("name",)
