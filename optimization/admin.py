from django.contrib import admin

from desk.admin_mixins import QuantLabModelAdmin

from .models import (
    OptimizationHolding,
    OptimizationRun,
    OptimizationScenario,
    OptimizationStudy,
)


class OptimizationHoldingInline(admin.TabularInline):
    model = OptimizationHolding
    extra = 0
    autocomplete_fields = ("security",)


@admin.register(OptimizationRun)
class OptimizationRunAdmin(QuantLabModelAdmin):
    list_display = ("created_at", "objective", "status")
    list_filter = ("status", "objective")


@admin.register(OptimizationStudy)
class OptimizationStudyAdmin(QuantLabModelAdmin):
    list_display = (
        "name",
        "portfolio",
        "model_level",
        "status",
        "created_at",
    )
    list_filter = ("status",)
    list_select_related = ("portfolio", "factor_build")
    search_fields = ("name", "portfolio__name")
    autocomplete_fields = ("portfolio", "factor_build")


@admin.register(OptimizationScenario)
class OptimizationScenarioAdmin(QuantLabModelAdmin):
    list_display = (
        "name",
        "variant",
        "status",
        "expected_return",
        "expected_volatility",
        "created_at",
    )
    list_filter = ("variant", "status")
    list_select_related = ("study", "portfolio")
    search_fields = ("name",)
    autocomplete_fields = ("study", "portfolio", "screen_run")
    inlines = (OptimizationHoldingInline,)


@admin.register(OptimizationHolding)
class OptimizationHoldingAdmin(QuantLabModelAdmin):
    list_display = (
        "scenario",
        "security",
        "original_weight",
        "equal_weight",
        "optimized_weight",
    )
    list_select_related = ("scenario", "security")
    autocomplete_fields = ("scenario", "security")
    search_fields = ("security__ticker",)
