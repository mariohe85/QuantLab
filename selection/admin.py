from django.contrib import admin

from desk.admin_mixins import QuantLabModelAdmin

from .models import (
    BreachEvent,
    Portfolio,
    PortfolioHolding,
    PortfolioRiskSnapshot,
    PortfolioSnapshot,
    RiskLimit,
    ScreenDefinition,
    ScreenResult,
    ScreenRun,
    StrategyDefinition,
)


class PortfolioHoldingInline(admin.TabularInline):
    model = PortfolioHolding
    extra = 0
    autocomplete_fields = ("security",)


@admin.register(StrategyDefinition)
class StrategyDefinitionAdmin(QuantLabModelAdmin):
    list_display = ("name", "version", "active", "updated_at")
    search_fields = ("name",)


@admin.register(Portfolio)
class PortfolioAdmin(QuantLabModelAdmin):
    list_display = ("name", "source", "archived", "updated_at")
    list_filter = ("source", "archived")
    search_fields = ("name",)
    autocomplete_fields = ("benchmark",)
    inlines = (PortfolioHoldingInline,)


@admin.register(PortfolioHolding)
class PortfolioHoldingAdmin(QuantLabModelAdmin):
    list_display = ("portfolio", "security", "weight", "shares")
    list_select_related = ("portfolio", "security")
    autocomplete_fields = ("portfolio", "security")
    search_fields = ("portfolio__name", "security__ticker")


@admin.register(PortfolioSnapshot)
class PortfolioSnapshotAdmin(QuantLabModelAdmin):
    list_display = ("portfolio", "as_of", "total_value", "source")
    list_select_related = ("portfolio",)
    autocomplete_fields = ("portfolio",)


@admin.register(PortfolioRiskSnapshot)
class PortfolioRiskSnapshotAdmin(QuantLabModelAdmin):
    list_display = (
        "portfolio",
        "as_of",
        "predicted_volatility",
        "factor_variance",
        "specific_variance",
        "coverage",
    )
    list_select_related = ("portfolio", "factor_build")
    search_fields = ("portfolio__name",)
    autocomplete_fields = ("portfolio", "factor_build")


@admin.register(RiskLimit)
class RiskLimitAdmin(QuantLabModelAdmin):
    list_display = (
        "name",
        "portfolio",
        "metric",
        "lower_bound",
        "upper_bound",
        "active",
    )
    list_select_related = ("portfolio", "factor")
    search_fields = ("name", "portfolio__name")
    autocomplete_fields = ("portfolio", "factor")


@admin.register(BreachEvent)
class BreachEventAdmin(QuantLabModelAdmin):
    list_display = ("limit", "status", "observed_value", "slack", "opened_at")
    list_filter = ("status",)
    list_select_related = ("limit", "risk_snapshot")
    autocomplete_fields = ("limit", "risk_snapshot")


@admin.register(ScreenDefinition)
class ScreenDefinitionAdmin(QuantLabModelAdmin):
    list_display = ("name", "model_level", "top_n", "active", "updated_at")
    search_fields = ("name",)
    autocomplete_fields = ("factor_build",)


@admin.register(ScreenRun)
class ScreenRunAdmin(QuantLabModelAdmin):
    list_display = ("definition", "as_of", "status", "created_at")
    list_select_related = ("definition",)
    search_fields = ("definition__name",)
    autocomplete_fields = ("definition",)


@admin.register(ScreenResult)
class ScreenResultAdmin(QuantLabModelAdmin):
    list_display = ("run", "security", "passed", "rank", "score")
    list_select_related = ("run", "security")
    autocomplete_fields = ("run", "security", "model_fit")
    search_fields = ("security__ticker",)
