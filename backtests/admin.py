from django.contrib import admin

from desk.admin_mixins import QuantLabModelAdmin

from .models import BacktestArtifact, BacktestRebalance, BacktestRun


class BacktestRebalanceInline(admin.TabularInline):
    model = BacktestRebalance
    extra = 0


@admin.register(BacktestRun)
class BacktestRunAdmin(QuantLabModelAdmin):
    list_display = (
        "name",
        "status",
        "lookback_months",
        "methodology_version",
        "created_at",
    )
    list_filter = ("status",)
    search_fields = ("name",)
    autocomplete_fields = ("portfolio", "screen", "optimization_scenario")
    inlines = (BacktestRebalanceInline,)


@admin.register(BacktestRebalance)
class BacktestRebalanceAdmin(QuantLabModelAdmin):
    list_display = (
        "run",
        "signal_date",
        "execution_date",
        "turnover",
        "costs",
    )
    list_select_related = ("run",)
    autocomplete_fields = ("run",)


@admin.register(BacktestArtifact)
class BacktestArtifactAdmin(QuantLabModelAdmin):
    list_display = ("run", "kind", "format", "created_at")
    list_select_related = ("run",)
    search_fields = ("path", "checksum")
    autocomplete_fields = ("run",)
