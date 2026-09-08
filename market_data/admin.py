from django.contrib import admin

from desk.admin_mixins import HugeTableAdmin, QuantLabModelAdmin

from .models import (
    Artifact,
    DataSnapshot,
    PriceBar,
    PriceSnapshot,
    RiskFreeRate,
    Security,
    Universe,
    UniverseSnapshot,
)


@admin.register(Universe)
class UniverseAdmin(QuantLabModelAdmin):
    list_display = ("name", "methodology", "created_at")
    search_fields = ("name",)


@admin.register(Security)
class SecurityAdmin(QuantLabModelAdmin):
    list_display = (
        "ticker",
        "name",
        "asset_type",
        "sector",
        "industry",
        "exchange",
        "active",
    )
    list_filter = ("asset_type", "sector", "active")
    search_fields = ("ticker", "name")
    ordering = ("ticker",)


@admin.register(PriceBar)
class PriceBarAdmin(HugeTableAdmin):
    list_display = ("date", "security", "close")
    list_select_related = ("security",)
    search_fields = ("security__ticker",)
    autocomplete_fields = ("security",)
    ordering = ("-date", "-id")
    date_hierarchy = None


@admin.register(UniverseSnapshot)
class UniverseSnapshotAdmin(QuantLabModelAdmin):
    list_display = ("universe", "as_of", "source", "point_in_time", "checksum")
    list_select_related = ("universe",)
    filter_horizontal = ("securities",)
    search_fields = ("source", "checksum")


@admin.register(DataSnapshot)
class DataSnapshotAdmin(QuantLabModelAdmin):
    list_display = ("created_at", "source", "as_of", "checksum")
    search_fields = ("source", "checksum")


@admin.register(PriceSnapshot)
class PriceSnapshotAdmin(QuantLabModelAdmin):
    list_display = (
        "as_of",
        "source",
        "row_count",
        "checksum",
        "created_at",
    )
    list_select_related = ("universe_snapshot",)
    search_fields = ("checksum", "source")


@admin.register(RiskFreeRate)
class RiskFreeRateAdmin(QuantLabModelAdmin):
    list_display = ("date", "annualized_rate", "source")
    search_fields = ("source",)
    ordering = ("-date",)


@admin.register(Artifact)
class ArtifactAdmin(QuantLabModelAdmin):
    list_display = ("created_at", "kind", "owner_type", "owner_id", "format")
    list_filter = ("kind", "format")
    search_fields = ("path", "checksum")
