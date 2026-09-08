from django.contrib import admin

from desk.admin_mixins import QuantLabModelAdmin

from .models import Job


@admin.register(Job)
class JobAdmin(QuantLabModelAdmin):
    list_display = (
        "id",
        "kind",
        "status",
        "progress",
        "stage",
        "created_at",
        "finished_at",
    )
    list_filter = ("status", "kind")
    search_fields = ("kind", "error", "idempotency_key")
    ordering = ("-created_at",)
