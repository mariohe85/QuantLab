from django.contrib import admin


class QuantLabModelAdmin(admin.ModelAdmin):
    list_per_page = 50


class HugeTableAdmin(QuantLabModelAdmin):
    """Avoid COUNT(*) / facet scans on million-row research tables."""

    show_full_result_count = False
    show_facets = admin.ShowFacets.NEVER
