from copy import deepcopy
from datetime import date

from django.core.management.base import BaseCommand
from django.db import connection, transaction
from django.db.models import Count, Max, Min

from factors.models import FactorBuild, FactorObservation
from market_data.models import PriceBar, PriceSnapshot
from market_data.store import clear_price_dataset_cache


class Command(BaseCommand):
    help = "Drop daily prices and factor observations before a cutoff date."

    def add_arguments(self, parser):
        parser.add_argument("--before", default="2018-01-01")
        parser.add_argument("--vacuum", action="store_true")

    def handle(self, *args, **options):
        cutoff = date.fromisoformat(options["before"])
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM market_data_pricebar WHERE date < %s", [cutoff]
                )
                deleted_prices = cursor.rowcount
                cursor.execute(
                    "DELETE FROM factors_factorobservation WHERE date < %s",
                    [cutoff],
                )
                deleted_observations = cursor.rowcount
                cursor.execute(
                    "DELETE FROM factors_factorequation WHERE date < %s", [cutoff]
                )
                deleted_equations = cursor.rowcount

        self._rebase_cumulative_indexes()
        self._refresh_factor_coverage(cutoff)
        self._refresh_price_snapshot(cutoff)
        clear_price_dataset_cache()

        self.stdout.write(
            f"Removed {deleted_prices} prices, {deleted_observations} factor "
            f"observations, {deleted_equations} equations before {cutoff}."
        )
        self.stdout.write(
            f"Remaining: {PriceBar.objects.count()} prices, "
            f"{FactorObservation.objects.count()} factor observations."
        )

        if options["vacuum"]:
            self.stdout.write("Vacuuming SQLite to reclaim disk space...")
            connection.close()
            with connection.cursor() as cursor:
                cursor.execute("VACUUM")
            self.stdout.write(self.style.SUCCESS("Vacuum complete."))

    def _rebase_cumulative_indexes(self):
        pairs = FactorObservation.objects.values_list(
            "build_id", "factor_id"
        ).distinct()
        for build_id, factor_id in pairs:
            rows = list(
                FactorObservation.objects.filter(
                    build_id=build_id, factor_id=factor_id
                ).order_by("date")
            )
            if not rows or not rows[0].cumulative_index:
                continue
            base = float(rows[0].cumulative_index)
            for row in rows:
                if row.cumulative_index is None:
                    continue
                row.cumulative_index = float(row.cumulative_index) / base
            FactorObservation.objects.bulk_update(
                rows, ["cumulative_index"], batch_size=2000
            )

    def _refresh_factor_coverage(self, cutoff):
        for build in FactorBuild.objects.all():
            panel_dates = (
                FactorObservation.objects.filter(build=build)
                .values("date")
                .distinct()
                .count()
            )
            coverage = dict(build.coverage) if isinstance(build.coverage, dict) else {}
            stats = (
                FactorObservation.objects.filter(build=build)
                .values("factor_id", "factor__name")
                .annotate(
                    observations=Count("id"),
                    start=Min("date"),
                    end=Max("date"),
                )
            )
            for row in stats:
                ratio = row["observations"] / panel_dates if panel_dates else 0.0
                coverage[row["factor__name"]] = {
                    "observations": row["observations"],
                    "panel_observations": panel_dates,
                    "ratio": ratio,
                    "start": row["start"].isoformat() if row["start"] else None,
                    "end": row["end"].isoformat() if row["end"] else None,
                }
                FactorObservation.objects.filter(
                    build=build, factor_id=row["factor_id"]
                ).update(coverage=ratio)
            diagnostics = (
                deepcopy(build.diagnostics)
                if isinstance(build.diagnostics, dict)
                else {}
            )
            diagnostics["truncated_before"] = cutoff.isoformat()
            build.coverage = coverage
            build.diagnostics = diagnostics
            build.save(update_fields=["coverage", "diagnostics"])

    def _refresh_price_snapshot(self, cutoff):
        remaining = PriceBar.objects.count()
        earliest = (
            PriceBar.objects.order_by("date").values_list("date", flat=True).first()
        )
        for snapshot in PriceSnapshot.objects.all():
            provenance = dict(snapshot.provenance or {})
            provenance["truncated_before"] = cutoff.isoformat()
            provenance["truncated_earliest_remaining"] = (
                earliest.isoformat() if earliest else None
            )
            snapshot.row_count = remaining
            snapshot.provenance = provenance
            snapshot.save(update_fields=["row_count", "provenance"])
