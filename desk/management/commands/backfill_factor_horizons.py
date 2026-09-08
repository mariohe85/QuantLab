import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from desk.workflows import _json
from factors.engine import FACTOR_HORIZONS, horizon_returns, rolling_zscores
from factors.models import FactorBuild, FactorObservation


class Command(BaseCommand):
    help = (
        "Recompute stored horizon returns and z-scores from the daily returns already "
        "held by a factor build, without refitting prices."
    )

    def add_arguments(self, parser):
        parser.add_argument("--build", type=int)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        build = (
            FactorBuild.objects.filter(pk=options["build"]).first()
            if options["build"]
            else FactorBuild.objects.filter(is_canonical=True)
            .order_by("-as_of", "-id")
            .first()
        )
        if not build:
            raise CommandError("No factor build to backfill")

        factor_ids = sorted(
            FactorObservation.objects.filter(build=build)
            .values_list("factor_id", flat=True)
            .distinct()
        )
        updated = 0
        for factor_id in factor_ids:
            observations = list(
                FactorObservation.objects.filter(
                    build=build, factor_id=factor_id
                ).order_by("date")
            )
            values = pd.Series(
                [float(item.scaled_return) for item in observations],
                index=[item.date for item in observations],
                dtype="float64",
            )
            returns = {
                horizon: horizon_returns(values, horizon)
                for horizon in FACTOR_HORIZONS
            }
            zscores = {
                horizon: rolling_zscores(values.to_frame("factor"), horizon)["factor"]
                for horizon in FACTOR_HORIZONS
            }
            for item in observations:
                item.horizons = _json(
                    {
                        str(horizon): returns[horizon].get(item.date)
                        for horizon in FACTOR_HORIZONS
                    }
                )
                item.zscores = _json(
                    {
                        str(horizon): zscores[horizon].get(item.date)
                        for horizon in FACTOR_HORIZONS
                    }
                )
            if not options["dry_run"]:
                with transaction.atomic():
                    FactorObservation.objects.bulk_update(
                        observations, ["horizons", "zscores"], batch_size=2000
                    )
            updated += len(observations)
            self.stdout.write(f"factor {factor_id}: {len(observations)} observations")

        verb = "Would refresh" if options["dry_run"] else "Refreshed"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb} {updated} observations across {len(factor_ids)} factors "
                f"in build {build.pk}"
            )
        )
