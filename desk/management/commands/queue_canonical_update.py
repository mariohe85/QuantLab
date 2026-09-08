from django.core.management.base import BaseCommand, CommandError

from factors.catalog import PROXY_MODEL_VERSION
from jobs.models import Job
from jobs.runner import submit_job
from market_data.models import PriceSnapshot


class Command(BaseCommand):
    help = "Queue the daily canonical factor and stock-decomposition update."

    def add_arguments(self, parser):
        parser.add_argument("--price-snapshot-id", type=int)
        parser.add_argument("--workers", type=int)

    def handle(self, *args, **options):
        if options["price_snapshot_id"]:
            snapshot = PriceSnapshot.objects.filter(
                pk=options["price_snapshot_id"]
            ).first()
        else:
            snapshot = PriceSnapshot.objects.order_by("-as_of", "-created_at").first()
        if not snapshot:
            raise CommandError("No price snapshot is available")
        parameters = {
            "price_snapshot_id": snapshot.pk,
            "model_version": PROXY_MODEL_VERSION,
            "levels": [
                "base",
                "base_sector",
                "base_sector_industry",
                "all_factors",
            ],
        }
        if options["workers"]:
            parameters["workers"] = options["workers"]
        existing = (
            Job.objects.filter(
                kind="canonical_update",
                status__in=["queued", "running"],
                parameters__price_snapshot_id=snapshot.pk,
            )
            .order_by("-id")
            .first()
        )
        if existing:
            self.stdout.write(
                f"Canonical update job {existing.pk} is already "
                f"{existing.status} for snapshot {snapshot.pk}"
            )
            return
        job = submit_job("canonical_update", parameters)
        self.stdout.write(
            self.style.SUCCESS(
                f"Queued canonical update job {job.pk} for snapshot "
                f"{snapshot.pk} ({snapshot.as_of})"
            )
        )
