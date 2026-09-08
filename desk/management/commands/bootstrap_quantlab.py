from django.core.management.base import BaseCommand

from desk.workflows import bootstrap_normalized


class Command(BaseCommand):
    help = "Create a deterministic offline dataset and sample research workspace."

    def handle(self, *args, **options):
        result = bootstrap_normalized({}, lambda *_: None)
        self.stdout.write(
            self.style.SUCCESS(
                f"Offline workspace ready: build {result['factor_build_id']}, portfolio {result['portfolio_id']}"
            )
        )
