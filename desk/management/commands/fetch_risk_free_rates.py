from django.core.management.base import BaseCommand, CommandError

from market_data.models import RiskFreeRate
from market_data.rates import refresh_risk_free_rates


class Command(BaseCommand):
    help = "Download 13-week Treasury bill yields used as the point-in-time cash rate."

    def add_arguments(self, parser):
        parser.add_argument("--start-year", type=int, default=2015)
        parser.add_argument("--end-year", type=int, default=None)

    def handle(self, *args, **options):
        try:
            count = refresh_risk_free_rates(options["start_year"], options["end_year"])
        except Exception as exc:
            raise CommandError(f"Risk-free rate download failed: {exc}") from exc
        latest = RiskFreeRate.objects.order_by("-date").first()
        self.stdout.write(
            self.style.SUCCESS(
                f"Stored {count} rate observations; latest {latest.date} "
                f"at {latest.annualized_rate:.2%}"
            )
        )
