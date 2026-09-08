from datetime import date, timedelta

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from desk.services import persist_snapshot
from desk.workflows import build_monthly_exposures, run_proxy_factor_build
from factors.models import FactorBuild, StockModelFit
from market_data.models import PriceBar, PriceSnapshot
from market_data.providers import YahooWikipediaProvider


class Command(BaseCommand):
    help = (
        "Populate an empty database with Yahoo prices, V2 factor returns, and "
        "monthly stock decompositions."
    )

    def add_arguments(self, parser):
        parser.add_argument("--start", default="2010-01-01")
        parser.add_argument(
            "--end",
            default=(date.today() + timedelta(days=1)).isoformat(),
            help="Exclusive Yahoo end date (default: tomorrow).",
        )
        parser.add_argument("--months", type=int, default=24)
        parser.add_argument("--trailing-days", type=int, default=756)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--skip-yahoo-tls-verification", action="store_true")
        parser.add_argument("--skip-rates", action="store_true")

    def handle(self, *args, **options):
        has_research = (
            PriceBar.objects.exists()
            or FactorBuild.objects.exists()
            or StockModelFit.objects.exists()
        )
        if has_research:
            raise CommandError(
                "Research tables are not empty. This command is for a clean database. "
                "Delete data/db.sqlite3 and rerun setup.ps1, or use the normal "
                "incremental workflows."
            )
        if options["months"] < 1:
            raise CommandError("--months must be at least 1.")
        if options["trailing_days"] < 126:
            raise CommandError("--trailing-days must be at least 126.")

        verify_ssl = not options["skip_yahoo_tls_verification"]
        if not verify_ssl:
            self.stdout.write(
                self.style.WARNING(
                    "Yahoo TLS verification is disabled. Use this only on a trusted network."
                )
            )

        self.stdout.write(
            f"[1/4] Downloading prices from {options['start']} through "
            f"{options['end']} (exclusive end)..."
        )
        try:
            dataset = YahooWikipediaProvider(verify_ssl=verify_ssl).download(
                options["start"], options["end"], limit=None
            )
        except Exception as exc:
            hint = ""
            if verify_ssl and "CERTIFICATE_VERIFY_FAILED" in str(exc).upper():
                hint = (
                    " Retry with --skip-yahoo-tls-verification only on a trusted network."
                )
            raise CommandError(f"Price download failed: {exc}.{hint}") from exc

        source = (
            "yfinance+wikipedia"
            if verify_ssl
            else "yfinance+wikipedia-unverified-tls"
        )
        persist_snapshot(dataset, source)
        price_snapshot = PriceSnapshot.objects.order_by("-id").first()
        if price_snapshot is None:
            raise CommandError("Price persistence completed without a PriceSnapshot.")
        self.stdout.write(
            self.style.SUCCESS(
                f"Stored {price_snapshot.row_count:,} closes through "
                f"{price_snapshot.as_of}; price snapshot {price_snapshot.pk}."
            )
        )

        def progress(value, stage=""):
            suffix = f" - {stage}" if stage else ""
            self.stdout.write(f"  {int(value):3d}%{suffix}")

        self.stdout.write("[2/4] Building the canonical V2 factor dataset...")
        factor_result = run_proxy_factor_build(
            {"price_snapshot_id": price_snapshot.pk}, progress
        )
        factor_build_id = factor_result["factor_build_id"]
        self.stdout.write(
            self.style.SUCCESS(
                f"Stored {factor_result['factor_count']} factors and "
                f"{factor_result['observations']:,} observations; "
                f"factor build {factor_build_id}."
            )
        )

        self.stdout.write(
            f"[3/4] Decomposing the full stock universe over "
            f"{options['months']} month-ends..."
        )
        exposure_parameters = {
            "factor_build_id": factor_build_id,
            "levels": [
                "base",
                "base_sector",
                "base_sector_industry",
                "all_factors",
            ],
            "trailing_days": options["trailing_days"],
            "minimum_observations": 252,
            "selection_mode": "elastic_net",
            "max_months": options["months"],
            "update_mode": "backfill",
        }
        if options["workers"] is not None:
            exposure_parameters["workers"] = options["workers"]
        exposure_result = build_monthly_exposures(exposure_parameters, progress)
        self.stdout.write(
            self.style.SUCCESS(
                f"Stored {exposure_result['model_fits']:,} model fits for "
                f"{len(exposure_result['tickers'])} tickers across "
                f"{exposure_result['months']} month-ends."
            )
        )

        if options["skip_rates"]:
            self.stdout.write("[4/4] Skipped risk-free rates.")
        else:
            self.stdout.write("[4/4] Downloading point-in-time risk-free rates...")
            call_command("fetch_risk_free_rates", start_year=2015)

        self.stdout.write(
            self.style.SUCCESS(
                "Research replication complete. Run .\\start_quantlab.ps1 and open "
                "http://127.0.0.1:8086."
            )
        )
