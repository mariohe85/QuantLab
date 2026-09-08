from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from backtests.models import BacktestRun
from factors.models import (
    FactorBuild,
    FactorDefinition,
    FactorEquation,
    FactorModelCatalog,
    FactorModelRun,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)
from factors.proxyfactorlib import PROXY_MODEL_VERSION
from jobs.models import Job
from market_data.models import Artifact
from optimization.models import (
    OptimizationRun,
    OptimizationScenario,
    OptimizationStudy,
)
from selection.models import Portfolio, ScreenDefinition, StrategyDefinition

REMOVED_JOB_KINDS = {
    "backtest",
    "factor_build",
    "factor_reconciliation",
    "factorstoday_build",
    "offline_bootstrap",
    "optimization",
    "risk_refresh",
    "scenario_backtest",
    "screen_run",
}


class Command(BaseCommand):
    help = (
        "Remove non-V2 model data and saved research state while preserving "
        "the canonical V2 factors, stock decompositions, and input snapshots."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Confirm destructive cleanup. Without this flag only counts are shown.",
        )

    def handle(self, *args, **options):
        v2_build_ids = set(
            FactorBuild.objects.filter(
                definition_version=PROXY_MODEL_VERSION
            ).values_list("id", flat=True)
        )
        removed_builds = list(
            FactorBuild.objects.exclude(definition_version=PROXY_MODEL_VERSION).values(
                "id", "definition_version", "artifact_path"
            )
        )
        removed_build_ids = {row["id"] for row in removed_builds}

        v2_backtest_ids = set()
        for run in BacktestRun.objects.all().only("id", "configuration"):
            build_id = (run.configuration or {}).get("factor_build_id")
            if build_id in v2_build_ids:
                v2_backtest_ids.add(run.pk)
        for job in Job.objects.filter(kind__in=["backtest", "scenario_backtest"]):
            build_id = (job.parameters or {}).get("factor_build_id")
            run_id = (job.result or {}).get("backtest_run_id")
            if build_id in v2_build_ids and run_id:
                v2_backtest_ids.add(run_id)
        removed_backtests = BacktestRun.objects.exclude(pk__in=v2_backtest_ids)
        stale_model_jobs = Job.objects.filter(kind="monthly_exposures").exclude(
            parameters__factor_build_id__in=v2_build_ids
        )

        counts = {
            "v2_builds_preserved": len(v2_build_ids),
            "non_v2_builds": len(removed_builds),
            "non_v2_definitions": FactorDefinition.objects.exclude(
                model_version=PROXY_MODEL_VERSION
            ).count(),
            "non_v2_catalogs": FactorModelCatalog.objects.exclude(
                model_version=PROXY_MODEL_VERSION
            ).count(),
            "portfolios": Portfolio.objects.count(),
            "screens": ScreenDefinition.objects.count(),
            "optimization_studies": OptimizationStudy.objects.count(),
            "optimization_scenarios": OptimizationScenario.objects.count(),
            "legacy_optimization_runs": OptimizationRun.objects.count(),
            "backtests_removed": removed_backtests.count(),
            "v2_backtests_preserved": len(v2_backtest_ids),
            "jobs_removed": (
                Job.objects.filter(kind__in=REMOVED_JOB_KINDS).count()
                + stale_model_jobs.count()
            ),
        }
        for name, count in counts.items():
            self.stdout.write(f"{name}: {count}")

        if not options["yes"]:
            self.stdout.write(
                self.style.WARNING(
                    "Dry run only. Pass --yes to apply the V2-only cleanup."
                )
            )
            return
        if not v2_build_ids:
            raise CommandError(
                f"No {PROXY_MODEL_VERSION} build exists; refusing to delete model data."
            )

        with transaction.atomic():
            removed_backtests.delete()
            OptimizationScenario.objects.all().delete()
            OptimizationStudy.objects.all().delete()
            OptimizationRun.objects.all().delete()
            ScreenDefinition.objects.all().delete()
            Portfolio.objects.all().delete()
            StrategyDefinition.objects.all().delete()
            FactorModelRun.objects.all().delete()
            Artifact.objects.filter(
                owner_type="FactorBuild", owner_id__in=removed_build_ids
            ).delete()
            StockExposureSnapshot.objects.filter(
                model_fit__build_id__in=removed_build_ids
            ).delete()
            stale_fits = StockModelFit.objects.filter(build_id__in=removed_build_ids)
            while True:
                fit_ids = list(stale_fits.values_list("id", flat=True)[:500])
                if not fit_ids:
                    break
                StockModelFit.objects.filter(pk__in=fit_ids).delete()
            FactorObservation.objects.filter(build_id__in=removed_build_ids).delete()
            FactorEquation.objects.filter(build_id__in=removed_build_ids).delete()
            FactorBuild.objects.filter(pk__in=removed_build_ids).delete()
            FactorModelCatalog.objects.exclude(
                model_version=PROXY_MODEL_VERSION
            ).delete()
            FactorDefinition.objects.exclude(model_version=PROXY_MODEL_VERSION).delete()
            Job.objects.filter(kind__in=REMOVED_JOB_KINDS).delete()
            Job.objects.filter(kind="canonical_update").exclude(
                parameters__model_version=PROXY_MODEL_VERSION
            ).delete()
            stale_model_jobs.delete()

        self.stdout.write(
            self.style.SUCCESS("V2-only database cleanup complete.")
        )
