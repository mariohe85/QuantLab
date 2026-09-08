from copy import deepcopy

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from backtests.models import BacktestRun
from factors.models import (
    FactorBuild,
    FactorEquation,
    FactorObservation,
    StockExposureSnapshot,
    StockModelFit,
)
from factors.proxyfactorlib import PROXY_MODEL_VERSION
from factors.versioning import build_stock_coverage, canonical_model_fingerprint
from optimization.models import OptimizationStudy
from selection.models import Portfolio, PortfolioRiskSnapshot, ScreenDefinition


def _replace_build_ids(value, source_id, target_id):
    if isinstance(value, dict):
        return {
            key: (
                target_id
                if key in {"build_id", "factor_build_id"} and item == source_id
                else _replace_build_ids(item, source_id, target_id)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_build_ids(item, source_id, target_id) for item in value]
    return value


class Command(BaseCommand):
    help = "Merge duplicate runs of one factor methodology into a canonical dataset."

    def add_arguments(self, parser):
        parser.add_argument("--canonical", type=int, required=True)
        parser.add_argument("--merge", type=int, nargs="+", required=True)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        target = FactorBuild.objects.filter(pk=options["canonical"]).first()
        if not target:
            raise CommandError("Canonical factor dataset does not exist")
        if target.definition_version != PROXY_MODEL_VERSION:
            raise CommandError("Only V2 proxy datasets can be consolidated")
        sources = list(FactorBuild.objects.filter(pk__in=options["merge"]))
        if len(sources) != len(set(options["merge"])):
            raise CommandError("One or more merge datasets do not exist")
        if any(
            source.definition_version != target.definition_version for source in sources
        ):
            raise CommandError(
                "Only datasets with the same model version can be merged"
            )
        if options["dry_run"]:
            self.stdout.write(
                f"Would merge {sum(s.stock_fits.count() for s in sources)} fits "
                f"from {len(sources)} dataset(s) into {target.pk}"
            )
            return

        counters = {"created": 0, "replaced": 0, "observations": 0}
        with transaction.atomic():
            for source in sources:
                self._merge_observations(target, source, counters)
                self._merge_fits(target, source, counters)
                self._repoint_consumers(target, source)
                source.is_canonical = False
                source.superseded_by = target
                source.update_state = "idle"
                source.save(
                    update_fields=[
                        "is_canonical",
                        "superseded_by",
                        "update_state",
                    ]
                )
            stable_configuration = deepcopy(target.configuration)
            fingerprint = canonical_model_fingerprint(
                target.definition_version,
                mode=target.mode,
                configuration=stable_configuration,
            )
            target.configuration = {
                **target.configuration,
                **stable_configuration,
            }
            target.model_fingerprint = fingerprint
            target.is_canonical = True
            target.superseded_by = None
            diagnostics = deepcopy(target.diagnostics)
            diagnostics["consolidation"] = {
                "merged_build_ids": [source.pk for source in sources],
                **counters,
            }
            target.diagnostics = diagnostics
            coverage = dict(target.coverage)
            coverage["stock_fits"] = {
                level: {
                    key: (value.isoformat() if hasattr(value, "isoformat") else value)
                    for key, value in build_stock_coverage(target, level).items()
                }
                for level in (
                    "base",
                    "base_sector",
                    "base_sector_industry",
                    "all_factors",
                )
            }
            target.coverage = coverage
            target.save(
                update_fields=[
                    "model_fingerprint",
                    "is_canonical",
                    "superseded_by",
                    "configuration",
                    "diagnostics",
                    "coverage",
                ]
            )
        self.stdout.write(
            self.style.SUCCESS(
                f"Canonical dataset {target.pk}: {counters['created']} fits added, "
                f"{counters['replaced']} overlaps refreshed, "
                f"{counters['observations']} factor dates added"
            )
        )

    @staticmethod
    def _merge_observations(target, source, counters):
        existing = set(
            FactorObservation.objects.filter(build=target).values_list(
                "factor_id", "date"
            )
        )
        rows = []
        for item in FactorObservation.objects.filter(build=source).iterator():
            if (item.factor_id, item.date) in existing:
                continue
            item.pk = None
            item.build = target
            rows.append(item)
        FactorObservation.objects.bulk_create(rows, batch_size=2000)
        counters["observations"] += len(rows)

        equation_keys = set(
            FactorEquation.objects.filter(build=target).values_list("factor_id", "date")
        )
        equations = []
        for item in FactorEquation.objects.filter(build=source).iterator():
            if (item.factor_id, item.date) in equation_keys:
                continue
            item.pk = None
            item.build = target
            equations.append(item)
        FactorEquation.objects.bulk_create(equations, batch_size=500)

    @staticmethod
    def _merge_fits(target, source, counters):
        fits = StockModelFit.objects.filter(build=source).prefetch_related("exposures")
        for source_fit in fits.iterator(chunk_size=200):
            defaults = {
                "as_of": source_fit.as_of,
                "is_provisional": source_fit.is_provisional,
                "source_price_checksum": source_fit.source_price_checksum,
                "source_factor_checksum": source_fit.source_factor_checksum,
                "model_catalog": source_fit.model_catalog,
                "alpha": source_fit.alpha,
                "adjusted_r2": source_fit.adjusted_r2,
                "residual_volatility": source_fit.residual_volatility,
                "active_factor_count": source_fit.active_factor_count,
                "design_factor_count": source_fit.design_factor_count,
                "observation_count": source_fit.observation_count,
                "coverage": source_fit.coverage,
                "inference_method": source_fit.inference_method,
            }
            fit, created = StockModelFit.objects.update_or_create(
                build=target,
                security=source_fit.security,
                period=source_fit.period,
                model_level=source_fit.model_level,
                defaults=defaults,
            )
            counters["created" if created else "replaced"] += 1
            fit.exposures.all().delete()
            StockExposureSnapshot.objects.bulk_create(
                [
                    StockExposureSnapshot(
                        model_fit=fit,
                        factor_id=exposure.factor_id,
                        beta=exposure.beta,
                        standard_error=exposure.standard_error,
                        t_stat=exposure.t_stat,
                        p_value=exposure.p_value,
                        confidence_low=exposure.confidence_low,
                        confidence_high=exposure.confidence_high,
                        selected=exposure.selected,
                    )
                    for exposure in source_fit.exposures.all()
                ],
                batch_size=500,
            )

    @staticmethod
    def _repoint_consumers(target, source):
        ScreenDefinition.objects.filter(factor_build=source).update(factor_build=target)
        PortfolioRiskSnapshot.objects.filter(factor_build=source).update(
            factor_build=target
        )
        OptimizationStudy.objects.filter(factor_build=source).update(
            factor_build=target
        )
        for model, field in (
            (Portfolio, "configuration"),
            (BacktestRun, "configuration"),
            (OptimizationStudy, "configuration"),
        ):
            for instance in model.objects.all().only("pk", field):
                original = getattr(instance, field)
                revised = _replace_build_ids(original, source.pk, target.pk)
                if revised != original:
                    model.objects.filter(pk=instance.pk).update(**{field: revised})
