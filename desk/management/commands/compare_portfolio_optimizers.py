from __future__ import annotations

import json
from contextlib import contextmanager

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from desk.optimizer_comparison import compare_portfolio_optimizers
from factors.models import FactorBuild
from selection.models import Portfolio


@contextmanager
def read_only_database():
    if connection.vendor != "sqlite":
        yield
        return
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA query_only")
        previous = bool(cursor.fetchone()[0])
        cursor.execute("PRAGMA query_only = ON")
    try:
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA query_only = {'ON' if previous else 'OFF'}")


def _percentage(value: float) -> str:
    return f"{value:>8.2%}"


class Command(BaseCommand):
    help = "Compare portfolio optimizers with 1/N without persisting any results."

    def add_arguments(self, parser):
        parser.add_argument(
            "--portfolio",
            default="Strong growth Portfolio",
            help="Portfolio name (default: Strong growth Portfolio).",
        )
        parser.add_argument("--months", type=int, default=24)
        parser.add_argument("--transaction-cost-bps", type=float, default=10)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Print machine-readable JSON instead of a formatted summary.",
        )

    def handle(self, *args, **options):
        if options["months"] <= 0:
            raise CommandError("--months must be positive")
        if options["transaction_cost_bps"] < 0:
            raise CommandError("--transaction-cost-bps cannot be negative")
        try:
            with read_only_database():
                portfolio = Portfolio.objects.get(name=options["portfolio"])
                selection = (portfolio.configuration or {}).get("stock_selection", {})
                build_id = selection.get("factor_build_id")
                if not build_id:
                    raise ValueError("The portfolio does not identify a factor build")
                build = FactorBuild.objects.get(pk=build_id)
                result = compare_portfolio_optimizers(
                    portfolio,
                    build,
                    months=options["months"],
                    transaction_cost_bps=options["transaction_cost_bps"],
                )
        except Portfolio.DoesNotExist as exc:
            raise CommandError(f"Portfolio not found: {options['portfolio']}") from exc
        except FactorBuild.DoesNotExist as exc:
            raise CommandError("The portfolio's factor build was not found") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        if options["json"]:
            self.stdout.write(json.dumps(result, indent=2))
            return

        self.stdout.write(
            f"{result['portfolio']} | {result['start']} to {result['end']} | "
            f"{result['months']} monthly selections"
        )
        self.stdout.write(
            "Method                         Return      Vol   Sharpe   Max DD"
            "  Turnover  Net ret  Net SR   SR vs 1/N  Fallbacks"
        )
        for row in result["rows"]:
            self.stdout.write(
                f"{row['method']:<29}"
                f"{_percentage(row['annual_return'])}"
                f"{_percentage(row['annual_volatility'])}"
                f"{row['sharpe']:>9.2f}"
                f"{_percentage(row['max_drawdown'])}"
                f"{row['turnover']:>10.2f}"
                f"{_percentage(row['net_annual_return'])}"
                f"{row['net_sharpe']:>8.2f}"
                f"{row['sharpe_vs_1_n']:>12.2f}"
                f"{row['fallbacks']:>11}"
            )
        self.stdout.write(
            f"\nNet results assume {result['transaction_cost_bps']:.1f} bps per "
            "100% one-way turnover."
        )
        for warning in result["methodology_warnings"]:
            self.stdout.write(self.style.WARNING(f"Warning: {warning}"))
        for key, warnings in result["warnings"].items():
            if warnings:
                self.stdout.write(
                    self.style.WARNING(
                        f"{key}: {len(warnings)} rebalance warning(s); "
                        f"first: {warnings[0]}"
                    )
                )
