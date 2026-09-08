from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Clear all local QuantLab database rows."

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Confirm destructive removal of local research state.",
        )

    def handle(self, *args, **options):
        if not options["yes"]:
            raise CommandError("Pass --yes to confirm the local workspace reset.")

        call_command("flush", interactive=False, verbosity=0)
        call_command("ensure_local_admin", verbosity=0)
        self.stdout.write(
            self.style.SUCCESS(
                "QuantLab reset complete; research rows removed and the local admin login restored."
            )
        )
