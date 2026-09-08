import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "quantlab"


class Command(BaseCommand):
    help = "Create or restore the local Django admin superuser."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset-password",
            action="store_true",
            help="Reset the password even if the user already exists.",
        )

    def handle(self, *args, **options):
        username = os.environ.get("QUANTLAB_ADMIN_USER", DEFAULT_USERNAME)
        password = os.environ.get("QUANTLAB_ADMIN_PASSWORD", DEFAULT_PASSWORD)
        User = get_user_model()
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "is_staff": True,
                "is_superuser": True,
                "is_active": True,
            },
        )
        user.is_staff = True
        user.is_superuser = True
        user.is_active = True
        reset = created or options["reset_password"] or "QUANTLAB_ADMIN_PASSWORD" in os.environ
        if reset:
            user.set_password(password)
        user.save()
        if created:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Created local admin '{username}'. Open /admin/ with that username."
                )
            )
        elif reset:
            self.stdout.write(self.style.SUCCESS(f"Updated local admin '{username}'."))
        else:
            self.stdout.write(f"Local admin '{username}' already exists.")
