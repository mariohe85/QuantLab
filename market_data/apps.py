from django.apps import AppConfig
from django.db.backends.signals import connection_created


def enable_sqlite_wal(sender, connection, **kwargs):
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA journal_mode=WAL")
            # Long factor/exposure builds write thousands of small rows while the
            # web process polls job status; 5s was short enough to abort them.
            cursor.execute("PRAGMA busy_timeout=120000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")


class MarketDataConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "market_data"

    def ready(self):
        connection_created.connect(enable_sqlite_wal, dispatch_uid="quantlab_sqlite_wal")
