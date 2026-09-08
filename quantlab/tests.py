from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


class ProjectGuardTests(SimpleTestCase):
    def test_default_runserver_port_is_local_8086(self):
        manage = (settings.BASE_DIR / "manage.py").read_text(encoding="utf-8")
        self.assertIn('sys.argv.append("127.0.0.1:8086")', manage)

    def test_python_sources_have_no_internal_system_references(self):
        forbidden = ["Mawer", "BoringFactory", "DynamicSense", "pyodbc", "psycopg", "sqlalchemy"]
        source_roots = [
            "quantlab", "market_data", "factors", "selection",
            "optimization", "backtests", "jobs", "desk",
        ]
        violations = []
        for root in source_roots:
            for path in (settings.BASE_DIR / root).rglob("*.py"):
                if path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                for term in forbidden:
                    if term.lower() in text.lower():
                        violations.append(f"{path.relative_to(settings.BASE_DIR)}:{term}")
        self.assertEqual(violations, [])

    def test_database_is_under_project_data_directory(self):
        source = (settings.BASE_DIR / "quantlab" / "settings.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"NAME": BASE_DIR / "data" / "db.sqlite3"', source)

    def test_removed_market_data_file_infrastructure_does_not_return(self):
        forbidden = [
            "par" + "quet",
            "py" + "arrow",
            "save_" + "snapshot",
            "load_" + "snapshot",
            "ohl" + "cv",
            "vol" + "ume",
            "DATA_" + "CACHE_DIR",
        ]
        source_roots = [
            "quantlab",
            "market_data",
            "factors",
            "selection",
            "optimization",
            "backtests",
            "jobs",
            "desk",
        ]
        violations = []
        for root in source_roots:
            for path in (settings.BASE_DIR / root).rglob("*.py"):
                relative = path.relative_to(settings.BASE_DIR)
                if (
                    path.resolve() == Path(__file__).resolve()
                    or "migrations" in relative.parts
                    or path.name.startswith("tests")
                ):
                    continue
                text = path.read_text(encoding="utf-8").lower()
                for term in forbidden:
                    if term.lower() in text:
                        violations.append(f"{relative}:{term}")
        self.assertEqual(violations, [])
