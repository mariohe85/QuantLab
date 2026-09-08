from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable
from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from .models import Job

logger = logging.getLogger("quantlab.jobs")


class JobCancelled(Exception):
    pass


def submit_job(
    kind: str,
    parameters: dict,
    work: Callable[[Callable[[int], None]], dict] | None = None,
    *,
    max_attempts: int = 3,
    priority: int = 0,
    idempotency_key: str | None = None,
) -> Job:
    """Enqueue only. The callable argument remains for source compatibility and is never executed."""
    payload = {
        "kind": kind,
        "parameters": parameters,
        "max_attempts": max_attempts,
        "priority": priority,
        "available_at": timezone.now(),
    }
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    return Job.objects.create(**payload)


def request_cancellation(job: Job) -> Job:
    if job.status == "queued":
        job.status = "cancelled"
        job.cancel_requested_at = timezone.now()
        job.finished_at = timezone.now()
        job.stage = "cancelled before start"
        job.save(
            update_fields=["status", "cancel_requested_at", "finished_at", "stage"]
        )
    elif job.status == "running":
        job.status = "cancel_requested"
        job.cancel_requested_at = timezone.now()
        job.save(update_fields=["status", "cancel_requested_at"])
    return job


def recover_stale_jobs(stale_after: timedelta = timedelta(minutes=5)) -> int:
    cutoff = timezone.now() - stale_after
    stale = Job.objects.filter(status="running").filter(
        Q(heartbeat_at__lt=cutoff) | Q(heartbeat_at__isnull=True, started_at__lt=cutoff)
    )
    recovered = 0
    for job in stale:
        if job.attempts < job.max_attempts:
            job.status = "queued"
            job.worker_id = ""
            job.available_at = timezone.now()
            job.error = "Recovered after stale worker heartbeat"
        else:
            job.status = "failed"
            job.finished_at = timezone.now()
            job.error = "Retry limit reached after stale worker heartbeat"
        job.save()
        recovered += 1
    return recovered


def claim_job(worker_id: str | None = None) -> Job | None:
    worker_id = worker_id or socket.gethostname()
    for _ in range(5):
        candidate = (
            Job.objects.filter(status="queued", available_at__lte=timezone.now())
            .order_by("-priority", "created_at")
            .first()
        )
        if candidate is None:
            return None
        now = timezone.now()
        updated = Job.objects.filter(pk=candidate.pk, status="queued").update(
            status="running",
            started_at=now,
            heartbeat_at=now,
            worker_id=worker_id,
            attempts=candidate.attempts + 1,
            progress=max(candidate.progress, 1),
        )
        if updated == 1:
            return Job.objects.get(pk=candidate.pk)
    return None


def _handler(kind: str):
    from desk.services import run_live
    from desk.workflows import (
        bootstrap_normalized,
        build_monthly_exposures,
        monitor_portfolio,
        run_backtest,
        run_canonical_update,
        run_optimization,
        run_proxy_factor_build,
        run_scenario_backtest,
        run_screen,
        sync_model_catalogs,
    )

    handlers = {
        "offline_bootstrap": bootstrap_normalized,
        "live_download": lambda p, progress: run_live(
            p["start"], p["end"], p.get("limit"), progress
        ),
        "proxy_factor_build": run_proxy_factor_build,
        "canonical_update": run_canonical_update,
        "factor_catalog_sync": sync_model_catalogs,
        "monthly_exposures": build_monthly_exposures,
        "screen_run": run_screen,
        "optimization": run_optimization,
        "risk_refresh": monitor_portfolio,
        "backtest": run_backtest,
        "scenario_backtest": run_scenario_backtest,
    }
    if kind not in handlers:
        raise ValueError(f"Unknown durable job kind: {kind}")
    return handlers[kind]


def execute_job(job: Job) -> Job:
    stop_heartbeat = threading.Event()
    interval = float(getattr(settings, "JOB_HEARTBEAT_SECONDS", 10))

    def heartbeat() -> None:
        while not stop_heartbeat.wait(interval):
            try:
                status = (
                    Job.objects.filter(pk=job.pk)
                    .values_list("status", flat=True)
                    .first()
                )
                if status == "cancel_requested":
                    continue
                Job.objects.filter(pk=job.pk, status="running").update(
                    heartbeat_at=timezone.now()
                )
            except Exception:
                logger.warning("Heartbeat update failed for job %s", job.pk)

    heartbeat_thread = threading.Thread(
        target=heartbeat, name=f"quant-heartbeat-{job.pk}", daemon=True
    )
    heartbeat_thread.start()

    def progress(value: int, stage: str = "") -> None:
        current = Job.objects.get(pk=job.pk)
        if current.status == "cancel_requested":
            raise JobCancelled()
        current.progress = max(1, min(99, int(value)))
        current.stage = stage
        current.heartbeat_at = timezone.now()
        current.save(update_fields=["progress", "stage", "heartbeat_at"])

    try:
        parameters = {
            **job.parameters,
            "_job_id": job.pk,
            "_idempotency_key": job.idempotency_key,
        }
        result = _handler(job.kind)(parameters, progress)
        Job.objects.filter(pk=job.pk).update(
            status="succeeded",
            progress=100,
            stage="complete",
            result=result,
            heartbeat_at=timezone.now(),
            finished_at=timezone.now(),
        )
    except JobCancelled:
        Job.objects.filter(pk=job.pk).update(
            status="cancelled", stage="cancelled", finished_at=timezone.now()
        )
    except Exception as exc:
        logger.exception("Durable job %s failed", job.pk)
        current = Job.objects.get(pk=job.pk)
        if current.attempts < current.max_attempts:
            current.status = "queued"
            current.available_at = timezone.now() + timedelta(
                seconds=2**current.attempts
            )
            current.error = str(exc)
            current.worker_id = ""
            current.save()
        else:
            current.status = "failed"
            current.error = str(exc)
            current.finished_at = timezone.now()
            current.save()
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=3)
    return Job.objects.get(pk=job.pk)
