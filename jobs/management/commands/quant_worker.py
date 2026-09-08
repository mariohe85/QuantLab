from __future__ import annotations

import logging
import socket
import time

from django.core.management.base import BaseCommand
from django.db import close_old_connections

from jobs.runner import claim_job, execute_job, recover_stale_jobs

logger = logging.getLogger("quantlab.worker")


class Command(BaseCommand):
    help = "Process QuantLab's durable SQLite job queue."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--poll-seconds", type=float, default=1.0)
        parser.add_argument("--worker-id", default=socket.gethostname())

    def handle(self, *args, **options):
        while True:
            close_old_connections()
            recovered = recover_stale_jobs()
            if recovered:
                logger.warning("Recovered %s stale jobs", recovered)
            job = claim_job(options["worker_id"])
            if job is not None:
                logger.info("Processing job id=%s kind=%s", job.pk, job.kind)
                execute_job(job)
            elif options["once"]:
                return
            else:
                time.sleep(max(0.1, options["poll_seconds"]))
            if options["once"]:
                return
