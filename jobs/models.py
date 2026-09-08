from django.db import models
import uuid


class Job(models.Model):
    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("succeeded", "Succeeded"),
        ("failed", "Failed"),
        ("cancel_requested", "Cancellation requested"),
        ("cancelled", "Cancelled"),
    ]
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    kind = models.CharField(max_length=60)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="queued")
    progress = models.PositiveSmallIntegerField(default=0)
    parameters = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    stage = models.CharField(max_length=80, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    available_at = models.DateTimeField(null=True, blank=True)
    worker_id = models.CharField(max_length=120, blank=True)
    attempts = models.PositiveSmallIntegerField(default=0)
    max_attempts = models.PositiveSmallIntegerField(default=3)
    priority = models.SmallIntegerField(default=0)
    cancel_requested_at = models.DateTimeField(null=True, blank=True)
    idempotency_key = models.CharField(
        max_length=64, db_index=True, default=uuid.uuid4, editable=False
    )

    class Meta:
        ordering = ["-priority", "created_at"]
        indexes = [models.Index(fields=["status", "available_at", "priority"])]

    def __str__(self):
        return f"{self.kind} #{self.pk}"
