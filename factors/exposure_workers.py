"""Stock exposure fits executed inside pool workers.

Kept free of Django model imports so ``spawn``-based process workers can import
it without initialising the app registry.
"""

from __future__ import annotations

from factors.engine import estimate_exposure


def run_exposure_estimate(payload: dict):
    """Fit one ticker/model-level pair, returning ``None`` for unusable samples."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    def _fit():
        try:
            estimate = estimate_exposure(
                payload["stock"],
                payload["design"],
                min_obs=payload["minimum"],
                selection_mode=payload["selection_mode"],
            )
        except ValueError:
            return None
        return {
            "ticker": payload["ticker"],
            "level": payload["level"],
            "as_of": payload["as_of"],
            "estimate": estimate,
            "coverage": payload["coverage"],
            "n_columns": payload["n_columns"],
        }

    # Each worker owns one core, so nested BLAS threads would only oversubscribe.
    if threadpool_limits is None:
        return _fit()
    with threadpool_limits(limits=1):
        return _fit()
