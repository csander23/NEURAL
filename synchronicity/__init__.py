"""NEURAL.synchronicity - FluoroSNAPP per-recording sync metrics.

NeuroCa was dropped 2026-06-07 per user request. Only FluoroSNAPP remains.
"""
from ._burst_sync_metrics import (
    sync_metrics_per_recording,
    SYNC_METRIC_NAMES,
    COINCIDENCE_WINDOW_SEC,
)

__all__ = [
    "sync_metrics_per_recording",
    "SYNC_METRIC_NAMES",
    "COINCIDENCE_WINDOW_SEC",
]
