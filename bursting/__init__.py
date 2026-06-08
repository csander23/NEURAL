"""NEURAL.bursting - chained-IEI burst grouping + 5 per-recording metrics."""
from ._burst_sync_metrics import (
    burst_metrics_per_recording,
    BURST_METRIC_NAMES,
    BURST_W_SEC,
)

__all__ = [
    "burst_metrics_per_recording",
    "BURST_METRIC_NAMES",
    "BURST_W_SEC",
]
