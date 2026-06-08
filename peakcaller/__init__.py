"""NEURAL.peakcaller - PeakCaller-faithful baseline + peak detection.

Verbatim port of the Hussman MATLAB PeakCaller (Artimovich et al., 2017):
convex-envelope (or two-sided EMA / diffusion / ALS-via-baseline) trend,
multiplicative D-scale (D = signal / trend), forward+backward find_peaks
with explicit peak deletion, half-before / half-after midpoint crossings.
"""
from ._peakcaller import (
    one_sided_ema, two_sided_ema, diffusion_smooth,
    convex_envelope_baseline, als_baseline, get_trend, detrend_ratio, smooth_trace,
    find_peaks, calculate_metrics, calculate_synchronicity,
)

__all__ = [
    "one_sided_ema", "two_sided_ema", "diffusion_smooth",
    "convex_envelope_baseline", "als_baseline",
    "get_trend", "detrend_ratio", "smooth_trace",
    "find_peaks", "calculate_metrics", "calculate_synchronicity",
]
