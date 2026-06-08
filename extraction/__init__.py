"""NEURAL.extraction - configurable trace extraction primitives.

Two layers, both opt-in via NEURAL config JSON:

  1. ROI size filter
     extraction.roi_area_filter.min_native_px (null | int) — drop ROIs smaller
     extraction.roi_area_filter.max_native_px (null | int) — drop ROIs larger
     Defaults: null/null (no filter), so NN ROI count is unchanged unless
     explicitly configured.

  2. Pixel weighting variants
     extraction.pixel_weighting: list of {name, kind, ...params}
     Supported kinds:
       - "mean"             — uniform mean over mask (always available)
       - "pixel_top_pct"    — keep only the brightest <percent>% pixels by
                              <rank_by> ("std"|"mean"|"var"), simple mean over those
       - "pixel_std_weighted" — weight each pixel by pixel_std**<exponent>,
                                normalized to sum=1

Each entry produces ONE per-ROI trace + downstream amplitude column named after
the entry's `name`. The basic "mean" variant is always computed; everything
else stacks on top.
"""
from ._extract import (
    apply_roi_area_filter,
    extract_traces_variants,
    PixelWeightingSpec,
    DEFAULT_PIXEL_WEIGHTING,
)

__all__ = [
    "apply_roi_area_filter",
    "extract_traces_variants",
    "PixelWeightingSpec",
    "DEFAULT_PIXEL_WEIGHTING",
]
