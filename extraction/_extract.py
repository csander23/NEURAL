"""Per-ROI trace extraction with configurable pixel weighting + size filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class PixelWeightingSpec:
    """One pixel-weighting recipe.

    name      : column name in the extended _summary.csv
    kind      : one of {"mean", "pixel_top_pct", "pixel_std_weighted"}
    percent   : (pixel_top_pct only) keep top % of pixels by `rank_by`
    rank_by   : (pixel_top_pct only) "std"|"mean"|"var"
    exponent  : (pixel_std_weighted only) raise pixel-std to this power before normalizing
    """
    name: str
    kind: str
    percent: float = 30.0
    rank_by: str = "std"
    exponent: float = 1.0


# Sensible default list — used when config doesn't specify any.
DEFAULT_PIXEL_WEIGHTING = [
    PixelWeightingSpec(name="mean",        kind="mean"),
    PixelWeightingSpec(name="pixel_top30", kind="pixel_top_pct", percent=30.0, rank_by="std"),
    PixelWeightingSpec(name="pixel_std",   kind="pixel_std_weighted", exponent=1.0),
]


def apply_roi_area_filter(masks: np.ndarray,
                          min_native_px: Optional[int] = None,
                          max_native_px: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Filter masks by ROI area in native pixels.

    Returns (filtered_masks, kept_indices). When both bounds are None, returns
    the input unchanged.
    """
    if masks is None or masks.shape[0] == 0:
        return masks, np.arange(masks.shape[0] if masks is not None else 0)
    if min_native_px is None and max_native_px is None:
        return masks, np.arange(masks.shape[0])
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = np.ones(masks.shape[0], dtype=bool)
    if min_native_px is not None:
        keep &= (areas >= min_native_px)
    if max_native_px is not None:
        keep &= (areas <= max_native_px)
    return masks[keep], np.where(keep)[0]


def _compute_pixel_stats(video: np.ndarray, mask: np.ndarray) -> dict:
    """Compute per-pixel summary statistics across time for one ROI.

    Returns dict with: 'mean' (P,), 'std' (P,), 'var' (P,), 'P' (n_pixels).
    Streams in chunks to avoid materializing the full float32 tensor.
    """
    ys, xs = np.where(mask.astype(bool))
    P = ys.size
    if P == 0:
        return dict(mean=np.zeros(0, dtype=np.float64),
                    std=np.zeros(0, dtype=np.float64),
                    var=np.zeros(0, dtype=np.float64), P=0)
    # Stream: accumulate sum and sum-of-squares
    s = np.zeros(P, dtype=np.float64)
    s2 = np.zeros(P, dtype=np.float64)
    T = video.shape[0]
    chunk = 200
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        ptr = video[t0:t1, ys, xs].astype(np.float64)   # (chunk, P)
        s  += ptr.sum(axis=0)
        s2 += (ptr * ptr).sum(axis=0)
    mean = s / T
    var  = s2 / T - mean * mean
    var[var < 0] = 0.0
    std  = np.sqrt(var)
    return dict(mean=mean, std=std, var=var, P=P, ys=ys, xs=xs)


def _weighted_trace(video: np.ndarray, mask: np.ndarray,
                    spec: PixelWeightingSpec,
                    stats: Optional[dict] = None) -> np.ndarray:
    """Extract one trace under one weighting recipe."""
    T = video.shape[0]
    if spec.kind == "mean":
        # Plain mean over mask. Use streaming dot-product.
        m = mask.astype(np.float32).ravel()
        area = max(m.sum(), 1.0)
        out = np.empty(T, dtype=np.float32)
        H, W = mask.shape
        for t in range(T):
            frame = video[t].reshape(-1).astype(np.float32, copy=False)
            out[t] = float(m @ frame) / area
        return out

    if stats is None:
        stats = _compute_pixel_stats(video, mask)
    if stats["P"] == 0:
        return np.zeros(T, dtype=np.float32)

    ys, xs = stats["ys"], stats["xs"]

    if spec.kind == "pixel_top_pct":
        # Rank pixels by chosen statistic, keep top %.
        rank_metric = stats[spec.rank_by]   # "std"|"mean"|"var"
        n_keep = max(1, int(round(stats["P"] * spec.percent / 100.0)))
        order = np.argsort(rank_metric)[::-1]
        keep_idx = order[:n_keep]
        kept_ys = ys[keep_idx]; kept_xs = xs[keep_idx]
        out = np.empty(T, dtype=np.float32)
        for t in range(T):
            vals = video[t, kept_ys, kept_xs].astype(np.float32)
            out[t] = float(vals.mean())
        return out

    if spec.kind == "pixel_std_weighted":
        w = stats["std"] ** float(spec.exponent)
        s = w.sum()
        if s <= 0:
            return np.zeros(T, dtype=np.float32)
        w = w / s
        out = np.empty(T, dtype=np.float32)
        for t in range(T):
            vals = video[t, ys, xs].astype(np.float32)
            out[t] = float((vals * w).sum())
        return out

    raise ValueError(f"unknown pixel weighting kind: {spec.kind!r}")


def extract_traces_variants(video: np.ndarray, masks: np.ndarray,
                             specs: list[PixelWeightingSpec]) -> dict:
    """For each spec, return a (N_rois, T) trace matrix.

    Returns dict {spec.name: traces_(N,T)}.
    Computes pixel stats once per ROI if any non-mean spec requires it.
    """
    if masks.shape[0] == 0:
        return {s.name: np.zeros((0, video.shape[0]), dtype=np.float32) for s in specs}

    needs_stats = any(s.kind != "mean" for s in specs)
    N, T = masks.shape[0], video.shape[0]
    out = {s.name: np.empty((N, T), dtype=np.float32) for s in specs}

    for i in range(N):
        mask = masks[i].astype(bool)
        stats = _compute_pixel_stats(video, mask) if needs_stats else None
        for s in specs:
            out[s.name][i] = _weighted_trace(video, mask, s, stats=stats)
    return out
