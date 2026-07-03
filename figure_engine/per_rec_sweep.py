"""Per-recording burst + synchronicity sweep generator.

Produces the ``nn_per_rec_thr<t>.csv`` table that the manuscript's Panels H/I
consume (one row per recording, at a fixed peak-detection threshold). Fully
self-contained inside NEURAL: peak detection via NEURAL.peakcaller, burst
metrics via NEURAL.bursting, synchronicity via NEURAL.synchronicity.

This replaces the legacy external ``diag_threshold_sweep.py`` (which lived
outside the repo), so NEURAL can regenerate these metrics on its own whenever
the NN ROIs change.

Usage:
    from NEURAL.figure_engine.per_rec_sweep import generate_nn_per_rec_sweep
    generate_nn_per_rec_sweep(
        inference_dir="Data/NN_Inference/iN_GCaMP/ca_buffer",
        out_csv="Data/NN_Inference/sweep_per_rec/iN_gcamp/nn_per_rec_thr0.0150.csv",
        threshold=0.015, fps=13.4, baseline="convex")
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from NEURAL.peakcaller import (
    find_peaks as _pc_find_peaks,
    calculate_metrics as _pc_calc_metrics,
    convex_envelope_baseline as _convex,
    detrend_ratio as _detrend_ratio,
)
from NEURAL.bursting import burst_metrics_per_recording, BURST_W_SEC
from NEURAL.synchronicity import sync_metrics_per_recording

# Column order matches the historical sweep CSV so downstream readers are happy.
FIELDS = [
    "condition", "recording", "fps", "n_events", "freq", "amp_mean",
    "amp_mean_per_roi", "burst_adj_freq", "burst_freq", "burstiness",
    "mean_burst_size", "frac_in_bursts", "fluorosnapp_sync", "fluorosnapp_fc",
    "side", "threshold",
]


def _als_baseline(y, *, lam=1e7, p=0.99, n_iter=10):
    """ALS baseline (matches Methods_Paper._common.als_baseline) for negative-going
    indicators (iGABASnFr). Positive-going indicators use the convex hull."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y); D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _detect_peaks(raw, threshold, baseline_kind):
    """Return (peaks, D, hb_idx, ha_idx) for one ROI, or None if the trace is
    inactive/invalid. Matches Methods_Paper.paper_engine detection exactly."""
    raw = np.asarray(raw, dtype=np.float64)
    if (not np.all(np.isfinite(raw))) or np.std(raw) < 1e-9 or np.min(raw) <= 0:
        return None
    if baseline_kind == "als":
        base = _als_baseline(raw)
        D = base / (raw + 1e-12)
    else:
        base = _convex(raw)
        D = _detrend_ratio(raw, base)
    peaks, _h, _b, hb_idx, ha_idx = _pc_find_peaks(
        D, required_rise=threshold, required_fall=threshold,
        max_lookback=26, max_lookahead=26, detect_negative=False,
    )
    return np.asarray(peaks, dtype=int), D, hb_idx, ha_idx


def _condition_from_path(rec_dir: Path) -> str:
    s = str(rec_dir).replace("\\", "/")
    if "/HB/" in s or "Without THC" in s or "WithoutTHC" in s:
        return "HB"
    return "LB"


def generate_nn_per_rec_sweep(inference_dir, out_csv, *, threshold, fps,
                              baseline="convex", w_sec=None):
    """Compute one burst/sync row per recording under ``inference_dir`` and write
    ``out_csv``. A recording is any folder containing ``_summary.csv``.

    Returns the number of recordings written.
    """
    inference_dir = Path(inference_dir)
    out_csv = Path(out_csv)
    if w_sec is None:
        w_sec = BURST_W_SEC.get("calcium", 2.5)

    rows = []
    for summ in sorted(inference_dir.rglob("_summary.csv")):
        rec = summ.parent
        roi_dirs = sorted(p for p in rec.iterdir()
                          if p.is_dir() and p.name.startswith("roi_"))
        per_roi_peaks, amps_per_roi, all_amps, T = [], [], [], 0
        for rd in roi_dirs:
            rp = rd / "raw_trace.npy"
            if not rp.exists():
                continue
            det = _detect_peaks(np.load(rp), threshold, baseline)
            if det is None:
                continue
            peaks, D, hb_idx, ha_idx = det
            T = max(T, D.shape[0])
            per_roi_peaks.append(peaks)
            if peaks.size:
                m = _pc_calc_metrics(D, peaks, hb_idx, ha_idx,
                                     sampling_rate=fps, detect_negative=False)
                a = np.asarray(m.get("amplitude", []), dtype=float)
                a = a[np.isfinite(a)]
                if a.size:
                    all_amps.append(a)
                    amps_per_roi.append(float(np.mean(a)))
        if not per_roi_peaks:
            continue
        dur_min = (T / fps / 60.0) if (T > 0 and fps > 0) else 0.0
        n_events = int(sum(p.size for p in per_roi_peaks))
        burst = burst_metrics_per_recording(per_roi_peaks, dur_min, w_sec, fps)
        sync = sync_metrics_per_recording(per_roi_peaks, T, fps, dur_min)
        flat = np.concatenate(all_amps) if all_amps else np.array([])
        rows.append({
            "condition": _condition_from_path(rec),
            "recording": rec.name,
            "fps": fps,
            "n_events": n_events,
            "freq": (n_events / dur_min) if dur_min > 0 else 0.0,
            "amp_mean": float(np.mean(flat)) if flat.size else 0.0,
            "amp_mean_per_roi": float(np.mean(amps_per_roi)) if amps_per_roi else 0.0,
            **burst, **sync,
            "side": "NN",
            "threshold": threshold,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)
