"""Burst + synchronicity metric calculators.

Per-recording per-(variant, threshold):
  burst_metrics_per_recording(per_roi_peaks, dur_min, w_sec, fps) -> dict (5 metrics)
  sync_metrics_per_recording (per_roi_peaks, T, fps, dur_min)     -> dict (2 metrics)

Burst grouping: chained inter-event-interval (IEI) with window W_sec (2.5 s
across all modalities — captures longer cell-level event clusters, not just
sub-second event packets).

Sync method: FluoroSNAPP (Patel et al. 2015) — pairwise Dice coincidence
with a ±W tolerance window + Pearson on smoothed spike trains (FC).

NeuroCa metrics were removed 2026-06-07 per user request — only FluoroSNAPP
remains. The Dice sync ("synch") is what gets shown in paper figures; FC
is computed but typically not displayed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


# Burst window for chained-IEI grouping (sec). 2.5 s uniformly across modalities.
BURST_W_SEC = {"calcium": 2.5, "iglusnfr": 2.5, "igabasnfr": 2.5}

# FluoroSNAPP coincidence window in SECONDS. The frame width per recording is
# round(COINCIDENCE_WINDOW_SEC * fps). Was previously 5 frames (~0.2 s);
# now uniformly 2.5 s per user request 2026-06-07.
COINCIDENCE_WINDOW_SEC = 2.5


# ── Burst grouping (chained IEI) ─────────────────────────────────────────────
def _group_bursts(peaks_frames: np.ndarray, w_frames: float) -> list[list[int]]:
    if len(peaks_frames) == 0:
        return []
    p = np.sort(np.asarray(peaks_frames, dtype=np.int64))
    groups: list[list[int]] = [[int(p[0])]]
    for x in p[1:]:
        if x - groups[-1][-1] < w_frames:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return groups


def _burst_per_roi(peaks: np.ndarray, dur_min: float, w_frames: float) -> tuple:
    groups = _group_bursts(peaks, w_frames)
    n_events = int(sum(len(g) for g in groups))
    n_groups = int(len(groups))
    n_bursts = int(sum(1 for g in groups if len(g) >= 2))
    n_isolated = n_groups - n_bursts
    if dur_min <= 0 or n_events == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    event_freq      = n_events / dur_min
    burst_adj_freq  = (n_isolated + n_bursts) / dur_min
    burst_freq      = n_bursts / dur_min
    burstiness      = (1.0 - burst_adj_freq / event_freq) if event_freq > 0 else 0.0
    burst_sizes     = [len(g) for g in groups if len(g) >= 2]
    mean_burst_size = float(np.mean(burst_sizes)) if burst_sizes else 0.0
    frac_in_bursts  = (float(sum(burst_sizes) / n_events)
                       if n_events > 0 and burst_sizes else 0.0)
    return event_freq, burst_adj_freq, burst_freq, burstiness, mean_burst_size, frac_in_bursts


def burst_metrics_per_recording(per_roi_peaks: list[np.ndarray],
                                  dur_min: float, w_sec: float, fps: float) -> dict:
    """Aggregate 5 burst metrics over active ROIs. Rates via shifted-geom-mean,
    ratios via arithmetic mean."""
    w_frames = w_sec * fps
    active_rows = []
    for p in per_roi_peaks:
        if p.size == 0: continue
        active_rows.append(_burst_per_roi(p, dur_min, w_frames))
    if not active_rows:
        return dict(burst_adj_freq=0.0, burst_freq=0.0, burstiness=0.0,
                    mean_burst_size=0.0, frac_in_bursts=0.0)
    arr = np.asarray(active_rows, dtype=float)  # (n_active, 6)
    def _geom(rates):
        rates = rates[np.isfinite(rates) & (rates >= 0)]
        if not rates.size: return 0.0
        return float(np.exp(np.mean(np.log(rates + 1.0))) - 1.0)
    return dict(
        burst_adj_freq   = _geom(arr[:, 1]),
        burst_freq       = _geom(arr[:, 2]),
        burstiness       = float(np.nanmean(arr[:, 3])),
        mean_burst_size  = float(np.nanmean(arr[:, 4])),
        frac_in_bursts   = float(np.nanmean(arr[:, 5])),
    )


# ── Sync helpers ─────────────────────────────────────────────────────────────
def _binary_spike_train(T: int, peaks: np.ndarray) -> np.ndarray:
    s = np.zeros(T, dtype=np.uint8)
    if peaks.size:
        valid = peaks[(peaks >= 0) & (peaks < T)]
        s[valid] = 1
    return s


def _smear(s: np.ndarray, window: int) -> np.ndarray:
    if not s.any():
        return s.astype(bool)
    idx = np.where(s)[0]
    out = np.zeros(s.size, dtype=bool)
    for i in idx:
        lo = max(0, i - window); hi = min(s.size, i + window + 1)
        out[lo:hi] = True
    return out


def sync_metrics_per_recording(per_roi_peaks: list[np.ndarray], T: int,
                                 fps: float, dur_min: float) -> dict:
    """Return 2 sync metrics:
       fluorosnapp_sync  - pairwise Dice coincidence with ±W tolerance
       fluorosnapp_fc    - Pearson r on smoothed spike trains

    W (coincidence half-width) = round(COINCIDENCE_WINDOW_SEC * fps).
    """
    nan_d = dict(fluorosnapp_sync=np.nan, fluorosnapp_fc=np.nan)
    n_rois = len(per_roi_peaks)
    if n_rois < 2 or T <= 0:
        return nan_d

    window = max(1, int(round(COINCIDENCE_WINDOW_SEC * fps)))
    spikes = [_binary_spike_train(T, p) for p in per_roi_peaks]

    # ── FluoroSNAPP Dice coincidence with ±window tolerance ──────────────────
    smeared_fs = [_smear(s, window) for s in spikes]
    spike_sums = np.array([int(s.sum()) for s in spikes], dtype=float)
    sync_vals = []
    for i in range(n_rois):
        si = spikes[i].astype(bool)
        for j in range(i + 1, n_rois):
            sj = spikes[j].astype(bool)
            coinc_ij = int(np.logical_and(si, smeared_fs[j]).sum())
            coinc_ji = int(np.logical_and(sj, smeared_fs[i]).sum())
            denom = spike_sums[i] + spike_sums[j]
            if denom > 0:
                sync_vals.append((coinc_ij + coinc_ji) / denom)
    fluorosnapp_sync = float(np.mean(sync_vals)) if sync_vals else np.nan

    # FC: Pearson on smoothed spike trains (window width = 2*window+1)
    smooth_w = window * 2 + 1
    kernel = np.ones(smooth_w, dtype=np.float32) / smooth_w
    rate_mat = np.vstack([np.convolve(s.astype(np.float32), kernel, mode="same")
                            for s in spikes])
    sd = rate_mat.std(axis=1)
    active = sd > 1e-12
    if active.sum() >= 2:
        X = rate_mat[active] - rate_mat[active].mean(axis=1, keepdims=True)
        sd_x = X.std(axis=1, keepdims=True); sd_x[sd_x < 1e-12] = 1.0
        X = X / sd_x
        R = (X @ X.T) / X.shape[1]
        iu = np.triu_indices(R.shape[0], k=1)
        fluorosnapp_fc = float(np.nanmean(R[iu]))
    else:
        fluorosnapp_fc = np.nan

    return dict(
        fluorosnapp_sync = fluorosnapp_sync,
        fluorosnapp_fc   = fluorosnapp_fc,
    )


# ── Metric names for downstream consumption ──────────────────────────────────
BURST_METRIC_NAMES = (
    "burst_adj_freq", "burst_freq", "burstiness",
    "mean_burst_size", "frac_in_bursts",
)
SYNC_METRIC_NAMES = (
    "fluorosnapp_sync", "fluorosnapp_fc",
)
