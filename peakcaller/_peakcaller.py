"""PeakCaller-faithful trace analysis (Artimovich et al., BMC Neurosci 2017).

PubMed: https://pubmed.ncbi.nlm.nih.gov/29037171/

These functions are copied verbatim from the Python port at
``N:\\Methods Paper\\Novel_Analysis\\actual_exports\\experiments post NN
parameterization\\experiments_peakdetection\\newpeakcaller_05.11.2026\\analysis.py``
which is itself a faithful translation of the Hussman Institute MATLAB
``PeakCaller.m`` script (forward + backward pass with explicit peak deletion,
multiplicative D-scale thresholds, half-before/half-after computed during
detection).

This file replaces the earlier near-PeakCaller port in EXPORT FOR METHODS
PAPER/scripts/analysis.py, which used additive thresholds on ΔF/F₀ and had
no backward-pass peak deletion. The newer version below matches PeakCaller's
actual algorithm on the D-scale (D = signal/trend, baseline ≈ 1).

If anything in newpeakcaller_05.11.2026/analysis.py changes upstream, re-sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ── Detrending ───────────────────────────────────────────────────────────────
#
# PeakCaller (Artimovich et al., BMC Neurosci 2017) defines its detrended
# trace as D = Observations / Smoothed (baseline ≈ 1) and runs all peak
# math on D directly. We follow that convention so the multiplicative
# rise/fall thresholds in find_peaks operate on the same scale as MATLAB.
# An earlier version of this module subtracted 1 (dF/F0 with baseline 0);
# that scale is incompatible with the multiplicative test, so detrend_ratio
# no longer subtracts.

def one_sided_ema(x, smoothness=60):
    # PeakCaller's "Exponential Moving Average (1-sided)" trend option.
    # Causal EMA: at time t uses only data up to t. PeakCaller recommends
    # this when culture conditions change over time (e.g., pharmacology).
    p = 2.0 / (smoothness + 1)
    q = 1.0 - p
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = p * x[t] + q * y[t - 1]
    return y


def two_sided_ema(x, smoothness=60):
    # PeakCaller's "Exponential Moving Average (2-sided)" option:
    # mean of forward and backward one-sided EMAs.
    # PeakCaller recommends this when culture conditions remain static.
    p = 2.0 / (smoothness + 1)
    q = 1.0 - p
    fwd = np.empty_like(x, dtype=float)
    bwd = np.empty_like(x, dtype=float)
    fwd[0] = x[0]
    for t in range(1, len(x)):
        fwd[t] = p * x[t] + q * fwd[t - 1]
    bwd[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        bwd[t] = p * x[t] + q * bwd[t + 1]
    return 0.5 * (fwd + bwd)


def diffusion_smooth(x, smoothness=60):
    # PeakCaller's "Finite Difference Diffusion" option: numerical heat
    # equation with insulated ends, iterated 4 * smoothness times. The
    # effective weight kernel is approximately Gaussian.
    x = x.astype(float).copy()
    for _ in range(4 * smoothness):
        x[1:-1] += 0.25 * (x[2:] - 2 * x[1:-1] + x[:-2])
        x[0] += 0.5 * (x[1] - x[0])
        x[-1] += 0.5 * (x[-2] - x[-1])
    return x


def als_baseline(y, *, lam=1e7, p=0.99, n_iter=10):
    """ALS asymmetric-least-squares baseline (Eilers 2003).

    Used as the alternative to convex_envelope_baseline for negative-going
    indicators (e.g. iGABASnFr) where the convex envelope wraps the wrong
    side of the trace.
    """
    import numpy as _np
    from scipy import sparse as _sparse
    from scipy.sparse.linalg import spsolve as _spsolve
    y = _np.asarray(y, dtype=_np.float64)
    L = len(y)
    if L < 3:
        return y.copy()
    D = _sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).tocsc()
    DTD = lam * (D.T @ D)
    w = _np.ones(L)
    z = y.copy()
    for _ in range(n_iter):
        W = _sparse.diags(w, 0, shape=(L, L), format="csc")
        z = _spsolve(W + DTD, w * y)
        w = _np.where(y > z, p, 1.0 - p)
    return z


def convex_envelope_baseline(x):
    # PeakCaller's "Convex Envelope" option (parameter-free): the lower
    # convex hull underneath the trace, used as the trend.
    # Implementation here is a left-to-right 3-point convexity scan, which
    # yields the same geometric envelope as PeakCaller's secant-slope walk.
    n = len(x)
    idx = [0]
    for i in range(1, n):
        idx.append(i)
        while len(idx) >= 3:
            x1, x2, x3 = idx[-3:]
            y1, y2, y3 = x[x1], x[x2], x[x3]
            if (y3 - y1) * (x2 - x1) <= (y2 - y1) * (x3 - x1):
                del idx[-2]
            else:
                break
    return np.interp(np.arange(n), idx, x[idx])


def get_trend(signal, method="two_sided_ema", smoothness=60):
    # Selector across PeakCaller's four trend methods plus "none"
    # (PeakCaller's "No Trend" option: trend = mean of the trace, so D has
    # mean 1 but preserves shape).
    if method == "two_sided_ema":
        return two_sided_ema(signal, smoothness)
    if method == "one_sided_ema":
        return one_sided_ema(signal, smoothness)
    if method == "diffusion":
        return diffusion_smooth(signal, smoothness)
    if method == "convex":
        return convex_envelope_baseline(signal)
    if method == "none":
        return np.full_like(signal, signal.mean())
    raise ValueError(f"Unknown detrending method '{method}'")


def detrend_ratio(signal, trend):
    # PeakCaller: D = signal / trend. Baseline ≈ 1, peaks rise above 1.
    # No "- 1" here — keeping PeakCaller's D-scale so the multiplicative
    # thresholds in find_peaks behave identically to the MATLAB script.
    trend = np.where(trend == 0, np.finfo(float).eps, trend)
    return signal / trend


def smooth_trace(trace, window_length=5, polyorder=1):
    # OPTIONAL Savitzky-Golay smoother. NOT part of PeakCaller; Python add-on.
    # Default usage: do not call. Apply manually to the detrended trace
    # before find_peaks only if shot noise is high enough to split otherwise-
    # clean peaks into multiple strict local maxima.
    if len(trace) < window_length:
        return trace
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(trace, window_length, polyorder)


# ════════════════════════════════════════════════════════════════
#  Peak detection — PeakCaller-faithful
# ════════════════════════════════════════════════════════════════
#
# Faithful to the MATLAB PeakCaller script (Artimovich et al., 2017):
#   - Threshold form is MULTIPLICATIVE (a fraction of the candidate peak's
#     D-value), not additive. PeakCaller's rule (rewritten in Python):
#         accept iff D[j_before] < (1 - required_rise) * D[peak]
#         accept iff D[j_after]  < (1 - required_fall) * D[peak]
#   - Two passes: a FORWARD pass nominates candidates passing rise + an
#     initial fall test, then a BACKWARD pass re-evaluates fall against
#     the *next confirmed peak* and DELETES peaks whose post-peak minimum
#     isn't deep enough under that tighter bound.
#   - Half-before / half-after indices follow PeakCaller exactly:
#         half-before = last index in [lb_s, peak) below 0.5 * (D_min + D_peak)
#         half-after  = first index in (peak, j_after] below 0.5 * (D_peak + D_min)
#
# Python-only additions kept:
#   - detect_negative: PeakCaller has no inverted-trace mode. Implemented
#     here by reflecting D around its baseline (D -> 2 - D) so negative-going
#     events become positive ones, then running the standard logic.
#
# Python-only features dropped to stay faithful to PeakCaller:
#   - min_amplitude (absolute-value floor on candidate value) — not in
#     PeakCaller, and largely redundant given that the multiplicative test
#     scales the required drop with the peak's D-value.

def find_peaks(trace, required_rise=0.10, required_fall=0.10,
               max_lookback=26, max_lookahead=26,
               relative_to_range=False, detect_negative=False):
    """
    PeakCaller-faithful peak detection.

    `trace` must be on PeakCaller's D-scale (baseline ≈ 1) — i.e. the
    output of detrend_ratio() in this module. required_rise / required_fall
    are fractions of the candidate peak's D-value (multiplicative form).

    Returns (peaks, heights, baselines, half_before, half_after).
        peaks      — indices of confirmed peaks (after the backward pass)
        heights    — dF/F0 = D[peak] - 1 at each peak (sign-flipped for
                     detect_negative so it reads as the original polarity)
        baselines  — dF/F0 at the immediately preceding local minimum
        half_before, half_after — indices of half-height crossings
    """
    # PeakCaller has no negative-peak mode; this is the Python add-on.
    # Reflecting around D = 1 keeps the baseline at 1 so the multiplicative
    # thresholds remain valid without re-parameterization.
    D = (2.0 - trace.astype(float)) if detect_negative else trace.astype(float)
    T = D.size

    # PeakCaller "Relative to Range" option: scale the threshold by the
    # observed dynamic range of D, applied multiplicatively just like the
    # default (so RelativeToRange is a per-trace scaling of required_rise/fall).
    if relative_to_range:
        rng = D.max() - D.min()
        rise_thr = required_rise * rng
        fall_thr = required_fall * rng
    else:
        rise_thr = required_rise
        fall_thr = required_fall

    # PeakCaller "Candidate" step: strict local maximum (greater than both neighbors).
    cand = np.zeros(T, dtype=bool)
    cand[1:-1] = (D[1:-1] > D[:-2]) & (D[1:-1] > D[2:])

    peaks, j_before_idx, j_after_idx, hb, ha = [], [], [], [], []
    prior = 0

    # ── Forward pass ─────────────────────────────────────────────────
    # PeakCaller `for jj=2:T-1`. For each strict local max, find the min in
    # [max(prior_peak, i - max_lookback), i) and apply the multiplicative
    # rise test. Then find the min in (i, min(T, i + max_lookahead + 1))
    # capped at the first index where D exceeds D(i), and apply the
    # multiplicative fall test. Peaks passing both are provisionally accepted.
    for i in np.where(cand)[0]:
        if i < 1 or i > T - 2:
            continue
        lb_s = max(prior, i - max_lookback)
        if lb_s >= i:
            continue
        j_before = lb_s + int(np.argmin(D[lb_s:i]))

        if D[j_before] >= (1.0 - rise_thr) * D[i]:
            continue

        ahead_end = min(T, i + max_lookahead + 1)
        ahead = D[i + 1: ahead_end]
        if ahead.size == 0:
            continue
        too_tall = np.where(ahead > D[i])[0]
        stop = (i + 1) + (int(too_tall[0]) if too_tall.size else ahead.size)
        j_after = (i + 1) + int(np.argmin(D[i + 1: stop]))

        if D[j_after] >= (1.0 - fall_thr) * D[i]:
            continue

        peaks.append(i)
        j_before_idx.append(j_before)
        j_after_idx.append(j_after)

        # PeakCaller half-before: last index in [lb_s, i) below midpoint(MinBefore, D_peak).
        hlvl = 0.5 * (D[j_before] + D[i])
        below = np.where(D[lb_s:i] < hlvl)[0]
        hb.append(lb_s + int(below[-1]) if below.size else j_before)
        ha.append(j_after)  # placeholder — finalized in the backward pass

        prior = i

    # ── Backward pass ────────────────────────────────────────────────
    # PeakCaller `for jj=T-1:-1:2`. Walk peaks from last to first; for each,
    # recompute the minimum in (i, next_confirmed_peak], capped by
    # i + max_lookback (in the MATLAB source, the lookback variable name is
    # reused here — preserved verbatim for fidelity). Apply the multiplicative
    # fall test against this tighter bound. Peaks failing the test are DELETED.
    # Surviving peaks get their final half-after index from this stricter min.
    if peaks:
        keep = [True] * len(peaks)
        next_peak = T
        for k in range(len(peaks) - 1, -1, -1):
            i = peaks[k]
            lookforward_end = min(next_peak, i + max_lookback + 1)
            if lookforward_end <= i + 1:
                keep[k] = False
                continue
            j_after = (i + 1) + int(np.argmin(D[i + 1: lookforward_end]))
            if D[j_after] >= (1.0 - fall_thr) * D[i]:
                keep[k] = False
                continue
            j_after_idx[k] = j_after
            # PeakCaller half-after: first index in (i, j_after] below midpoint(D_peak, MinAfter).
            hlvl2 = 0.5 * (D[i] + D[j_after])
            below = np.where(D[i + 1: j_after + 1] < hlvl2)[0]
            ha[k] = (i + 1) + int(below[0]) if below.size else j_after
            next_peak = i

        peaks        = [v for v, k in zip(peaks,        keep) if k]
        j_before_idx = [v for v, k in zip(j_before_idx, keep) if k]
        j_after_idx  = [v for v, k in zip(j_after_idx,  keep) if k]
        hb           = [v for v, k in zip(hb,           keep) if k]
        ha           = [v for v, k in zip(ha,           keep) if k]

    pk_arr = np.array(peaks, int)
    if pk_arr.size:
        # PeakCaller reports peak heights as D - 1 (its histogram code subtracts 1).
        heights = D[pk_arr] - 1.0
        baselines = D[np.array(j_before_idx, int)] - 1.0
        if detect_negative:
            # Report in the ORIGINAL trace's polarity (so a downward event has
            # a negative height — same sign convention as the input data).
            heights = -heights
            baselines = -baselines
    else:
        heights = np.array([], float)
        baselines = np.array([], float)

    return (pk_arr, heights, baselines,
            np.array(hb, int), np.array(ha, int))


# ════════════════════════════════════════════════════════════════
#  Per-peak metrics
# ════════════════════════════════════════════════════════════════
#
# PeakCaller-faithful metrics (the three columns its histograms and
# spreadsheet export report):
#   amplitude — D[peak] - 1 (dF/F0 above baseline). PeakCaller's "Height".
#   rise_hw   — (peak_idx - half_before_idx) / sampling_rate. PeakCaller's
#               "Rise Time (half max to max)".
#   decay_hw  — (half_after_idx - peak_idx) / sampling_rate. PeakCaller's
#               "Decay Time (max to half max)".
#   width     — rise_hw + decay_hw. PeakCaller's "FWHM".
#
# Critical: these use the half_before / half_after indices that find_peaks
# set during detection — i.e., crossings of the midpoint between the local
# MIN and the peak (PeakCaller's definition). This differs from an amp/2.0
# rescan, which is what the earlier Python version did and which only
# matches PeakCaller when the local minimum sits exactly at baseline.
#
# Python-only additions kept:
#   snr — std of the trace in a window around the peak. Diagnostic only;
#         not present in PeakCaller and NOT used to filter peaks here.

_METRIC_COLUMNS = [
    "peak_idx", "amplitude", "rise_hw", "decay_hw", "width", "snr",
]


def calculate_metrics(trace, peaks, hb_idx, ha_idx,
                      snr_window=20, sampling_rate=1.0,
                      detect_negative=False):
    """
    `trace`  — detrended D-scale series (baseline ≈ 1).
    `peaks`, `hb_idx`, `ha_idx` — index arrays from find_peaks.
    """
    if len(peaks) == 0:
        return pd.DataFrame(columns=_METRIC_COLUMNS)
    peaks = np.asarray(peaks, int)
    hb_idx = np.asarray(hb_idx, int)
    ha_idx = np.asarray(ha_idx, int)
    T = trace.size
    rows = []
    for k, idx in enumerate(peaks):
        # dF/F0 at peak; sign-flipped for the detect_negative path so the
        # reported amplitude matches the original trace's polarity.
        amp = trace[idx] - 1.0
        if detect_negative:
            amp = -amp
        rise_hw = (idx - hb_idx[k]) / sampling_rate
        decay_hw = (ha_idx[k] - idx) / sampling_rate
        # Diagnostic SNR (Python add-on, not used for filtering).
        lo, hi = max(0, idx - snr_window), min(T, idx + snr_window + 1)
        noise = np.std(trace[lo:hi])
        snr = abs(amp) / (noise + 1e-12)
        rows.append([idx, amp, rise_hw, decay_hw, rise_hw + decay_hw, snr])
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


# ── Synchronicity ────────────────────────────────────────────────────────────

def _peak_regions(T, hb, ha):
    diff = hb.astype(np.int8) - ha.astype(np.int8)
    return 2 * np.cumsum(diff, dtype=np.int16) - 1


def _autocorr(rv):
    T = rv.size
    R = rv[:, None] * rv[None, :]
    return np.array([R.diagonal(k).mean() for k in range(1, T)])


def calculate_synchronicity(hb_list, ha_list):
    if not hb_list:
        return np.zeros((0, 0), np.float32)
    P = np.vstack([
        _autocorr(_peak_regions(hb.size, hb, ha))
        for hb, ha in zip(hb_list, ha_list)
    ])
    P -= P.mean(axis=1, keepdims=True)
    P /= P.std(axis=1, keepdims=True) + 1e-12
    S = (P @ P.T) / (P.shape[1] - 1)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)
