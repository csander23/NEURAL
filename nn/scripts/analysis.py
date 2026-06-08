"""Post-segmentation analysis: trace extraction, EMA detrending, peak
detection (PeakCaller), per-ROI metrics, synchronicity, plus a batch driver.

Most functions are ported verbatim from `EXPORT FOR METHODS PAPER/scripts/
04_analysis_utils.py` (model-agnostic). The ROI-detection step is replaced
by a wrapper that calls our FlowField inference pipeline.

Public entry points
-------------------
- `batch_process(input_dir, output_dir, run_dir, params)` — process every
  .nd2/.tif/.tiff in `input_dir`, write per-file results under
  `output_dir/<base>/`, return a summary DataFrame.
- `process_single_file(...)` — same for one file.
- The lower-level building blocks (`detect_rois_flowfield`, `extract_trace`,
  `get_trend_two_sided_ema`, `detect_peaks_peakcaller`,
  `calculate_roi_metrics`, `compute_synchronicity`) can be imported
  individually if you want to assemble a custom pipeline.

Output schema per file
----------------------
    <output_dir>/<base>/
    ├── _summary.csv              one row per ROI: n_peaks, freq, SNR, etc.
    ├── roi_masks.npy             (N, H, W) bool — same shape as input
    ├── synchronicity_matrix.npy  (N, N) float — pairwise sync values
    └── roi_<i>/
        ├── raw_trace.npy
        ├── detrended_trace.npy
        ├── peaks.npy             frame indices of detected peaks
        └── trace_plot.png
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config
from scripts.resnet_unet import ResNetUNet
from scripts.approach_flowfield import predict_full_image, flowfield_instances
from scripts.summary_images import (
    summary_from_path,
    summary_from_video,            # kept for callers that already have a numpy video
    _iter_nd2_frames, _iter_tiff_frames,
    _load_video,                   # back-compat re-export for downstream helpers
)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming frame access (one file pass = one open + iterate)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_frames(path: Path):
    """Open a video and yield (frame_iterator, H, W, pixel_size_um).

    Routes through summary_images' lazy readers so we never materialise the
    full (T, H, W) array. Use this anywhere we'd otherwise call _load_video.
    """
    suf = path.suffix.lower()
    if suf == ".nd2":
        return _iter_nd2_frames(path)
    if suf in (".tif", ".tiff"):
        return _iter_tiff_frames(path)
    raise ValueError(f"unsupported video format: {suf}")


def extract_traces_streaming(
    path: Path, masks: np.ndarray,
) -> np.ndarray:
    """One pass over the file → (N, T) raw fluorescence traces.

    Equivalent to `[extract_trace(video, masks[i]) for i in range(N)]` but
    streams frames instead of holding the full video in RAM. Peak memory is
    dominated by the flattened mask matrix (N * H * W * 4 bytes), which for
    N=200, H=W=1192 is ~1.1 GB — fine.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (N, H, W); got {masks.shape}")
    N, H, W = masks.shape
    if N == 0:
        # Need T to return correct shape; peek the iterator once
        frames, _, _, _ = _iter_frames(path)
        T = sum(1 for _ in frames)
        return np.zeros((0, T), dtype=np.float32)

    masks_flat = masks.reshape(N, -1).astype(np.float32, copy=False)
    areas = masks_flat.sum(axis=1).clip(min=1.0).astype(np.float32)  # (N,)

    frames, _H, _W, _px = _iter_frames(path)
    if (_H, _W) != (H, W):
        raise ValueError(
            f"mask shape {(H, W)} doesn't match video frame shape {(_H, _W)}"
        )

    per_frame: list[np.ndarray] = []
    for frame in frames:
        flat = np.asarray(frame, dtype=np.float32).reshape(-1)
        per_frame.append(masks_flat @ flat / areas)   # (N,)
    if not per_frame:
        return np.zeros((N, 0), dtype=np.float32)
    return np.stack(per_frame, axis=1).astype(np.float32)   # (N, T)


def _zscore_chw(summary_hw3: np.ndarray) -> np.ndarray:
    out = np.empty_like(summary_hw3, dtype=np.float32)
    for c in range(summary_hw3.shape[-1]):
        ch = summary_hw3[..., c].astype(np.float32)
        out[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out.transpose(2, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ROI detection via FlowField (replaces Mask R-CNN detect_rois)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    model: ResNetUNet
    config: Config
    device: "object"


def load_run(run_dir: Path) -> LoadedModel:
    """Reconstruct a Config + ResNetUNet from a trained run directory."""
    import torch
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg_data = json.load(f)
    allowed = {k: v for k, v in cfg_data.items() if k in Config.__dataclass_fields__}
    config = Config(**allowed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(
        out_ch=3,
        pretrained=False,
        apply_imagenet_norm=config.apply_imagenet_norm,
    ).to(device)
    model.load_state_dict(torch.load(
        run_dir / "checkpoints" / "best_model.pth",
        map_location=device, weights_only=True,
    ))
    model.eval()
    return LoadedModel(model=model, config=config, device=device)


def detect_rois_flowfield(
    summary_rgb: np.ndarray,
    loaded: LoadedModel,
    *,
    input_pixel_size_um: float | None = None,
    apply_pixel_resize: bool = False,
) -> np.ndarray:
    """Run FlowField on a (H, W, 3) summary image, return (M, H, W) bool masks.

    Pixel-size resize behavior (controlled by `apply_pixel_resize`):
      False (default, matches the 2026-05-22 cache): infer at native resolution.
            The model is applied to the raw 1192x1192 image regardless of
            pixel-size mismatch between input and training scale.
      True : if both `loaded.config.training_pixel_size_um` and
            `input_pixel_size_um` are known and differ, resize input to match
            the model's training scale before inference and resize masks back.
            This produces dramatically more (and smaller) ROIs at upscale > 1
            and is the source of the 9.25.25 over-detection saga.
    """
    from skimage.transform import resize as sk_resize
    config = loaded.config
    img = _zscore_chw(summary_rgb)
    H_orig, W_orig = img.shape[1:]

    work_hw = (H_orig, W_orig)
    if apply_pixel_resize and input_pixel_size_um and config.training_pixel_size_um:
        scale = float(input_pixel_size_um) / float(config.training_pixel_size_um)
        if abs(scale - 1.0) > 0.01:
            new_h = max(64, int(round(H_orig * scale)))
            new_w = max(64, int(round(W_orig * scale)))
            img = sk_resize(
                img.transpose(1, 2, 0), (new_h, new_w),
                preserve_range=True, anti_aliasing=True,
            ).astype(np.float32).transpose(2, 0, 1)
            work_hw = (new_h, new_w)

    fg, dy, dx = predict_full_image(
        loaded.model, img,
        tile_size=config.val_tile_size,
        overlap=config.val_tile_overlap,
        device=loaded.device,
    )
    masks = flowfield_instances(
        fg, dy, dx,
        fg_threshold=config.ff_fg_threshold,
        n_steps=config.ff_n_steps,
        step_size=config.ff_step_size,
        min_distance=config.ff_min_distance,
        vote_threshold=config.ff_vote_threshold,
        min_roi_px=config.ff_min_roi_px,
        compactness=config.watershed_compactness,
    )

    if work_hw != (H_orig, W_orig) and masks.shape[0] > 0:
        n_masks = masks.shape[0]
        out = np.empty((n_masks, H_orig, W_orig), dtype=bool)
        for i in range(n_masks):
            r = sk_resize(
                masks[i].astype(np.float32),
                (H_orig, W_orig),
                preserve_range=True, anti_aliasing=False, order=0,
            )
            out[i] = r > 0.5
        masks = out

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Trace extraction / detrending / peak detection — model-agnostic
# (ported from 04_analysis_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_trace(video: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean fluorescence over ROI per frame. video: (T,H,W), mask: (H,W) bool."""
    m = mask.astype(bool)
    area = max(int(m.sum()), 1)
    return (video.astype(np.float32) * m[None, :, :]).sum(axis=(1, 2)) / area


def get_trend_two_sided_ema(signal: np.ndarray, smoothness: float = 1500.0) -> np.ndarray:
    """Two-sided exponential moving average baseline.

    Kept for back-compat. As of 2026-05-20 the default detrending used by
    process_single_file is the convex envelope (matches the manual side
    of the methods-paper comparison).
    """
    alpha = 2.0 / (smoothness + 1.0)
    fwd = np.zeros_like(signal, dtype=np.float64)
    fwd[0] = signal[0]
    for i in range(1, len(signal)):
        fwd[i] = alpha * signal[i] + (1 - alpha) * fwd[i - 1]
    bwd = np.zeros_like(signal, dtype=np.float64)
    bwd[-1] = signal[-1]
    for i in range(len(signal) - 2, -1, -1):
        bwd[i] = alpha * signal[i] + (1 - alpha) * bwd[i + 1]
    return ((fwd + bwd) / 2.0).astype(signal.dtype)


def convex_envelope_baseline(x: np.ndarray) -> np.ndarray:
    """PeakCaller's "Convex Envelope" trend option (parameter-free).

    The lower convex hull underneath the trace, computed by a left-to-right
    3-point convexity scan. Same algorithm as helpers/_peakcaller.py to keep
    the manual and NN sides apples-to-apples.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()
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


def detrend_ratio(signal: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """Return PeakCaller D-scale: D = F / F₀ (baseline ≈ 1).

    Previously this subtracted 1 to return ΔF/F₀ (baseline ≈ 0). That was
    incompatible with the MATLAB-faithful multiplicative `detect_peaks_peakcaller`
    below, which expects D-scale where peaks rise above 1. The saved
    detrended_trace.npy files in the predictions tree are already on D-scale,
    so this aligns the code with the data.
    """
    return signal.astype(np.float64) / (trend.astype(np.float64) + 1e-12)


def detect_peaks_peakcaller(
    signal: np.ndarray,
    required_rise: float = 0.02,
    required_fall: float = 0.02,
    max_lookback: int = 26,
    max_lookahead: int = 26,
    detect_negative: bool = False,
    # Kept for back-compat with old call sites; ignored.
    min_amplitude: float | None = None,
) -> tuple[np.ndarray, dict]:
    """MATLAB-faithful multiplicative PeakCaller (Artimovich et al. 2017).

    `signal` must be on D-scale (baseline ≈ 1, i.e. F/F₀). Thresholds are
    relative to the candidate peak's D-value:
        accept iff D[j_before] < (1 - required_rise) * D[peak]
        accept iff D[j_after]  < (1 - required_fall) * D[peak]

    Two-pass with backward deletion. Replaces the earlier additive port that
    used absolute frame-to-frame differences plus a min_amplitude floor.
    """
    D = (2.0 - signal.astype(np.float64)) if detect_negative else signal.astype(np.float64)
    T = D.size

    # Strict local maxima
    cand = np.zeros(T, dtype=bool)
    if T >= 3:
        cand[1:-1] = (D[1:-1] > D[:-2]) & (D[1:-1] > D[2:])

    peaks, j_before_idx, j_after_idx, amps = [], [], [], []
    prior = 0

    # Forward pass
    for i in np.where(cand)[0]:
        if i < 1 or i > T - 2:
            continue
        lb_s = max(prior, i - max_lookback)
        if lb_s >= i:
            continue
        j_before = lb_s + int(np.argmin(D[lb_s:i]))
        if D[j_before] >= (1.0 - required_rise) * D[i]:
            continue
        ahead_end = min(T, i + max_lookahead + 1)
        ahead = D[i + 1: ahead_end]
        if ahead.size == 0:
            continue
        too_tall = np.where(ahead > D[i])[0]
        stop = (i + 1) + (int(too_tall[0]) if too_tall.size else ahead.size)
        j_after = (i + 1) + int(np.argmin(D[i + 1: stop]))
        if D[j_after] >= (1.0 - required_fall) * D[i]:
            continue
        peaks.append(i); j_before_idx.append(j_before); j_after_idx.append(j_after)
        amps.append(D[i] - 1.0)
        prior = i

    # Backward pass: re-check fall against the *next confirmed peak* and
    # delete peaks whose post-peak minimum isn't deep enough under that
    # tighter bound.
    if peaks:
        keep = [True] * len(peaks)
        next_peak = T
        for k in range(len(peaks) - 1, -1, -1):
            i = peaks[k]
            lookforward_end = min(next_peak, i + max_lookback + 1)
            if lookforward_end <= i + 1:
                keep[k] = False; continue
            j_after = (i + 1) + int(np.argmin(D[i + 1: lookforward_end]))
            if D[j_after] >= (1.0 - required_fall) * D[i]:
                keep[k] = False; continue
            j_after_idx[k] = j_after
            next_peak = i
        peaks        = [v for v, k in zip(peaks,        keep) if k]
        j_before_idx = [v for v, k in zip(j_before_idx, keep) if k]
        j_after_idx  = [v for v, k in zip(j_after_idx,  keep) if k]
        amps         = [v for v, k in zip(amps,         keep) if k]

    amp_arr = np.array(amps, dtype=float)
    if detect_negative:
        amp_arr = -amp_arr
    return np.array(peaks, dtype=int), {
        "amplitudes": amp_arr,
        "onsets":     np.array(j_before_idx, dtype=int),
        "offsets":    np.array(j_after_idx,  dtype=int),
    }


def calculate_roi_metrics(
    peaks: np.ndarray, properties: dict, trace: np.ndarray, sampling_rate: float,
) -> dict:
    n = len(peaks)
    duration = len(trace) / max(sampling_rate, 1e-9)
    trace_std = float(np.std(trace))
    return {
        "n_peaks":        n,
        "frequency_hz":   float(n / duration) if duration > 0 else 0.0,
        "mean_amplitude": float(np.mean(properties["amplitudes"])) if n > 0 else 0.0,
        "std_amplitude":  float(np.std(properties["amplitudes"]))  if n > 0 else 0.0,
        "trace_mean":     float(np.mean(trace)),
        "trace_std":      trace_std,
        "trace_snr":      float(np.mean(properties["amplitudes"]) / trace_std)
                              if (n > 0 and trace_std > 0) else 0.0,
    }


def compute_synchronicity(all_peaks: list[np.ndarray], window: int = 5) -> np.ndarray:
    N = len(all_peaks)
    sync = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        sync[i, i] = 1.0
        for j in range(i + 1, N):
            pi, pj = all_peaks[i], all_peaks[j]
            if len(pi) == 0 or len(pj) == 0:
                continue
            count = sum(int(np.any(np.abs(pj - p) <= window)) for p in pi)
            v = count / len(pi)
            sync[i, j] = sync[j, i] = v
    return sync


# ─────────────────────────────────────────────────────────────────────────────
# Per-file + batch drivers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisParams:
    sampling_rate: float = 180.0
    trend_smoothness: float = 1500.0
    # Multiplicative MATLAB-faithful PeakCaller thresholds (fraction of D[peak])
    required_rise: float = 0.02
    required_fall: float = 0.02
    # max_lookback / max_lookahead lowered from 50 -> 26 to match MATLAB default
    max_lookback: int = 26
    max_lookahead: int = 26
    detect_negative: bool = False
    sync_window: int = 5
    save_trace_plots: bool = True
    # Deprecated: kept so legacy config / call sites don't break, ignored by
    # the multiplicative algorithm.
    min_amplitude: float | None = None


def _plot_trace(raw: np.ndarray, detrended: np.ndarray, peaks: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(raw, "k-", linewidth=0.5)
    axes[0].set_ylabel("raw F")
    axes[0].grid(alpha=0.3)
    axes[1].plot(detrended, "b-", linewidth=0.5)
    if len(peaks) > 0:
        axes[1].plot(peaks, detrended[peaks], "ro", markersize=4)
    axes[1].set_ylabel("ΔF/F₀")
    axes[1].set_xlabel("frame")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_single_file(
    video_path: Path,
    run_dir: Path,
    output_dir: Path,
    *,
    params: AnalysisParams = AnalysisParams(),
    loaded: LoadedModel | None = None,
    pixel_size_um_override: float | None = None,
) -> dict:
    base = video_path.stem
    out_base = output_dir / base
    out_base.mkdir(parents=True, exist_ok=True)

    if loaded is None:
        loaded = load_run(run_dir)

    # Pass 1 over the file: build the (H, W, 3) summary, streaming.
    summary, input_px = summary_from_path(video_path)
    if pixel_size_um_override is not None:
        input_px = pixel_size_um_override

    masks = detect_rois_flowfield(summary, loaded, input_pixel_size_um=input_px)
    print(f"  {base}: {masks.shape[0]} ROI(s) at {summary.shape[:2]}")
    np.save(out_base / "roi_masks.npy", masks)

    # Pass 2 over the file: per-ROI raw traces, streaming.
    raw_traces = (
        extract_traces_streaming(video_path, masks)
        if masks.shape[0] > 0 else np.zeros((0, 0), dtype=np.float32)
    )

    per_roi_rows = []
    all_peaks: list[np.ndarray] = []
    for i, m in enumerate(masks):
        roi_dir = out_base / f"roi_{i+1:03d}"
        roi_dir.mkdir(exist_ok=True)
        raw = raw_traces[i]
        # CONVEX ENVELOPE baseline — matches the manual side of the methods-paper
        # comparison so both pipelines are using the same detrending algorithm.
        # (Use get_trend_two_sided_ema(raw, params.trend_smoothness) if you ever
        #  want to revert to the EMA baseline.)
        trend = convex_envelope_baseline(raw)
        detrended = detrend_ratio(raw, trend)
        peaks, props = detect_peaks_peakcaller(
            detrended,
            required_rise=params.required_rise,
            required_fall=params.required_fall,
            max_lookback=params.max_lookback,
            max_lookahead=params.max_lookahead,
            detect_negative=params.detect_negative,
        )
        metrics = calculate_roi_metrics(peaks, props, detrended, params.sampling_rate)
        metrics["roi_id"] = i + 1
        per_roi_rows.append(metrics)
        all_peaks.append(peaks)
        np.save(roi_dir / "raw_trace.npy", raw)
        np.save(roi_dir / "detrended_trace.npy", detrended)
        np.save(roi_dir / "peaks.npy", peaks)
        if params.save_trace_plots:
            _plot_trace(raw, detrended, peaks, roi_dir / "trace_plot.png")

    sync = compute_synchronicity(all_peaks, window=params.sync_window)
    np.save(out_base / "synchronicity_matrix.npy", sync)

    # Per-file summary CSV
    import csv
    if per_roi_rows:
        with open(out_base / "_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_roi_rows[0].keys()))
            w.writeheader()
            w.writerows(per_roi_rows)

    return {
        "base":   base,
        "n_rois": int(masks.shape[0]),
        "rows":   per_roi_rows,
    }


def batch_process(
    input_dir: Path,
    output_dir: Path,
    run_dir: Path,
    *,
    params: AnalysisParams = AnalysisParams(),
    extensions: tuple[str, ...] = (".nd2", ".tif", ".tiff"),
    resume: bool = True,
):
    """Process every video in `input_dir`. Returns a pandas DataFrame summary
    (one row per file) if pandas is available, else a list of dicts."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_run(Path(run_dir))

    rows: list[dict] = []
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]
    for f in files:
        base = f.stem
        out_base = output_dir / base
        if resume and (out_base / "_summary.csv").exists():
            print(f"  skip (already analyzed): {base}")
            continue
        try:
            res = process_single_file(f, Path(run_dir), output_dir, params=params, loaded=loaded)
            rows.append({"file": base, "n_rois": res["n_rois"]})
        except Exception as e:
            print(f"  ERROR on {base}: {e!r}")
            traceback.print_exc()
            rows.append({"file": base, "n_rois": -1, "error": repr(e)})

    # Save batch summary as CSV always; try pandas for nicer return
    import csv
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(output_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows


def scan_for_invalid(input_dir: Path, extensions: tuple[str, ...] = (".nd2", ".tif", ".tiff")) -> dict:
    """Quick sanity scan: list files in input_dir, flag any that can't be opened.

    Streams 1-2 frames instead of loading the full video, so this is fast and
    cannot OOM on long recordings.
    """
    input_dir = Path(input_dir)
    valid, invalid = [], []
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() not in extensions:
            continue
        try:
            frames, H, W, _ = _iter_frames(f)
            # Pull one frame to confirm the file is readable end-to-end.
            next(iter(frames))
            valid.append({"name": f.name, "shape": [H, W]})
        except Exception as e:
            invalid.append({"name": f.name, "error": repr(e)})
    return {"valid": valid, "invalid": invalid}
