"""Single-dataset NN-side renderer.

Given a config (JSON path or dict), renders one or more of:
  - trace_grid:    N_rois traces per video, picked by activity / random / index
  - roi_overlay:   ROI contours over the mean projection
  - summary_panel: brief per-recording stats card (n_rois, mean_freq, etc.)
  - all:           all three of the above

Reads NN ROIs from a cached inference dir if provided, otherwise runs inference
on the input(s). Detects peaks with PeakCaller. Honors fps / pixel_size /
n_frames / duration overrides from the input section.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PeakDetectionCfg:
    threshold: float = 0.03
    baseline: str = "convex"     # "convex" | "als"
    negate: bool = False


@dataclass
class NNCfg:
    model: str = "iglusnfr"
    min_roi_px: int = 32
    fg_threshold: float = 0.7
    use_cached_inference: bool = True


@dataclass
class VisualizationCfg:
    kind: str = "all"            # "trace_grid" | "roi_overlay" | "summary_panel" | "all"
    n_rois_per_video: int = 10
    roi_selection: str = "top_activity"   # "top_activity" | "random" | "all" | "by_index"
    roi_indices: Optional[list] = None
    videos_to_show: Optional[list] = None


@dataclass
class InputCfg:
    path: str = ""
    format: str = "auto"         # "auto" | "nd2" | "tif"
    inference_dir: Optional[str] = None
    fps: Optional[float] = None
    pixel_size_um: Optional[float] = None
    n_frames: Optional[int] = None
    duration_s: Optional[float] = None


@dataclass
class OutputCfg:
    dir: str = "./output"
    format: str = "pdf"          # "pdf" | "png"
    dpi: int = 140


@dataclass
class DatasetConfig:
    peak_detection: PeakDetectionCfg = field(default_factory=PeakDetectionCfg)
    nn: NNCfg = field(default_factory=NNCfg)
    visualization: VisualizationCfg = field(default_factory=VisualizationCfg)
    input: InputCfg = field(default_factory=InputCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            peak_detection=PeakDetectionCfg(**d.get("peak_detection", {})),
            nn=NNCfg(**d.get("nn", {})),
            visualization=VisualizationCfg(**d.get("visualization", {})),
            input=InputCfg(**d.get("input", {})),
            output=OutputCfg(**d.get("output", {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_recordings(input_cfg: InputCfg) -> list[Path]:
    """A single file or every supported file under a folder."""
    p = Path(input_cfg.path)
    exts = {".nd2"} if input_cfg.format == "nd2" else (
            {".tif", ".tiff"} if input_cfg.format == "tif" else
            {".nd2", ".tif", ".tiff"})
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])
    raise FileNotFoundError(f"input.path not found: {p}")


def _model_dir(model_name: str) -> Path:
    here = Path(__file__).resolve().parent
    return here.parent / "nn" / "models" / f"{model_name}_production"


def _load_or_run_nn(recording_path: Path, cfg: DatasetConfig) -> tuple[np.ndarray, Optional[dict]]:
    """Return (roi_masks, meta) — meta may be None if no cached _summary.csv."""
    if cfg.nn.use_cached_inference and cfg.input.inference_dir:
        cache_root = Path(cfg.input.inference_dir)
        rec_dir = _find_cached_rec_dir(cache_root, recording_path.stem)
        if rec_dir is not None:
            roi_masks_path = rec_dir / "roi_masks.npy"
            if roi_masks_path.exists():
                meta = None
                summary_csv = rec_dir / "_summary.csv"
                if summary_csv.exists():
                    import pandas as pd
                    meta = pd.read_csv(summary_csv).to_dict("list")
                return np.load(roi_masks_path), meta
    # No cache hit -> run inference
    return _run_nn_inference(recording_path, cfg), None


def _find_cached_rec_dir(cache_root: Path, stem: str) -> Optional[Path]:
    for d in cache_root.rglob(stem):
        if d.is_dir() and (d / "roi_masks.npy").exists():
            return d
    return None


def _run_nn_inference(recording_path: Path, cfg: DatasetConfig) -> np.ndarray:
    """Run the NN on a single recording. Honors pixel_size override."""
    import sys
    nn_scripts = Path(__file__).resolve().parent.parent / "nn" / "scripts"
    sys.path.insert(0, str(nn_scripts.parent))
    from nn.scripts.analysis import process_single_file, load_run, AnalysisParams
    model_dir = _model_dir(cfg.nn.model)
    loaded = load_run(model_dir)
    out_root = Path(cfg.output.dir) / "_nn_cache"
    out_root.mkdir(parents=True, exist_ok=True)
    res = process_single_file(
        recording_path, model_dir, out_root,
        params=AnalysisParams(),
        loaded=loaded,
        pixel_size_um_override=cfg.input.pixel_size_um,
    )
    roi_masks_path = out_root / recording_path.stem / "roi_masks.npy"
    return np.load(roi_masks_path)


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection (thin wrapper around NEURAL.peakcaller)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_peaks(trace: np.ndarray, cfg: PeakDetectionCfg) -> np.ndarray:
    """Return indices of detected peaks. Uses convex or ALS baseline."""
    from NEURAL.peakcaller import (
        convex_envelope_baseline, als_baseline, detrend_ratio, find_peaks,
    )
    raw = trace.astype(np.float32)
    if cfg.negate:
        raw = -raw
        raw = raw - raw.min() + 1e-6
    baseline = convex_envelope_baseline(raw) if cfg.baseline == "convex" else als_baseline(raw)
    if cfg.baseline == "als":
        D = baseline / (raw + 1e-12)
    else:
        D = detrend_ratio(raw, baseline)
    peaks, _h, _b, _hb, _ha = find_peaks(
        D, required_rise=cfg.threshold, required_fall=cfg.threshold,
        max_lookback=26, max_lookahead=26, detect_negative=False,
    )
    return np.asarray(peaks, dtype=int)


def _extract_traces(video: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mean fluorescence per ROI per frame. video (T,H,W), masks (N,H,W) -> (N,T).

    Streams one frame at a time so memory is O(N*H*W) for the flattened masks
    plus O(H*W) per frame, NOT O(T*H*W).
    """
    T, H, W = video.shape
    N = masks.shape[0]
    if N == 0:
        return np.zeros((0, T), dtype=np.float32)
    masks_flat = masks.reshape(N, H * W).astype(np.float32)
    areas = masks_flat.sum(axis=1)
    areas[areas == 0] = 1.0
    out = np.empty((N, T), dtype=np.float32)
    for t in range(T):
        frame = video[t].reshape(-1).astype(np.float32, copy=False)
        out[:, t] = masks_flat @ frame / areas
    return out


def _select_rois(traces: np.ndarray, peaks_per_roi: list[np.ndarray],
                 vc: VisualizationCfg) -> np.ndarray:
    N = traces.shape[0]
    n_pick = min(vc.n_rois_per_video, N)
    if vc.roi_selection == "all":
        return np.arange(N)
    if vc.roi_selection == "by_index":
        if not vc.roi_indices:
            raise ValueError("roi_selection='by_index' requires roi_indices in config")
        return np.asarray(vc.roi_indices, dtype=int)
    if vc.roi_selection == "random":
        rng = np.random.default_rng(42)
        return rng.choice(N, size=n_pick, replace=False)
    # top_activity: pick ROIs with the most detected events
    n_events = np.array([len(p) for p in peaks_per_roi])
    return np.argsort(n_events)[::-1][:n_pick]


# ─────────────────────────────────────────────────────────────────────────────
# Panel renderers
# ─────────────────────────────────────────────────────────────────────────────
def _render_trace_grid(traces, picks, peaks_per_roi, fps, out_pdf, title):
    n = len(picks)
    cols = 2; rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.6 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()
    t = np.arange(traces.shape[1]) / max(fps, 1e-6)
    for ax, k in zip(axes, picks):
        ax.plot(t, traces[k], "k-", lw=0.6)
        for pk in peaks_per_roi[k]:
            ax.axvline(pk / fps, color="#FF3D00", lw=0.4, alpha=0.6)
        ax.set_title(f"ROI {k+1}  ({len(peaks_per_roi[k])} events)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(picks):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_pdf, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_roi_overlay(mean_img, masks, picks, out_pdf, title, color="#00E5FF"):
    from skimage.measure import find_contours
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    p_lo, p_hi = np.percentile(mean_img, (2, 99))
    img = np.clip((mean_img - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Mean projection")
    axes[1].imshow(img, cmap="gray")
    segs = []
    for k in picks:
        m = masks[k].astype(np.uint8)
        if not m.any(): continue
        for c in find_contours(m, 0.5):
            segs.append(np.column_stack([c[:, 1], c[:, 0]]))
    if segs:
        axes[1].add_collection(LineCollection(segs, colors=["black"], linewidths=2.0, alpha=0.75))
        axes[1].add_collection(LineCollection(segs, colors=[color],  linewidths=0.9))
    axes[1].set_title(f"NN ROIs outlined  (n shown = {len(picks)} of {masks.shape[0]})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_pdf, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_summary_panel(traces, peaks_per_roi, fps, n_rois, out_pdf, title):
    durations = traces.shape[1] / max(fps, 1e-6)
    freq_per_roi = np.array([len(p) / max(durations, 1e-6) for p in peaks_per_roi])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(freq_per_roi, bins=30, color="#3D6FB6")
    axes[0].set_xlabel("Event frequency (Hz)"); axes[0].set_ylabel("# ROIs")
    axes[0].set_title(f"freq per ROI  (n = {n_rois})")
    axes[1].text(0.05, 0.95,
        f"n_rois: {n_rois}\n"
        f"mean freq: {freq_per_roi.mean():.3f} Hz\n"
        f"median freq: {np.median(freq_per_roi):.3f} Hz\n"
        f"duration: {durations:.1f} s\n"
        f"fps: {fps:.2f}",
        va="top", fontfamily="monospace", fontsize=10,
        transform=axes[1].transAxes)
    axes[1].set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_pdf, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def render_dataset(config: Union[str, Path, dict], out_dir: Optional[Union[str, Path]] = None) -> dict:
    """Render one dataset (single recording or a folder of them).

    Args:
      config:  path to a config JSON, or a dict matching the schema.
      out_dir: optional override for config.output.dir.

    Returns dict of {recording_stem: {kind: pdf_path}}.
    """
    if isinstance(config, (str, Path)):
        cfg_path = Path(config).resolve()
        cfg_d = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    else:
        cfg_d = dict(config)
    cfg = DatasetConfig.from_dict(cfg_d)
    if out_dir is not None:
        cfg.output.dir = str(out_dir)

    from NEURAL.io import load_recording

    out = Path(cfg.output.dir); out.mkdir(parents=True, exist_ok=True)
    recordings = _resolve_recordings(cfg.input)
    if cfg.visualization.videos_to_show:
        want = set(cfg.visualization.videos_to_show)
        recordings = [r for r in recordings if r.stem in want or r.name in want]

    print(f"=== NEURAL.figure_engine.render_dataset ===")
    print(f"  recordings: {len(recordings)}")
    print(f"  model:      {cfg.nn.model}")
    print(f"  output:     {out}")

    results: dict[str, dict] = {}
    for rec_path in recordings:
        print(f"\n▶ {rec_path.stem}")
        results[rec_path.stem] = {}

        kinds = ["trace_grid", "roi_overlay", "summary_panel"] if cfg.visualization.kind == "all" else [cfg.visualization.kind]

        # Resume-mode skip: if every kind's PDF already exists, skip
        if all((out / f"{rec_path.stem}_{k}.{cfg.output.format}").exists() for k in kinds):
            print(f"  SKIP (all PDFs already exist)")
            results[rec_path.stem] = {k: str(out / f"{rec_path.stem}_{k}.{cfg.output.format}") for k in kinds}
            continue

        # Per-recording try/except: a corrupt nd2 or NN crash for ONE recording must
        # not take down the whole sub-experiment.
        try:
            # NN ROIs
            masks, _ = _load_or_run_nn(rec_path, cfg)
            if masks.shape[0] == 0:
                print("  no ROIs"); continue
            # Recording (only needed for traces or roi_overlay)
            need_video = cfg.visualization.kind in ("trace_grid", "summary_panel", "all")
            need_mean  = cfg.visualization.kind in ("roi_overlay", "all")
            rec = load_recording(
                rec_path, fps=cfg.input.fps, pixel_size_um=cfg.input.pixel_size_um,
                n_frames=cfg.input.n_frames, duration_s=cfg.input.duration_s,
                format=cfg.input.format,
            ) if (need_video or need_mean) else None
            # Traces + peaks
            if rec is not None:
                traces = _extract_traces(rec.video, masks)
                peaks_per_roi = [_detect_peaks(traces[i], cfg.peak_detection)
                                  for i in range(masks.shape[0])]
            else:
                traces = None; peaks_per_roi = None

            picks = _select_rois(traces, peaks_per_roi, cfg.visualization) if peaks_per_roi else np.arange(masks.shape[0])
            title = rec_path.stem

            for k in kinds:
                pdf = out / f"{rec_path.stem}_{k}.{cfg.output.format}"
                try:
                    if k == "trace_grid":
                        _render_trace_grid(traces, picks, peaks_per_roi, rec.fps, pdf, title)
                    elif k == "roi_overlay":
                        mean_img = rec.video.mean(axis=0)
                        _render_roi_overlay(mean_img, masks, picks, pdf, title)
                    elif k == "summary_panel":
                        _render_summary_panel(traces, peaks_per_roi, rec.fps, masks.shape[0], pdf, title)
                    results[rec_path.stem][k] = str(pdf)
                    print(f"  wrote {pdf.name}")
                except Exception as e:
                    print(f"  ERROR rendering {k}: {e}")
        except Exception as e:
            print(f"  SKIP {rec_path.stem}: {e.__class__.__name__}: {e}")
            results[rec_path.stem] = None

    print(f"\nDone. {len(results)} recording(s) rendered.")
    return results
