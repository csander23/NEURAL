"""NEURAL.io - load a recording (.nd2 or .tif) with optional manual overrides
for the metadata that downstream code needs:
  - fps                (Hz)
  - pixel_size_um      (µm/px)
  - n_frames           (T)
  - duration_s

Auto-detects from .nd2 metadata when override is None. .tif files have no such
metadata, so manual overrides are required there (or the caller must supply
them in the config).

Usage:
    from NEURAL.io import load_recording
    rec = load_recording("path/to/video.nd2",
                         fps=None, pixel_size_um=None)  # auto-detect
    rec.video        # (T, H, W) numpy array
    rec.fps          # 25.5
    rec.pixel_size_um
    rec.n_frames
    rec.duration_s
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Recording:
    path: Path
    video: np.ndarray              # (T, H, W) uint16 / float
    fps: float
    pixel_size_um: float
    n_frames: int
    duration_s: float
    source: str                    # "nd2" | "tif"
    overrides_applied: dict = field(default_factory=dict)


def _load_nd2(path: Path,
              fps: Optional[float],
              pixel_size_um: Optional[float],
              n_frames: Optional[int],
              duration_s: Optional[float]) -> Recording:
    import nd2
    overrides = {}
    with nd2.ND2File(str(path)) as f:
        sizes = f.sizes
        T = sizes.get("T", 1)
        arr = f.asarray()
        if arr.ndim == 2:
            arr = arr[None, ...]
        # auto-detect pixel_size
        vox = f.voxel_size()
        detected_px = float(vox.x)
        # auto-detect fps: extract period from metadata (best-effort)
        detected_fps = None
        try:
            md = f.metadata
            chan = md.channels[0] if md.channels else None
            if chan is not None:
                # exp.parameters.periods[0].periodDiff.avg = period in ms
                exp_list = f.experiment
                if exp_list:
                    periods = getattr(exp_list[0].parameters, "periods", None)
                    if periods:
                        period_ms = periods[0].periodDiff.avg
                        if period_ms and period_ms > 0:
                            detected_fps = 1000.0 / period_ms
        except Exception:
            detected_fps = None

    if pixel_size_um is None:
        pixel_size_um = detected_px
    else:
        overrides["pixel_size_um"] = (detected_px, pixel_size_um)

    if fps is None:
        if detected_fps is None:
            raise ValueError(f"{path}: fps not in nd2 metadata; "
                              "supply 'fps' explicitly in the config")
        fps = detected_fps
    else:
        overrides["fps"] = (detected_fps, fps)

    if n_frames is None:
        n_frames = int(arr.shape[0])
    else:
        overrides["n_frames"] = (arr.shape[0], n_frames)
        # truncate if user wants fewer frames than present
        if n_frames < arr.shape[0]:
            arr = arr[:n_frames]

    if duration_s is None:
        duration_s = n_frames / fps if fps > 0 else 0.0
    else:
        overrides["duration_s"] = (n_frames / fps if fps > 0 else 0.0, duration_s)

    return Recording(
        path=path, video=arr, fps=float(fps), pixel_size_um=float(pixel_size_um),
        n_frames=int(n_frames), duration_s=float(duration_s), source="nd2",
        overrides_applied=overrides,
    )


def _load_tif(path: Path,
              fps: Optional[float],
              pixel_size_um: Optional[float],
              n_frames: Optional[int],
              duration_s: Optional[float]) -> Recording:
    import tifffile
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    # tif has no canonical fps/pixel_size in standard tags -> require user input
    if fps is None:
        raise ValueError(f"{path}: .tif input requires explicit fps in config")
    if pixel_size_um is None:
        raise ValueError(f"{path}: .tif input requires explicit pixel_size_um in config")
    if n_frames is None:
        n_frames = int(arr.shape[0])
    elif n_frames < arr.shape[0]:
        arr = arr[:n_frames]
    if duration_s is None:
        duration_s = n_frames / fps
    return Recording(
        path=path, video=arr, fps=float(fps), pixel_size_um=float(pixel_size_um),
        n_frames=int(n_frames), duration_s=float(duration_s), source="tif",
    )


def load_recording(path,
                   fps: Optional[float] = None,
                   pixel_size_um: Optional[float] = None,
                   n_frames: Optional[int] = None,
                   duration_s: Optional[float] = None,
                   format: str = "auto") -> Recording:
    """Load .nd2 or .tif into a Recording. Any None field is auto-detected
    from nd2 metadata (nd2 only); a non-None value overrides the detected one.

    For .tif input, fps and pixel_size_um must be explicit (no embedded metadata).

    Args:
        path: file path
        fps: frames per second; None to auto-detect from nd2
        pixel_size_um: µm per pixel; None to auto-detect from nd2
        n_frames: optionally truncate to first N frames; None = use all
        duration_s: optionally set duration; computed from fps×n_frames if None
        format: "auto" | "nd2" | "tif"
    """
    path = Path(path)
    if format == "auto":
        ext = path.suffix.lower()
        format = "nd2" if ext == ".nd2" else ("tif" if ext in (".tif", ".tiff") else None)
    if format == "nd2":
        return _load_nd2(path, fps, pixel_size_um, n_frames, duration_s)
    if format == "tif":
        return _load_tif(path, fps, pixel_size_um, n_frames, duration_s)
    raise ValueError(f"unsupported format: {format} (path={path})")
