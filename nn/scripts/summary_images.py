"""Build the 3-channel summary image (mean, std, temporal correlation) from a
microscopy video file.

Inputs supported: .nd2 (multi-frame), .tif/.tiff (T,H,W or T,Z,H,W or T,C,H,W).
Output: `<stem>_summary_rgb.npy` with shape (H, W, 3) float32 — same format
the training pipeline + ROI labeler expect.

The summary computation runs as a **single-pass streaming algorithm** with
float64 per-pixel accumulators (mean, second moment, lag-1 cross/auto sums).
Peak RAM per file is ~7 * H * W * 8 bytes ≈ 80 MB at 1192² regardless of how
many frames the recording has — replaces a previous implementation that
materialised the full (T, H, W) float32 video plus two centred copies
(60+ GB at T=3000), causing MemoryError on long recordings.

Algebraic equivalence
---------------------
The lag-1 correlation identity used:
    Σ(x_{t-1}-μ)(x_t-μ) ≡ Σ x_{t-1} x_t  − μ·(Σ x_{t-1} + Σ x_t)  + (T-1)·μ²
holds exactly in real arithmetic. The streaming form differs from the older
two-array implementation only at the float-precision level (~1e-7 relative
per pixel with float64 accumulators); the corr channel is z-scored before
hitting the NN, so the drift is invisible downstream.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Streaming summary core
# ─────────────────────────────────────────────────────────────────────────────

def _summary_from_frame_iter(
    frames: Iterator[np.ndarray],
    H: int,
    W: int,
) -> np.ndarray:
    """Single-pass streaming summary. Yields one frame at a time.

    All accumulators are (H, W) float64 for numerical stability (uncentred
    sums of large pixel values × thousands of frames otherwise lose precision
    on subtraction).
    """
    S0 = np.zeros((H, W), dtype=np.float64)
    SQ = np.zeros((H, W), dtype=np.float64)
    # Lag-1 cross + per-side sums and squared sums
    C  = np.zeros((H, W), dtype=np.float64)
    L  = np.zeros((H, W), dtype=np.float64)  # Σ x_t for t in 0..T-2
    R  = np.zeros((H, W), dtype=np.float64)  # Σ x_t for t in 1..T-1
    LL = np.zeros((H, W), dtype=np.float64)
    RR = np.zeros((H, W), dtype=np.float64)

    prev: np.ndarray | None = None
    T = 0
    for cur in frames:
        cur64 = np.asarray(cur, dtype=np.float64)
        T += 1
        S0 += cur64
        SQ += cur64 * cur64
        if prev is not None:
            C  += prev * cur64
            L  += prev
            R  += cur64
            LL += prev * prev
            RR += cur64 * cur64
        prev = cur64

    if T == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    mean = S0 / T
    # std: sqrt(E[X²] - E[X]²), clamped to 0 for numerical safety
    var = SQ / T - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    if T < 2:
        corr = np.zeros((H, W), dtype=np.float64)
    else:
        n_pairs = float(T - 1)
        mu_sq = mean * mean
        num   = C  - mean * L - mean * R + n_pairs * mu_sq
        den_a = LL - 2.0 * mean * L + n_pairs * mu_sq
        den_b = RR - 2.0 * mean * R + n_pairs * mu_sq
        den_a = np.maximum(den_a, 0.0)
        den_b = np.maximum(den_b, 0.0)
        corr  = num / (np.sqrt(den_a * den_b) + 1e-8)

    return np.stack([mean, std, corr], axis=-1).astype(np.float32)


def summary_from_video(video_thw: np.ndarray) -> np.ndarray:
    """Streaming summary computed from an in-memory (T, H, W) array.

    Routes the array through the same streaming kernel as the on-disk path so
    both code paths produce bit-identical output for the same input.
    """
    if video_thw.ndim != 3:
        raise ValueError(f"expected (T, H, W); got shape {video_thw.shape}")
    T, H, W = video_thw.shape
    def _iter() -> Iterator[np.ndarray]:
        for t in range(T):
            yield video_thw[t]
    return _summary_from_frame_iter(_iter(), H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming frame readers (per file type)
# ─────────────────────────────────────────────────────────────────────────────

def _squeeze_frame_to_2d(frame: np.ndarray) -> np.ndarray:
    """Reduce a single-frame slice to (H, W). Mirrors the legacy _load_video
    squeeze logic: singleton dims dropped, otherwise channel/Z index 0 kept."""
    while frame.ndim > 2:
        if 1 in frame.shape:
            frame = np.squeeze(frame)
        else:
            frame = frame[0]
    return frame


def _iter_nd2_frames(path: Path) -> tuple[Iterator[np.ndarray], int, int, float | None]:
    """Open an ND2 lazily and return (frame_iterator, H, W, pixel_size_um).

    The iterator opens its own ND2File handle (so the caller doesn't need to
    keep a context manager open) and yields one (H, W) frame at a time.
    """
    import nd2
    # Peek to get shape + pixel size, then close. The iterator reopens.
    with nd2.ND2File(str(path)) as f:
        shape = tuple(f.shape)
        try:
            px = float(f.voxel_size().x)
        except Exception:
            px = None
        # Determine the per-frame shape after the same squeeze/select logic
        # the old _load_video applied to the full array.
        if len(shape) == 2:
            n_frames = 1
            H, W = shape
        else:
            n_frames = shape[0]
            sample = _squeeze_frame_to_2d(np.asarray(f.read_frame(0)))
            H, W = sample.shape

    def _iter() -> Iterator[np.ndarray]:
        with nd2.ND2File(str(path)) as f2:
            if len(shape) == 2:
                yield _squeeze_frame_to_2d(np.asarray(f2.asarray()))
                return
            for t in range(n_frames):
                yield _squeeze_frame_to_2d(np.asarray(f2.read_frame(t)))

    return _iter(), H, W, px


def _iter_tiff_frames(path: Path) -> tuple[Iterator[np.ndarray], int, int, float | None]:
    """Lazy TIFF frame iterator using tifffile's page interface."""
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        n_pages = len(tf.pages)
        sample = np.asarray(tf.pages[0].asarray())
    sample = _squeeze_frame_to_2d(sample)
    H, W = sample.shape

    def _iter() -> Iterator[np.ndarray]:
        with tifffile.TiffFile(str(path)) as tf2:
            for p in tf2.pages:
                yield _squeeze_frame_to_2d(np.asarray(p.asarray()))

    return _iter(), H, W, None


def summary_from_path(path: Path) -> tuple[np.ndarray, float | None]:
    """Build the (H, W, 3) summary directly from a file, streaming frames.

    Never materialises the full video in memory. Returns (summary, pixel_size_um).
    """
    suf = path.suffix.lower()
    if suf == ".nd2":
        frames, H, W, px = _iter_nd2_frames(path)
    elif suf in (".tif", ".tiff"):
        frames, H, W, px = _iter_tiff_frames(path)
    else:
        raise ValueError(f"unsupported input: {suf}")
    summary = _summary_from_frame_iter(frames, H, W)
    return summary, px


# ─────────────────────────────────────────────────────────────────────────────
# Legacy in-memory loader (kept for callers that already have a numpy video)
# ─────────────────────────────────────────────────────────────────────────────

def _load_video(path: Path) -> tuple[np.ndarray, float | None]:
    """Legacy: load the entire (T, H, W) video into RAM. AVOID on long files.

    Prefer `summary_from_path(path)` which streams frames and never holds the
    full video in memory.
    """
    suf = path.suffix.lower()
    if suf == ".nd2":
        import nd2
        with nd2.ND2File(str(path)) as f:
            arr = f.asarray()
            px = float(f.voxel_size().x)
        while arr.ndim > 3:
            if 1 in arr.shape:
                arr = arr.squeeze()
            else:
                arr = arr[:, 0]
        if arr.ndim == 2:
            arr = arr[None]
        return arr, px

    if suf in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 4:
            arr = arr.squeeze() if 1 in arr.shape else arr[:, 0]
        if arr.ndim == 2:
            arr = arr[None]
        return arr, None

    raise ValueError(f"unsupported input: {suf}")


# ─────────────────────────────────────────────────────────────────────────────
# Build entry points
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(path: Path, output_dir: Path | None = None) -> Path:
    """Build and save the summary image — streaming, low RAM. Returns out path."""
    summary, _px = summary_from_path(path)
    out_dir = output_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_summary_rgb.npy"
    np.save(out_path, summary)
    return out_path


def build_summaries(
    raw_dir: Path,
    training_dir: Path,
    extensions: tuple[str, ...] = (".nd2", ".tif", ".tiff"),
    skip_existing: bool = True,
) -> list[Path]:
    """Convert every video in `raw_dir` to a summary in `training_dir`."""
    raw_dir = Path(raw_dir)
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for f in sorted(raw_dir.iterdir()):
        if f.suffix.lower() not in extensions:
            continue
        target = training_dir / f"{f.stem}_summary_rgb.npy"
        if skip_existing and target.exists():
            print(f"  skip (already built): {target.name}")
            out_paths.append(target)
            continue
        print(f"  building summary: {f.name} -> {target.name}")
        out_paths.append(build_summary(f, training_dir))
    return out_paths


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path,
                   help="Either a single file or a directory (then process all .nd2/.tif).")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    if args.input.is_dir():
        build_summaries(args.input, args.out_dir or args.input)
    else:
        out = build_summary(args.input, args.out_dir)
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
