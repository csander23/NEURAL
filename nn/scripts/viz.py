"""Coloured ROI overlay visualisation (shared by all segmentation approaches).

Channels in ``*_summary_rgb.npy`` (after ``load_and_resize``) are z-scored per
channel: index 0 = mean summary, 1 = std dev, 2 = correlation (see roi_labeler).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import torch


def _channel_to_display_bg(
    ch: np.ndarray,
    *,
    brightness_gamma: float = 0.55,
    brightness_gain: float = 4.0,   # kept for API compat but no longer used
    plo: float = 1.0,
    phi: float = 99.0,
) -> np.ndarray:
    """Map one z-scored channel to a [0, 1] grayscale for imshow.

    Uses percentile clipping so each channel's actual data range drives
    the display window (handles correlation, which has a tighter distribution
    than mean/std without blowing out).  A mild gamma < 1 lifts mid-tones.

    Parameters
    ----------
    ch              : (H, W) array, any dtype
    brightness_gamma: power applied after normalisation; < 1 brightens mid-tones
    plo / phi       : lower / upper percentile for the display window (default 1–99)
    """
    ch = np.asarray(ch, dtype=np.float64)
    lo = float(np.percentile(ch, plo))
    hi = float(np.percentile(ch, phi))
    if hi - lo < 1e-12:
        return np.zeros(ch.shape, dtype=np.float32)
    x = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    x = x ** float(brightness_gamma)
    return x.astype(np.float32)


def _show_grayscale(ax: plt.Axes, bg: np.ndarray, title: str) -> None:
    ax.imshow(bg, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def _draw_instances(ax: plt.Axes, bg: np.ndarray, masks: np.ndarray, label: str) -> None:
    """Overlay coloured filled regions + contours for each mask on ax.

    Parameters
    ----------
    bg    : (H, W) float image in [0, 1], shown as gray background
    masks : (N, H, W) binary uint8/bool
    label : axis title prefix
    """
    ax.imshow(bg, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    n = len(masks)
    if n == 0:
        ax.set_title(f"{label} (n=0)")
        ax.axis("off")
        return

    cmap = plt.get_cmap("hsv")
    hues = [(i / n + 0.618 * i) % 1.0 for i in range(n)]

    fill = np.zeros((*bg.shape, 4), dtype=np.float32)
    for mask, hue in zip(masks, hues):
        binary = mask.astype(bool)
        rgba = cmap(hue)
        fill[binary] = (rgba[0], rgba[1], rgba[2], 0.28)
    ax.imshow(fill, interpolation="nearest")

    for mask, hue in zip(masks, hues):
        binary = mask.astype(np.float32)
        if binary.max() == 0:
            continue
        color = cmap(hue)
        try:
            ax.contour(binary, levels=[0.5], colors=[color], linewidths=[0.9])
        except Exception:
            pass

    ax.set_title(f"{label} (n={n})")
    ax.axis("off")


def _to_numpy_chw(img_tensor: "torch.Tensor | np.ndarray") -> np.ndarray:
    if hasattr(img_tensor, "detach"):
        return img_tensor.detach().cpu().numpy()
    return np.asarray(img_tensor)


def _downsample_for_overlay(
    img: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stride-downsample image and masks so the longer side ≤ max_dim.

    For overlay PNGs only — keeps matplotlib's per-call (H, W, 4) float32 copies
    bounded. Masks are stride-decimated (no thresholding); a single-pixel ROI
    will survive if it lands on a kept pixel.
    """
    _, H, W = img.shape
    longest = max(H, W)
    if longest <= max_dim:
        return img, gt_masks, pred_masks
    stride = int(np.ceil(longest / max_dim))
    img_ds = img[:, ::stride, ::stride].copy()
    gt_ds  = gt_masks[:, ::stride, ::stride].copy() if len(gt_masks)  else gt_masks
    pr_ds  = pred_masks[:, ::stride, ::stride].copy() if len(pred_masks) else pred_masks
    return img_ds, gt_ds, pr_ds


def save_comparison_panel(
    img_tensor: "torch.Tensor | np.ndarray",
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    out_path: Path,
    title: str = "",
    *,
    brightness_gamma: float = 0.42,
    brightness_gain: float = 4.0,
    dpi: int = 100,
    max_dim: int = 768,
) -> None:
    """Nine-panel figure: Mean, Std dev, and Correlation rows (z-scored ch 0–2).

    Each row: channel only | + ground truth | + predictions. Matches the
    three summary channels in ``*_summary_rgb.npy`` after ``load_and_resize``.
    Images larger than ``max_dim`` on the longest side are stride-downsampled
    before plotting to keep matplotlib's internal copies bounded.
    """
    img = _to_numpy_chw(img_tensor)
    if img.shape[0] < 3:
        raise ValueError("save_comparison_panel expects 3 channels (mean, std, correlation)")

    img, gt_masks, pred_masks = _downsample_for_overlay(img, gt_masks, pred_masks, max_dim)

    kw = {"brightness_gamma": brightness_gamma, "brightness_gain": brightness_gain}
    bg_mean = _channel_to_display_bg(img[0], **kw)
    bg_std  = _channel_to_display_bg(img[1], **kw)
    bg_corr = _channel_to_display_bg(img[2], **kw)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    _show_grayscale(axes[0, 0], bg_mean, "Mean (ch 0, z-scored)")
    _draw_instances(axes[0, 1], bg_mean, gt_masks,    "Mean + ground truth")
    _draw_instances(axes[0, 2], bg_mean, pred_masks, "Mean + predictions")

    _show_grayscale(axes[1, 0], bg_std, "Std dev (ch 1, z-scored)")
    _draw_instances(axes[1, 1], bg_std, gt_masks,    "Std dev + ground truth")
    _draw_instances(axes[1, 2], bg_std, pred_masks, "Std dev + predictions")

    _show_grayscale(axes[2, 0], bg_corr, "Correlation (ch 2, z-scored)")
    _draw_instances(axes[2, 1], bg_corr, gt_masks,    "Correlation + ground truth")
    _draw_instances(axes[2, 2], bg_corr, pred_masks, "Correlation + predictions")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    import gc
    gc.collect()


def save_training_roi_panel(
    img_tensor: "torch.Tensor | np.ndarray",
    gt_masks: np.ndarray,
    out_path: Path,
    title: str = "",
    *,
    brightness_gamma: float = 0.42,
    brightness_gain: float = 4.0,
    max_dim: int = 768,
) -> None:
    """Six-panel QC figure: Mean, Std, Correlation each with channel + GT overlays."""
    img = _to_numpy_chw(img_tensor)
    if img.shape[0] < 3:
        raise ValueError("save_training_roi_panel expects 3 channels (mean, std, correlation)")

    img, gt_masks, _ = _downsample_for_overlay(img, gt_masks, np.zeros((0, 0, 0), dtype=bool), max_dim)

    kw = {"brightness_gamma": brightness_gamma, "brightness_gain": brightness_gain}
    bg_mean = _channel_to_display_bg(img[0], **kw)
    bg_std  = _channel_to_display_bg(img[1], **kw)
    bg_corr = _channel_to_display_bg(img[2], **kw)

    fig, axes = plt.subplots(3, 2, figsize=(14, 21))
    _show_grayscale(axes[0, 0], bg_mean, "Mean (ch 0, z-scored)")
    _draw_instances(axes[0, 1], bg_mean, gt_masks, "Mean + ground truth")
    _show_grayscale(axes[1, 0], bg_std, "Std dev (ch 1, z-scored)")
    _draw_instances(axes[1, 1], bg_std, gt_masks, "Std dev + ground truth")
    _show_grayscale(axes[2, 0], bg_corr, "Correlation (ch 2, z-scored)")
    _draw_instances(axes[2, 1], bg_corr, gt_masks, "Correlation + ground truth")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
