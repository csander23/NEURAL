"""Native-resolution data loading: patch sampling for training, sliding-window
tiling for validation.

Key differences vs 04.10.2026/scripts/data.py
---------------------------------------------
* No global resize. Inputs are loaded at their native (H, W).
* Training samples are random crops of size `patch_size`; instance masks are
  cropped at the patch boundary and small slivers are dropped.
* Optional scale augmentation chooses a source region size in
  [patch_size / scale_max, patch_size / scale_min] and resizes to patch_size.
  This is opt-in via `Config.scale_aug` — the model does not depend on it.
* Validation provides full native-resolution images plus a helper to iterate
  (top, left) tile positions for sliding-window inference.
"""
from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Recording descriptor + discovery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecordingExample:
    stem: str
    summary_path: str
    mask_path: str
    roi_count: int
    height: int
    width: int


def discover_examples(training_dir: str) -> list[RecordingExample]:
    all_rgb = sorted(glob.glob(str(Path(training_dir) / "*_summary_rgb.npy")))
    examples: list[RecordingExample] = []
    for summary_path in all_rgb:
        mask_path = summary_path.replace("_summary_rgb.npy", "_instance_masks.npy")
        if not Path(mask_path).exists():
            continue
        masks = np.load(mask_path, mmap_mode="r")
        if masks.ndim != 3 or masks.shape[0] == 0:
            continue
        examples.append(RecordingExample(
            stem=Path(summary_path).stem.replace("_summary_rgb", ""),
            summary_path=summary_path,
            mask_path=mask_path,
            roi_count=int(masks.shape[0]),
            height=int(masks.shape[1]),
            width=int(masks.shape[2]),
        ))
    return examples


def split_examples(
    examples: list[RecordingExample],
    val_fraction: float,
    seed: int,
) -> tuple[list[RecordingExample], list[RecordingExample]]:
    """Stratify by ROI count — evenly spaced positions land in val."""
    if not examples or val_fraction <= 0 or len(examples) == 1:
        return list(examples), []
    ordered = sorted(examples, key=lambda e: e.roi_count)
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    positions = set(int(x) for x in np.linspace(0, len(ordered) - 1, num=val_count, dtype=int).tolist())
    # consume `seed` to keep the API stable across the codebase
    random.Random(seed)
    train = [e for i, e in enumerate(ordered) if i not in positions]
    val   = [e for i, e in enumerate(ordered) if i in positions]
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Native-resolution loading + per-channel normalisation
# ─────────────────────────────────────────────────────────────────────────────

def load_native(
    summary_path: str,
    mask_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one recording at native resolution.

    Returns
    -------
    img   : float32 (3, H, W)  per-channel z-scored
    masks : uint8   (N, H, W)  binary
    """
    img_raw = np.asarray(np.load(summary_path, mmap_mode="r"), dtype=np.float32)  # (H, W, 3)
    masks_raw = np.asarray(np.load(mask_path,    mmap_mode="r"), dtype=np.uint8)  # (N, H, W)

    img = img_raw.copy()
    for c in range(3):
        ch = img[:, :, c]
        mu, sigma = ch.mean(), ch.std()
        img[:, :, c] = (ch - mu) / (sigma + 1e-8)
    img = img.transpose(2, 0, 1)  # (3, H, W)
    return img, masks_raw


# ─────────────────────────────────────────────────────────────────────────────
# Patch sampling (training)
# ─────────────────────────────────────────────────────────────────────────────

def _crop_and_filter_masks(
    masks: np.ndarray,
    top: int,
    left: int,
    src_h: int,
    src_w: int,
    min_roi_px: int,
) -> np.ndarray:
    """Crop (N,H,W) masks to source window and drop ROIs with too few in-window pixels."""
    if masks.shape[0] == 0:
        return masks[:, :src_h, :src_w].copy()
    crop = masks[:, top:top + src_h, left:left + src_w]
    sums = crop.reshape(crop.shape[0], -1).sum(axis=1)
    keep = sums >= min_roi_px
    return crop[keep].copy() if keep.any() else np.zeros((0, src_h, src_w), dtype=np.uint8)


def sample_patch(
    img: np.ndarray,
    masks: np.ndarray,
    patch_size: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    *,
    scale_aug: bool = False,
    scale_range: tuple[float, float] = (0.7, 1.5),
    min_roi_after_crop_px: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Random crop a (patch_size, patch_size) view from a native-res image.

    When `scale_aug` is True, the source region size is randomly picked in
    `[patch_size / scale_max, patch_size / scale_min]` and then resized to
    `patch_size × patch_size`. This is purely a data-time augmentation; the
    model and post-processing do not depend on it.
    """
    _, H, W = img.shape
    if scale_aug:
        s_lo, s_hi = scale_range
        s = float(np_rng.uniform(s_lo, s_hi))
        src_size = max(8, int(round(patch_size / s)))
    else:
        src_size = patch_size
    src_size = min(src_size, H, W)

    top  = rng.randint(0, H - src_size) if H > src_size else 0
    left = rng.randint(0, W - src_size) if W > src_size else 0

    img_crop = img[:, top:top + src_size, left:left + src_size]
    masks_crop = _crop_and_filter_masks(masks, top, left, src_size, src_size, min_roi_after_crop_px)

    if src_size != patch_size:
        img_crop = sk_resize(
            img_crop.transpose(1, 2, 0),
            (patch_size, patch_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32).transpose(2, 0, 1)
        if masks_crop.shape[0] > 0:
            resized = sk_resize(
                masks_crop.transpose(1, 2, 0).astype(np.float32),
                (patch_size, patch_size),
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            ).transpose(2, 0, 1)
            masks_crop = (resized > 0.5).astype(np.uint8)
            sums = masks_crop.reshape(masks_crop.shape[0], -1).sum(axis=1)
            masks_crop = masks_crop[sums >= min_roi_after_crop_px]
        else:
            masks_crop = np.zeros((0, patch_size, patch_size), dtype=np.uint8)

    return np.ascontiguousarray(img_crop), np.ascontiguousarray(masks_crop)


class PatchDataset(Dataset):
    """Training dataset: each item is a freshly sampled patch.

    Length = `n_examples * patches_per_image`. Indexing maps to an example and
    a random patch is drawn each call. Native images are cached in RAM (n_train
    is small in this project).
    """

    def __init__(
        self,
        examples: list[RecordingExample],
        patch_size: int,
        patches_per_image: int,
        augment: bool,
        scale_aug: bool,
        scale_range: tuple[float, float],
        seed: int,
    ) -> None:
        self.examples = examples
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.scale_aug = scale_aug
        self.scale_range = scale_range
        self.seed = seed
        self._native: list[tuple[np.ndarray, np.ndarray]] = [
            load_native(ex.summary_path, ex.mask_path) for ex in examples
        ]
        # Per-worker RNG initialisation happens at first __getitem__ (see _ensure_rng).
        self._py_rng: random.Random | None = None
        self._np_rng: np.random.RandomState | None = None

    def _ensure_rng(self) -> tuple[random.Random, np.random.RandomState]:
        if self._py_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            wid = worker_info.id if worker_info is not None else 0
            self._py_rng = random.Random(self.seed + 1000 * wid + 1)
            self._np_rng = np.random.RandomState(self.seed + 1000 * wid + 7)
        return self._py_rng, self._np_rng  # type: ignore[return-value]

    def __len__(self) -> int:
        return max(1, len(self.examples) * self.patches_per_image)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        if not self.examples:
            raise IndexError("empty dataset")
        ex_idx = idx % len(self.examples)
        img, masks = self._native[ex_idx]
        rng, np_rng = self._ensure_rng()
        img_p, masks_p = sample_patch(
            img, masks,
            patch_size=self.patch_size,
            rng=rng, np_rng=np_rng,
            scale_aug=self.scale_aug,
            scale_range=self.scale_range,
        )
        if self.augment:
            from .augmentation import strong_augment
            img_p, masks_p = strong_augment(img_p, masks_p, rng, np_rng)
        return torch.from_numpy(img_p.copy()), masks_p


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window tiling (validation)
# ─────────────────────────────────────────────────────────────────────────────

def tile_positions(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return list of (top, left) tile origins covering the image.

    The last row / column is right/bottom-aligned so that every pixel is in at
    least one tile, even when `(H - tile_size) % stride != 0`.
    """
    stride = max(1, tile_size - overlap)

    def _axis(n: int) -> list[int]:
        if n <= tile_size:
            return [0]
        coords = list(range(0, n - tile_size + 1, stride))
        if coords[-1] != n - tile_size:
            coords.append(n - tile_size)
        return coords

    tops  = _axis(height)
    lefts = _axis(width)
    return [(t, l) for t in tops for l in lefts]
