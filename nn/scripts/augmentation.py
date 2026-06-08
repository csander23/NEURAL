"""Spatial + intensity augmentation for already-cropped patches.

This module is the same as 04.10.2026 except that **scale augmentation is
handled in `data.py` at crop time** (not here) because it needs access to the
full native image to vary the source region size. Functions here operate on a
patch that has already been cropped (and optionally rescaled).
"""
from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ─────────────────────────────────────────────────────────────────────────────
# Spatial transforms
# ─────────────────────────────────────────────────────────────────────────────

def elastic_deform(
    img: np.ndarray,
    masks: np.ndarray,
    alpha: float = 25.0,
    sigma: float = 8.0,
    np_rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random elastic deformation applied consistently to image and masks."""
    if np_rng is None:
        np_rng = np.random.RandomState()

    H, W = img.shape[1], img.shape[2]
    dy = gaussian_filter(np_rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32), sigma) * alpha
    dx = gaussian_filter(np_rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32), sigma) * alpha

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    coords_y = np.clip(ys + dy, 0.0, H - 1)
    coords_x = np.clip(xs + dx, 0.0, W - 1)

    img_out = np.empty_like(img)
    for c in range(img.shape[0]):
        img_out[c] = map_coordinates(img[c], [coords_y, coords_x], order=1, mode="reflect")

    if masks.shape[0] > 0:
        masks_out = np.empty_like(masks)
        for n in range(masks.shape[0]):
            deformed = map_coordinates(
                masks[n].astype(np.float32), [coords_y, coords_x], order=0, mode="constant"
            )
            masks_out[n] = (deformed > 0.5).astype(np.uint8)
    else:
        masks_out = masks

    return img_out, masks_out


# ─────────────────────────────────────────────────────────────────────────────
# Intensity transforms (image only)
# ─────────────────────────────────────────────────────────────────────────────

def random_blur(
    img: np.ndarray,
    sigma_max: float = 1.5,
    np_rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if np_rng is None:
        np_rng = np.random.RandomState()
    sigma = float(np_rng.uniform(0.3, sigma_max))
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        out[c] = gaussian_filter(img[c], sigma)
    return out


def random_jitter(
    img: np.ndarray,
    strength: float = 0.25,
    np_rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if np_rng is None:
        np_rng = np.random.RandomState()
    bright   = float(np_rng.uniform(-strength, strength))
    contrast = float(np_rng.uniform(1.0 - strength, 1.0 + strength))
    return (img * contrast + bright).astype(img.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline (post-crop)
# ─────────────────────────────────────────────────────────────────────────────

def strong_augment(
    img: np.ndarray,
    masks: np.ndarray,
    rng: random.Random,
    np_rng: np.random.RandomState,
    elastic_p: float = 0.5,
    blur_p: float = 0.7,
    jitter_p: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply flips/rotations → elastic → blur → jitter to a pre-cropped patch."""
    if rng.random() > 0.5:
        img   = img[:, :, ::-1].copy()
        masks = masks[:, :, ::-1].copy()
    if rng.random() > 0.5:
        img   = img[:, ::-1, :].copy()
        masks = masks[:, ::-1, :].copy()
    k = rng.randint(0, 3)
    if k:
        img   = np.rot90(img,   k, axes=(1, 2)).copy()
        masks = np.rot90(masks, k, axes=(1, 2)).copy()

    if rng.random() < elastic_p:
        img, masks = elastic_deform(img, masks, alpha=25.0, sigma=8.0, np_rng=np_rng)

    if rng.random() < blur_p:
        img = random_blur(img, sigma_max=1.5, np_rng=np_rng)

    if rng.random() < jitter_p:
        img = random_jitter(img, strength=0.25, np_rng=np_rng)

    return img, masks
