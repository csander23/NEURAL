"""FlowField v2 — native-resolution patch training with calibrated losses.

Five changes vs 05.12.2026 (each motivated by the v1 threshold sweep — see
`config.py` docstring for the full rationale):

1. **Auto pos_weight_fg** computed from training-data fg pixel ratio
   (capped at `auto_pos_weight_cap`). Replaces hardcoded 8.0 so the BCE
   optimum sits near 0.5 instead of 0.7.
2. **BCE + Tversky combined fg loss**. Direct IoU optimization to lift the
   matched-IoU ceiling that capped v1 at ~0.58–0.61.
3. **Loss weighting**: `loss = loss_fg_weight * loss_fg + loss_flow_weight * loss_flow`.
   Defaults rebalance flow vs fg gradient magnitudes (~3× under in v1).
4. **Early stopping** on val F1 with configurable patience.
5. **Watershed compactness** is now a config knob (default 0.0); v1 had a
   hardcoded 0.01 which biased toward round blobs.

The training-target generation (`make_targets`), sliding-window inference
(`predict_full_image`), and bbox-restricted matching (`match_instances`) are
unchanged from 05.12 — those were already correct, the v1 issues were in
loss formulation and post-processing defaults.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import center_of_mass, gaussian_filter, label as ndi_label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import (
    PatchDataset, RecordingExample, discover_examples, load_native,
    split_examples, tile_positions,
)
from .resnet_unet import ResNetUNet
from .viz import save_comparison_panel


# ─────────────────────────────────────────────────────────────────────────────
# Target generation (operates on a patch)
# ─────────────────────────────────────────────────────────────────────────────

def make_targets(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (fg, dy, dx) maps from (N, H, W) binary masks."""
    H, W = masks.shape[1], masks.shape[2]
    if masks.shape[0] == 0:
        z = np.zeros((H, W), dtype=np.float32)
        return z, z, z

    fg = masks.any(axis=0).astype(np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
    dx = np.zeros((H, W), dtype=np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    for m in masks:
        if not m.any():
            continue
        m_bool = m.astype(bool)
        cy, cx = center_of_mass(m_bool)
        vec_y = (cy - ys) * m_bool
        vec_x = (cx - xs) * m_bool
        dist  = np.sqrt(vec_y ** 2 + vec_x ** 2)
        dist_safe = np.where(dist > 1e-6, dist, 1.0)
        unit_y = np.where(m_bool, vec_y / dist_safe, 0.0).astype(np.float32)
        unit_x = np.where(m_bool, vec_x / dist_safe, 0.0).astype(np.float32)
        dy = np.where(m_bool, unit_y, dy)
        dx = np.where(m_bool, unit_x, dx)

    return fg, dy, dx


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: flow integration → vote map → watershed instances
# ─────────────────────────────────────────────────────────────────────────────

def flowfield_instances(
    pred_fg: np.ndarray,
    pred_dy: np.ndarray,
    pred_dx: np.ndarray,
    fg_threshold: float,
    n_steps: int,
    step_size: float,
    min_distance: int,
    vote_threshold: float,
    min_roi_px: int,
    compactness: float = 0.01,            # v2: now a parameter; default kept at v1 value for back-compat
) -> np.ndarray:
    """Convert dense (fg, dy, dx) predictions to binary instance masks (M, H, W).

    `compactness=0.0` follows the natural contour of the cell;
    `compactness>0` biases segments toward compact (round-ish) shapes.
    """
    fg_mask = pred_fg > fg_threshold
    if not fg_mask.any():
        return np.zeros((0, *pred_fg.shape), dtype=bool)

    H, W = pred_fg.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    Y = ys.copy()
    X = xs.copy()
    for _ in range(n_steps):
        py = np.clip(np.round(Y).astype(int), 0, H - 1)
        px = np.clip(np.round(X).astype(int), 0, W - 1)
        Y  = np.clip(Y + pred_dy[py, px] * step_size, 0.0, H - 1)
        X  = np.clip(X + pred_dx[py, px] * step_size, 0.0, W - 1)

    fg_ys, fg_xs = np.where(fg_mask)
    conv_y = np.clip(np.round(Y[fg_ys, fg_xs]).astype(int), 0, H - 1)
    conv_x = np.clip(np.round(X[fg_ys, fg_xs]).astype(int), 0, W - 1)

    votes = np.zeros((H, W), dtype=np.float32)
    np.add.at(votes, (conv_y, conv_x), 1.0)
    votes_smooth = gaussian_filter(votes, sigma=2.0)

    coords = peak_local_max(
        votes_smooth,
        min_distance=min_distance,
        threshold_abs=vote_threshold,
        labels=fg_mask,
        exclude_border=False,
    )
    if len(coords) == 0:
        return np.zeros((0, *pred_fg.shape), dtype=bool)

    seed_img = np.zeros(pred_fg.shape, dtype=bool)
    seed_img[coords[:, 0], coords[:, 1]] = True
    seed_labeled, _ = ndi_label(seed_img)
    labels = watershed(-votes_smooth, seed_labeled, mask=fg_mask, compactness=compactness)

    n_labels = int(labels.max())
    if n_labels == 0:
        return np.zeros((0, *pred_fg.shape), dtype=bool)
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    kept = [k for k in range(1, n_labels + 1) if areas[k] >= min_roi_px]
    if not kept:
        return np.zeros((0, *pred_fg.shape), dtype=bool)
    out = np.empty((len(kept), *pred_fg.shape), dtype=bool)
    for i, k in enumerate(kept):
        np.equal(labels, k, out=out[i])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Validation matching — bbox-restricted IoU (carried over from 05.12)
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_and_area(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = masks.shape[0]
    bboxes = np.zeros((N, 4), dtype=np.int32)
    areas  = np.zeros(N, dtype=np.int64)
    for i in range(N):
        m = np.asarray(masks[i]).astype(bool, copy=False)
        if not m.any():
            continue
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        ys = np.where(rows)[0]
        xs = np.where(cols)[0]
        bboxes[i] = (int(ys[0]), int(xs[0]), int(ys[-1]) + 1, int(xs[-1]) + 1)
        areas[i]  = int(m.sum())
    return bboxes, areas


def match_instances(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float,
) -> dict[str, float]:
    """Bbox-restricted greedy IoU matching → F1, precision, recall, count error."""
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    if n_pred == 0 and n_gt == 0:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0,
                "count_error": 0.0, "matched_iou": 1.0}
    if n_pred == 0 or n_gt == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "count_error": float(abs(n_pred - n_gt)), "matched_iou": 0.0}

    p_bbox, p_area = _bbox_and_area(pred_masks)
    g_bbox, g_area = _bbox_and_area(gt_masks)

    iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
    g_y0, g_x0, g_y1, g_x1 = g_bbox[:, 0], g_bbox[:, 1], g_bbox[:, 2], g_bbox[:, 3]
    for i in range(n_pred):
        py0, px0, py1, px1 = p_bbox[i]
        if py0 >= py1 or p_area[i] == 0:
            continue
        bbox_hits = np.where((g_y0 < py1) & (g_y1 > py0) & (g_x0 < px1) & (g_x1 > px0))[0]
        if bbox_hits.size == 0:
            continue
        p_i = np.asarray(pred_masks[i]).astype(bool, copy=False)
        for j in bbox_hits:
            uy0 = int(min(py0, g_y0[j]))
            ux0 = int(min(px0, g_x0[j]))
            uy1 = int(max(py1, g_y1[j]))
            ux1 = int(max(px1, g_x1[j]))
            pc = p_i[uy0:uy1, ux0:ux1]
            gc = np.asarray(gt_masks[j, uy0:uy1, ux0:ux1]).astype(bool, copy=False)
            inter = int(np.logical_and(pc, gc).sum())
            if inter == 0:
                continue
            union = int(p_area[i]) + int(g_area[j]) - inter
            if union > 0:
                iou_mat[i, j] = inter / union

    matched_ious: list[float] = []
    used_gt: set[int] = set()
    for pi in np.argsort(-iou_mat.max(axis=1)):
        best_gt = int(np.argmax(iou_mat[pi]))
        if iou_mat[pi, best_gt] >= iou_threshold and best_gt not in used_gt:
            matched_ious.append(float(iou_mat[pi, best_gt]))
            used_gt.add(best_gt)

    tp   = len(matched_ious)
    prec = tp / (n_pred + 1e-8)
    rec  = tp / (n_gt   + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "f1":          f1,
        "precision":   prec,
        "recall":      rec,
        "count_error": float(abs(n_pred - n_gt)),
        "matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_flow_loss(
    pred_dy: torch.Tensor,
    pred_dx: torch.Tensor,
    gt_dy:   torch.Tensor,
    gt_dx:   torch.Tensor,
    fg:      torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mag = torch.sqrt(pred_dy ** 2 + pred_dx ** 2 + eps)
    pdy_n = pred_dy / mag
    pdx_n = pred_dx / mag
    cos_sim = pdy_n * gt_dy + pdx_n * gt_dx
    per_pixel = (1.0 - cos_sim) * fg
    denom = fg.sum().clamp(min=1.0)
    return per_pixel.sum() / denom


def _mse_flow_loss(
    pred_dy: torch.Tensor,
    pred_dx: torch.Tensor,
    gt_dy:   torch.Tensor,
    gt_dx:   torch.Tensor,
    fg:      torch.Tensor,
) -> torch.Tensor:
    pdy_m = pred_dy * fg
    pdx_m = pred_dx * fg
    gdy_m = gt_dy   * fg
    gdx_m = gt_dx   * fg
    return nn.functional.mse_loss(pdy_m, gdy_m) + nn.functional.mse_loss(pdx_m, gdx_m)


def _tversky_loss(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta:  float,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss on binary fg target.

    alpha weights false positives, beta weights false negatives.
    alpha=beta=0.5 → Dice. alpha=0.7, beta=0.3 → recall-favouring (dense-ROI standard).
    Returns 1 - Tversky index averaged over batch.
    """
    pred = torch.sigmoid(pred_logit)
    target = target.float()
    # Per-sample reduction (batch is dim 0).
    dims = tuple(range(1, pred.dim()))
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Auto pos_weight from training data
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weight_fg(
    train_examples: list[RecordingExample],
    cap: float,
    fallback: float,
) -> float:
    """Inverse class frequency, capped — see config rationale section 1.

    Reads instance masks from disk (mmap), collapses to a single fg map per
    image, then averages fg-pixel ratio across the sampled images. Returns
    `min(cap, 1/ratio)` so an empty-ish image can't blow this up.
    """
    if not train_examples:
        return fallback
    fg_pixels = 0
    total_pixels = 0
    for ex in train_examples:
        masks = np.load(ex.mask_path, mmap_mode="r")
        if masks.shape[0] == 0:
            total_pixels += ex.height * ex.width
            continue
        fg = np.asarray(masks).any(axis=0)
        fg_pixels += int(fg.sum())
        total_pixels += int(fg.size)
    if total_pixels == 0:
        return fallback
    ratio = max(fg_pixels / total_pixels, 1e-4)
    return float(min(cap, 1.0 / ratio))


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window val inference (unchanged from 05.12)
# ─────────────────────────────────────────────────────────────────────────────

def _hann_2d(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    return np.outer(w, w) + 1e-3


def predict_full_image(
    model: nn.Module,
    img: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native-res sliding-window inference, GPU-side accumulation."""
    _, H, W = img.shape
    fg_accum = torch.zeros((H, W), dtype=torch.float32, device=device)
    dy_accum = torch.zeros((H, W), dtype=torch.float32, device=device)
    dx_accum = torch.zeros((H, W), dtype=torch.float32, device=device)
    wt_accum = torch.zeros((H, W), dtype=torch.float32, device=device)

    window = torch.from_numpy(_hann_2d(tile_size)).to(device)
    positions = tile_positions(H, W, tile_size, overlap)
    img_t_full = torch.from_numpy(img).to(device)

    model.eval()
    with torch.no_grad():
        for (top, left) in positions:
            tile = img_t_full[:, top:top + tile_size, left:left + tile_size].unsqueeze(0)
            _, _, th, tw = tile.shape
            if th < tile_size or tw < tile_size:
                tile = nn.functional.pad(tile, (0, tile_size - tw, 0, tile_size - th), mode="reflect")
            pred = model(tile)
            fg_t = torch.sigmoid(pred[0, 0])
            dy_t = torch.tanh(pred[0, 1])
            dx_t = torch.tanh(pred[0, 2])
            h_eff = min(tile_size, H - top)
            w_eff = min(tile_size, W - left)
            wt = window[:h_eff, :w_eff]
            fg_accum[top:top + h_eff, left:left + w_eff].add_(fg_t[:h_eff, :w_eff] * wt)
            dy_accum[top:top + h_eff, left:left + w_eff].add_(dy_t[:h_eff, :w_eff] * wt)
            dx_accum[top:top + h_eff, left:left + w_eff].add_(dx_t[:h_eff, :w_eff] * wt)
            wt_accum[top:top + h_eff, left:left + w_eff].add_(wt)

    wt_safe = wt_accum.clamp(min=1e-6)
    fg_full = (fg_accum / wt_safe).cpu().numpy()
    dy_full = (dy_accum / wt_safe).cpu().numpy()
    dx_full = (dx_accum / wt_safe).cpu().numpy()

    del fg_accum, dy_accum, dx_accum, wt_accum, wt_safe, img_t_full, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fg_full, dy_full, dx_full


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader collate
# ─────────────────────────────────────────────────────────────────────────────

def _collate(batch):
    imgs  = torch.stack([b[0] for b in batch])
    masks = [b[1] for b in batch]
    return imgs, masks


def _make_target_tensors(
    masks_list: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fgs, dys, dxs = [], [], []
    for masks in masks_list:
        fg, dy, dx = make_targets(masks)
        fgs.append(torch.from_numpy(fg))
        dys.append(torch.from_numpy(dy))
        dxs.append(torch.from_numpy(dx))
    return (
        torch.stack(fgs).to(device),
        torch.stack(dys).to(device),
        torch.stack(dxs).to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_flowfield_native(
    config: Config,
    train_examples_override: list[RecordingExample] | None = None,
    val_examples_override:   list[RecordingExample] | None = None,
) -> dict:
    """Train one FlowField model.

    Modes
    -----
    - **CV / held-out val** (default): set `config.val_fraction > 0` or pass
      `val_examples_override` with a non-empty list. Early stopping watches
      val F1 with `config.early_stop_patience`; `best_model.pth` = the
      highest-val-F1 weights.
    - **Production** (no val): pass `val_examples_override=[]`. The val loop
      is skipped, no early stopping, the LAST epoch's weights become both
      `last_model.pth` AND `best_model.pth` (so downstream code that expects
      best_model.pth still works).
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    overlays_dir = run_dir / "validation_overlays"
    overlays_dir.mkdir(exist_ok=True)

    print(f"Run dir   : {run_dir}")
    print(f"Device    : {device}")
    print(f"Flow loss : {config.ff_flow_loss}")
    print(f"Patch     : {config.patch_size}  x {config.patches_per_image}/img"
          f"   scale_aug={config.scale_aug}")
    print(f"Val tile  : {config.val_tile_size}  overlap={config.val_tile_overlap}")

    if train_examples_override is not None and val_examples_override is not None:
        train_ex, val_ex = train_examples_override, val_examples_override
    else:
        examples = discover_examples(config.training_dir)
        train_ex, val_ex = split_examples(examples, config.val_fraction, config.seed)
    print(f"Examples  : train={len(train_ex)}  val={len(val_ex)}")
    production_mode = (len(val_ex) == 0)
    if production_mode:
        print("Production mode: no val set; will skip per-epoch eval and treat last "
              "epoch as best.")

    # ── v2: auto fg pos_weight (see config docstring §1) ─────────────────
    if config.auto_pos_weight_fg:
        pw = compute_pos_weight_fg(train_ex, cap=config.auto_pos_weight_cap,
                                   fallback=config.pos_weight_fg)
        print(f"Auto pos_weight_fg: {pw:.3f}  (cap={config.auto_pos_weight_cap})")
    else:
        pw = config.pos_weight_fg
        print(f"pos_weight_fg (fixed): {pw:.3f}")
    config.pos_weight_fg = float(pw)  # persisted into config.json on disk

    train_ds = PatchDataset(
        train_ex,
        patch_size=config.patch_size,
        patches_per_image=config.patches_per_image,
        augment=config.augment,
        scale_aug=config.scale_aug,
        scale_range=config.scale_range,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
    )

    val_native: list[tuple[RecordingExample, np.ndarray, np.ndarray]] = [
        (ex, *load_native(ex.summary_path, ex.mask_path)) for ex in val_ex
    ]

    model = ResNetUNet(
        out_ch=3,
        pretrained=config.pretrained_encoder,
        apply_imagenet_norm=config.apply_imagenet_norm,
    ).to(device)

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("enc_")]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("enc_")]
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": config.lr * 0.1},
        {"params": decoder_params, "lr": config.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    pw_t   = torch.tensor([config.pos_weight_fg], device=device)
    bce_fg = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    flow_loss_fn = _cosine_flow_loss if config.ff_flow_loss == "cosine" else _mse_flow_loss

    history: list[dict] = []
    best_f1   = -1.0
    no_improve = 0
    best_ckpt = run_dir / "checkpoints" / "best_model.pth"

    with open(run_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    for epoch in range(config.epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []
        bce_acc, tv_acc, flow_acc = [], [], []
        for imgs, masks_list in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{config.epochs} Train", leave=False):
            imgs = imgs.to(device)
            fg_t, dy_t, dx_t = _make_target_tensors(masks_list, device)

            pred = model(imgs)
            pred_fg_logit = pred[:, 0]
            pred_dy       = torch.tanh(pred[:, 1])
            pred_dx       = torch.tanh(pred[:, 2])

            # v2: BCE + Tversky combined fg loss (see config §2)
            loss_bce = bce_fg(pred_fg_logit, fg_t)
            loss_tv  = _tversky_loss(pred_fg_logit, fg_t,
                                     config.tversky_alpha, config.tversky_beta)
            loss_fg  = (config.loss_fg_bce_weight     * loss_bce
                      + config.loss_fg_tversky_weight * loss_tv)

            loss_flow = flow_loss_fn(pred_dy, pred_dx, dy_t, dx_t, fg_t)

            # v2: weighted total loss (see config §3) — rebalances fg vs flow gradient mass
            loss = config.loss_fg_weight * loss_fg + config.loss_flow_weight * loss_flow

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            bce_acc.append(float(loss_bce.item()))
            tv_acc.append(float(loss_tv.item()))
            flow_acc.append(float(loss_flow.item()))

        scheduler.step()
        mean_train_loss = float(np.mean(train_losses))
        mean_bce  = float(np.mean(bce_acc))
        mean_tv   = float(np.mean(tv_acc))
        mean_flow = float(np.mean(flow_acc))

        # ── Validation ────────────────────────────────────────────
        val_metrics: list[dict] = []
        saved_overlays = 0
        for vi, (ex, img_native, gt_masks) in enumerate(val_native):
            fg_full, dy_full, dx_full = predict_full_image(
                model, img_native,
                tile_size=config.val_tile_size,
                overlap=config.val_tile_overlap,
                device=device,
            )
            pred_instances = flowfield_instances(
                fg_full, dy_full, dx_full,
                fg_threshold=config.ff_fg_threshold,
                n_steps=config.ff_n_steps,
                step_size=config.ff_step_size,
                min_distance=config.ff_min_distance,
                vote_threshold=config.ff_vote_threshold,
                min_roi_px=config.ff_min_roi_px,
                compactness=config.watershed_compactness,   # v2 §5
            )
            gt_bool = gt_masks.astype(bool)
            val_metrics.append(match_instances(pred_instances, gt_bool, config.metric_iou_threshold))

            if config.save_val_overlays and saved_overlays < config.max_val_overlays:
                save_comparison_panel(
                    img_native, gt_bool, pred_instances,
                    overlays_dir / f"val_example_{vi+1:02d}.png",
                    title=(
                        f"FlowField v2 native | ep{epoch+1} | "
                        f"GT={len(gt_bool)} Pred={len(pred_instances)}"
                    ),
                )
                saved_overlays += 1

        mean_metrics = (
            {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0]}
            if val_metrics else {}
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "loss_bce":   mean_bce,
            "loss_tv":    mean_tv,
            "loss_flow":  mean_flow,
            **mean_metrics,
        }
        history.append(row)

        print(
            f"Ep {epoch+1:02d}/{config.epochs} | "
            f"loss={mean_train_loss:.4f} (bce={mean_bce:.4f} tv={mean_tv:.4f} flow={mean_flow:.4f}) | "
            f"f1={mean_metrics.get('f1', 0):.4f} | "
            f"count_err={mean_metrics.get('count_error', 0):.1f}"
        )

        if production_mode:
            # No val set: every epoch overwrites best_model.pth so the FINAL
            # epoch's weights are what ship as the production checkpoint.
            torch.save(model.state_dict(), best_ckpt)
        else:
            # v2: early stopping on val F1 (see config §4)
            if mean_metrics.get("f1", 0) > best_f1:
                best_f1 = mean_metrics.get("f1", 0)
                no_improve = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                no_improve += 1

        torch.save(model.state_dict(), run_dir / "checkpoints" / "last_model.pth")
        _save_csv(history, run_dir / "history.csv")
        _plot_history(history, run_dir)

        if (not production_mode
            and config.early_stop_patience > 0
            and no_improve >= config.early_stop_patience):
            print(f"Early stop: val F1 has not improved for {no_improve} epochs "
                  f"(patience={config.early_stop_patience}).")
            break

    summary = {
        "run_dir":         str(run_dir),
        "best_f1":         best_f1,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(run_dir / "checkpoints" / "last_model.pth"),
        "n_train":         len(train_ex),
        "n_val":           len(val_ex),
        "epochs_completed": len(history),
        "auto_pos_weight_fg_used": float(config.pos_weight_fg),
        "apply_imagenet_norm":     bool(config.apply_imagenet_norm),
        "training_pixel_size_um":  config.training_pixel_size_um,
        "final_metrics":   history[-1] if history else {},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del model, optimizer, scheduler, train_loader, train_ds, val_native
    plt.close("all")
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def _plot_history(history: list[dict], run_dir: Path) -> None:
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [r["train_loss"]            for r in history], "o-", label="total")
    axes[0].plot(epochs, [r.get("loss_bce",  math.nan) for r in history], "o-", label="bce")
    axes[0].plot(epochs, [r.get("loss_tv",   math.nan) for r in history], "o-", label="tversky")
    axes[0].plot(epochs, [r.get("loss_flow", math.nan) for r in history], "o-", label="flow")
    axes[0].set_title("Train loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    if "f1" in history[0]:
        axes[1].plot(epochs, [r.get("f1",        math.nan) for r in history], "o-", label="F1")
        axes[1].plot(epochs, [r.get("recall",    math.nan) for r in history], "o-", label="Recall")
        axes[1].plot(epochs, [r.get("precision", math.nan) for r in history], "o-", label="Precision")
        axes[1].legend()
        axes[1].set_title("Val metrics"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
