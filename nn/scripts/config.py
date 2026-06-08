"""Configuration for 05.13.2026 — FlowField native-resolution v2.

What's new vs 05.12.2026 (driven by the post-hoc threshold sweep on 05.12 v1):

1. **auto_pos_weight_fg**: derived from training-data fg ratio, capped at 5.
   Reason: every preset's calibrated best `fg_threshold` was 0.70, far from
   the BCE-optimal 0.50. That means the fg head was producing a peaked
   distribution — `pos_weight=8` was over-correcting class imbalance. With
   the auto-weight we expect the decision boundary to land near 0.5 and the
   model to be less brittle to threshold choices.

2. **Tversky + BCE combined fg loss**. BCE optimizes per-pixel correctness,
   Tversky directly optimizes overlap (IoU). matched_iou stayed at 0.58–0.61
   across v1 — boundary-quality bottleneck. Tversky with alpha=0.7, beta=0.3
   favours recall (false-negative cost > false-positive cost), matching
   dense-ROI cell-detection convention.

3. **loss_flow_weight=2.0**. v1 final-epoch losses on calcium: fg=1.23 vs
   flow=0.46. With weights 1:1 the flow gradient was effectively 1/3 of fg.
   Flow undertraining is also why the calibrated vote_threshold varied so
   much between modalities (5 / 8 / 12) — predicted flow magnitudes were
   modality-dependent.

4. **early_stop_patience=10**. iGABASnFr v1 best F1 was at epoch 8 of 30;
   train loss kept falling while val F1 dropped. 22 wasted epochs and a
   risk of threshold-drift damage to the saved best.

5. **watershed_compactness=0.0**. v1 hardcoded 0.01, which biases watershed
   toward round blobs. 0.0 lets cells follow their natural boundary contour.
   Targeted at the matched_iou ceiling.

Also: starting from `ff_fg_threshold=0.7` since that was the unanimous
sweep winner — even after retraining we expect the optimal to stay near
0.5 (since auto-weight changes the distribution), but 0.7 is the safe
initial value for any single forward pass before threshold tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_ACTUAL_EXPORTS = Path(__file__).parent.parent.parent.parent

EXPERIMENT_ROOT = Path(__file__).parent.parent

PRESET_TRAINING_DIRS: dict[str, Path] = {
    "calcium_imaging": _ACTUAL_EXPORTS / "calcium_imaging" / "data" / "training_data",
    "iglusnfr":        _ACTUAL_EXPORTS / "iglusnfr"        / "data" / "training_data",
    "igabasnfr":       _ACTUAL_EXPORTS / "igabasnfr"       / "data" / "training_data",
}


@dataclass
class Config:
    training_dir: str
    preset_name: str | None = None
    run_name: str | None = None
    approach: str = "flowfield_native_v2"

    # ── Data: native-resolution patch training ────────────────────
    patch_size: int = 384
    patches_per_image: int = 8
    min_fg_fraction_in_patch: float = 0.0
    val_fraction: float = 0.2
    seed: int = 42

    # ── Validation: sliding-window inference at native resolution ─
    val_tile_size: int = 384
    val_tile_overlap: int = 192

    # ── Training ──────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    num_workers: int = 0

    # ── Foreground loss (v2: BCE + Tversky, auto class-balance weight) ────
    # See module docstring section 1 + 2 for rationale.
    auto_pos_weight_fg: bool = True
    auto_pos_weight_cap: float = 5.0          # cap so a single very-empty image can't blow this up
    pos_weight_fg: float = 8.0                # used as fallback if auto disabled OR computation fails
    loss_fg_bce_weight: float = 0.5
    loss_fg_tversky_weight: float = 0.5
    tversky_alpha: float = 0.7                # FP weight (lower = more permissive about FPs)
    tversky_beta: float = 0.3                 # FN weight (higher = more aggressive about recall)

    # ── Loss combine weights (v2: rebalance fg vs flow gradient magnitudes) ─
    loss_fg_weight: float = 1.0
    loss_flow_weight: float = 2.0             # was implicit 1.0 in v1

    # ── Augmentation ──────────────────────────────────────────────
    augment: bool = True
    scale_aug: bool = False
    scale_range: tuple[float, float] = (0.7, 1.5)

    # ── Model (ResNet18 U-Net) ────────────────────────────────────
    pretrained_encoder: bool = True
    # When True, ResNetUNet applies ImageNet (mean, std) normalisation to its
    # input before the encoder. Use with `pretrained_encoder=True` so conv1's
    # filters see the input distribution they were trained on. Note: input is
    # already per-image z-scored upstream — see ResNetUNet docstring for the
    # subtle interaction. Default False = behaviour matches the 05.13 v2 runs.
    apply_imagenet_norm: bool = False

    # ── Pixel-scale awareness ─────────────────────────────────────
    # Physical pixel size (μm/px) of the training images. Used at inference
    # time by `predict.py` to resize new inputs to the scale the model was
    # trained at. None = unknown; predict.py will warn and skip the resize.
    # For Nikon ND2 inputs, can be auto-detected with detect_pixel_size.py
    # (or by `nd2.ND2File(path).voxel_size().x`).
    training_pixel_size_um: float | None = None

    # ── Flow-field loss / post-processing ─────────────────────────
    ff_flow_loss: str = "cosine"
    ff_n_steps: int = 10
    ff_step_size: float = 3.0
    ff_min_distance: int = 12
    ff_fg_threshold: float = 0.7              # v1 sweep winner; expect retune after v2 training
    ff_vote_threshold: float = 5.0
    ff_min_roi_px: int = 32

    # ── Watershed post-processing (v2: compactness=0 for boundary fidelity) ─
    watershed_compactness: float = 0.0

    # ── Early stopping (v2) ──────────────────────────────────────
    # >0 enables; abort training if val F1 hasn't improved for N consecutive epochs.
    early_stop_patience: int = 10

    # ── Validation / output ───────────────────────────────────────
    metric_iou_threshold: float = 0.5
    output_root: str = str(EXPERIMENT_ROOT / "runs")
    save_val_overlays: bool = False           # default OFF — overlays are an OOM hazard at native res
    max_val_overlays: int = 4

    def run_dir(self) -> Path:
        name = self.run_name or f"{self.approach}_{self.preset_name or 'custom'}"
        return Path(self.output_root) / name

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["scale_range"] = list(self.scale_range)
        return d
