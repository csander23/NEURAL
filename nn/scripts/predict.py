"""Standalone inference for 05.13 FlowField checkpoints with pixel-scale awareness.

The model was trained at a fixed pixel scale (`training_pixel_size_um` in
config.json — saved when known). At inference time, a new image may come from
a different microscope / objective / binning, so cells could be larger or
smaller in pixels than the model expects. This script:

1. Reads the input file (.nd2, .tif/.tiff, or .npy). For .nd2 / .tif the pixel
   size is auto-detected from the file's metadata (override with
   --input-pixel-size-um). For .npy the user must specify, since the numpy
   container has no calibration metadata.
2. Computes a resize factor `s = input_pixel_size / training_pixel_size`.
   If `s > 1` the input has *larger* pixels (cells are smaller in px); we
   upsample the input. If `s < 1` we downsample. After model inference, the
   predicted instance masks are resized back to the original input shape.
3. For .nd2 / .tif inputs that are *time-series videos*, computes the same
   3-channel summary (mean / std / temporal correlation) the training pipeline
   used. For .npy inputs we assume the file is already a 3-channel summary.

Usage
-----
    # ND2 — pixel size auto-detected from metadata
    python predict.py path/to/recording.nd2 --preset iglusnfr --out out.npy

    # NPY — must supply both pixel sizes
    python predict.py path/to/recording_summary_rgb.npy --preset iglusnfr \
        --training-pixel-size-um 0.223 --input-pixel-size-um 0.223 --out out.npy

    # Explicit checkpoint instead of preset
    python predict.py path/to/img.tif --checkpoint runs/.../best_model.pth \
        --config     runs/.../config.json --out out.npy

Output: a single `.npy` file with shape `(M, H_orig, W_orig)` bool — one
binary mask per detected ROI, at the input's original spatial size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize as sk_resize

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, PRESET_TRAINING_DIRS
from scripts.resnet_unet import ResNetUNet
from scripts.approach_flowfield import predict_full_image, flowfield_instances


RUNS_ROOT = Path(__file__).parent.parent / "data" / "trained_models"


# ─────────────────────────────────────────────────────────────────────────────
# Input loading: produce (3, H, W) z-scored float32 + (H_orig, W_orig)
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_per_channel(img_hw3: np.ndarray) -> np.ndarray:
    """Per-channel z-score, returns (3, H, W) float32 — same as training."""
    out = np.empty_like(img_hw3, dtype=np.float32)
    for c in range(img_hw3.shape[-1]):
        ch = img_hw3[..., c].astype(np.float32)
        mu, sigma = ch.mean(), ch.std()
        out[..., c] = (ch - mu) / (sigma + 1e-8)
    return out.transpose(2, 0, 1)


def load_input(path: Path, manual_pixel_size_um: float | None = None) -> tuple[np.ndarray, tuple[int, int], float | None]:
    """Load an input file, return (img_3hw_zscored, (H_orig,W_orig), pixel_size_um_or_None).

    For .nd2 / .tif video inputs, builds the 3-channel summary by **streaming
    frames** through `summary_from_path` — never holds the full (T,H,W) video
    in memory. Replaces a previous version that called `.asarray()` and then
    cast to float32, blowing past 50 GB of RAM on long recordings.
    """
    from scripts.summary_images import summary_from_path

    suf = path.suffix.lower()
    if suf == ".nd2":
        summary, detected_px = summary_from_path(path)
        px = manual_pixel_size_um if manual_pixel_size_um is not None else detected_px
        H, W = summary.shape[:2]
        return _zscore_per_channel(summary), (H, W), px

    if suf in (".tif", ".tiff"):
        # TIFFs can be either a multi-frame video or a pre-baked 2D / 3D
        # summary image. Peek at the first page to distinguish without
        # materialising the whole file.
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            n_pages = len(tf.pages)
            sample = np.asarray(tf.pages[0].asarray())

        if n_pages == 1 and sample.ndim == 3 and sample.shape[-1] == 3:
            summary = sample.astype(np.float32)
        elif n_pages == 1 and sample.ndim == 2:
            summary = np.stack([sample, sample, sample], axis=-1).astype(np.float32)
        else:
            summary, _ = summary_from_path(path)

        H, W = summary.shape[:2]
        px = manual_pixel_size_um
        if px is None:
            try:
                from detect_pixel_size import detect_tif
                px, _ = detect_tif(path)
            except Exception:
                px = None
        return _zscore_per_channel(summary), (H, W), px

    if suf == ".npy":
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f".npy input must be (H, W, 3); got shape {arr.shape}")
        H, W = arr.shape[:2]
        # Already-summary, re-z-score for safety (training pipeline does this).
        return _zscore_per_channel(arr.astype(np.float32)), (H, W), manual_pixel_size_um

    raise ValueError(f"unsupported file type: {suf}")


# ─────────────────────────────────────────────────────────────────────────────
# Resize / un-resize helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_img_3hw(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a (3, H, W) array to (3, target_h, target_w)."""
    H, W = target_hw
    out = sk_resize(img.transpose(1, 2, 0), (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)
    return out.transpose(2, 0, 1)


def resize_masks_back(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a (M, H, W) bool stack to (M, target_h, target_w)."""
    if masks.shape[0] == 0:
        return np.zeros((0, *target_hw), dtype=bool)
    H, W = target_hw
    # skimage's batch axis is the LAST dim, so transpose around it.
    resized = sk_resize(
        masks.transpose(1, 2, 0).astype(np.float32),
        (H, W),
        preserve_range=True, anti_aliasing=False, order=0,
    )
    return (resized > 0.5).transpose(2, 0, 1).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint + config resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(args) -> tuple[Path, Path]:
    if args.checkpoint and args.config:
        return Path(args.checkpoint), Path(args.config)
    if args.run_name:
        run_dir = RUNS_ROOT / args.run_name
        return run_dir / "checkpoints" / "best_model.pth", run_dir / "config.json"
    raise SystemExit("Provide either --run-name or both --checkpoint and --config.")


def _load_model(ckpt: Path, config: Config, device: torch.device) -> ResNetUNet:
    model = ResNetUNet(
        out_ch=3,
        pretrained=False,
        apply_imagenet_norm=config.apply_imagenet_norm,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def _build_config(cfg_path: Path) -> Config:
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    allowed = {k: v for k, v in cfg_data.items() if k in Config.__dataclass_fields__}
    # training_dir not used at inference; preset already located the ckpt.
    if "training_dir" not in allowed:
        allowed["training_dir"] = ""
    return Config(**allowed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path,
                   help="Input file: .nd2 / .tif / .npy")
    p.add_argument("--run-name", type=str, default=None,
                   help="Use the run dir at data/trained_models/<run-name>/.")
    p.add_argument("--checkpoint", type=Path, help="Path to best_model.pth (use with --config)")
    p.add_argument("--config",     type=Path, help="Path to that run's config.json")
    p.add_argument("--training-pixel-size-um", type=float, default=None,
                   help="Override the training pixel size. Required if the model's "
                        "config.json doesn't have it set.")
    p.add_argument("--input-pixel-size-um", type=float, default=None,
                   help="Override the input pixel size. Auto-detected for .nd2/.tif "
                        "from file metadata; required for .npy.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .npy path for the (M, H_orig, W_orig) bool mask stack.")
    p.add_argument("--no-resize", action="store_true",
                   help="Skip the pixel-scale resize even when both sizes are known.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    ckpt, cfg_path = _resolve_checkpoint(args)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    config = _build_config(cfg_path)
    training_px = args.training_pixel_size_um or config.training_pixel_size_um
    print(f"checkpoint    : {ckpt}")
    print(f"config        : {cfg_path}")
    print(f"training_px_um: {training_px}")
    print(f"apply_imagenet_norm: {config.apply_imagenet_norm}")

    # Load input
    img_3hw, orig_hw, input_px = load_input(args.input, manual_pixel_size_um=args.input_pixel_size_um)
    print(f"input         : {args.input.name}  shape={orig_hw}  input_px_um={input_px}")

    # Resize input if both pixel sizes are known and differ.
    work_hw = orig_hw
    if (not args.no_resize) and training_px and input_px:
        scale = input_px / training_px
        if abs(scale - 1.0) > 0.01:
            new_h = max(64, int(round(orig_hw[0] * scale)))
            new_w = max(64, int(round(orig_hw[1] * scale)))
            print(f"resize        : {orig_hw} -> ({new_h}, {new_w})  (scale={scale:.3f})")
            img_3hw = resize_img_3hw(img_3hw, (new_h, new_w))
            work_hw = (new_h, new_w)
        else:
            print(f"resize        : skip (scale {scale:.3f} ~= 1.0)")
    elif not training_px or not input_px:
        print(f"resize        : skip (training_px or input_px is None — pass --training-pixel-size-um")
        print(f"                and --input-pixel-size-um to enable scale correction)")

    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device        : {device}")
    model = _load_model(ckpt, config, device)
    fg, dy, dx = predict_full_image(
        model, img_3hw,
        tile_size=config.val_tile_size,
        overlap=config.val_tile_overlap,
        device=device,
    )
    instances = flowfield_instances(
        fg, dy, dx,
        fg_threshold=config.ff_fg_threshold,
        n_steps=config.ff_n_steps,
        step_size=config.ff_step_size,
        min_distance=config.ff_min_distance,
        vote_threshold=config.ff_vote_threshold,
        min_roi_px=config.ff_min_roi_px,
        compactness=config.watershed_compactness,
    )
    print(f"detected      : {instances.shape[0]} instances at work resolution {work_hw}")

    # Resize masks back to original input resolution
    if work_hw != orig_hw:
        instances = resize_masks_back(instances, orig_hw)
        print(f"resized masks back to {orig_hw}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, instances)
    print(f"saved         : {args.out}  shape={instances.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
