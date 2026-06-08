"""Post-hoc sweep of ff_fg_threshold x ff_vote_threshold for 05.13 v2 runs.

Same caching strategy as ../05.12.2026/tune_thresholds.py:
1. Tiled sliding-window inference via `predict_full_image` (the only
   inference path the model has ever seen).
2. FlowField only (this folder has no other approaches).
3. Cache `(fg, dy, dx)` once, sweep purely on CPU.
4. Save overlays only at the best threshold (matplotlib OOM hazard).

v2-specific differences vs 05.12 sweep:
- Default fg grid extends DOWN to 0.20. We expect the optimum to drop
  from 05.12's unanimous 0.70 back toward 0.50 because the auto pos_weight
  produces a less skewed fg distribution.
- Passes `compactness=config.watershed_compactness` to flowfield_instances
  (the v2 default is 0.0; old configs default to 0.01 via the function
  signature, so back-compat is preserved).

Usage:
    python tune_thresholds.py                    # all 3 presets
    python tune_thresholds.py --preset iglusnfr  # one preset
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, PRESET_TRAINING_DIRS
from scripts.data import discover_examples, split_examples, load_native
from scripts.resnet_unet import ResNetUNet
from scripts.approach_flowfield import (
    predict_full_image, flowfield_instances, match_instances,
)
from scripts.viz import save_comparison_panel


RUNS_ROOT = Path(__file__).parent.parent / "data" / "trained_models"

# v2: extended downward to 0.20 to test the auto-pos-weight hypothesis
# (optimum should drop from 05.12's universal 0.70 toward 0.50).
DEFAULT_FG_GRID   = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
DEFAULT_VOTE_GRID = [2.0, 3.0, 5.0, 8.0, 12.0, 16.0]


def _build_config_from_disk(cfg_data: dict) -> Config:
    """Construct a Config from a run's saved config.json (training_dir as-is)."""
    allowed = {k: v for k, v in cfg_data.items() if k in Config.__dataclass_fields__}
    return Config(**allowed)


def _load_model(ckpt: Path, device: torch.device) -> ResNetUNet:
    model = ResNetUNet(out_ch=3, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def sweep_preset(
    preset: str,
    fg_grid: list[float],
    vote_grid: list[float],
    run_name_override: str | None = None,
) -> dict | None:
    # Final_NN convention: run names are user-chosen; default to the preset name.
    run_name = run_name_override or preset
    run_dir  = RUNS_ROOT / run_name
    ckpt     = run_dir / "checkpoints" / "best_model.pth"
    cfg_path = run_dir / "config.json"
    if not ckpt.exists():
        print(f"  [skip] no checkpoint: {ckpt}")
        return None
    if not cfg_path.exists():
        print(f"  [skip] no config: {cfg_path}")
        return None

    with open(cfg_path) as f:
        cfg_data = json.load(f)
    config = _build_config_from_disk(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FlowField v2  |  {preset}  |  {run_name}")
    print(f"  device : {device}")
    print(f"  ckpt   : {ckpt.name}  (best_f1 from training)")
    print(f"  grid   : {len(fg_grid)} x {len(vote_grid)} = {len(fg_grid)*len(vote_grid)} combos")
    print(f"  watershed_compactness: {config.watershed_compactness}")
    print(f"{'='*60}")

    examples = discover_examples(config.training_dir)
    _, val_ex = split_examples(examples, config.val_fraction, config.seed)
    if not val_ex:
        print(f"  [skip] no validation examples for {preset}")
        return None
    print(f"  val examples: {len(val_ex)}")

    model = _load_model(ckpt, device)

    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for ex in val_ex:
        img, gt_masks = load_native(ex.summary_path, ex.mask_path)
        fg, dy, dx = predict_full_image(
            model, img,
            tile_size=config.val_tile_size,
            overlap=config.val_tile_overlap,
            device=device,
        )
        cache.append((fg, dy, dx, gt_masks.astype(bool)))
        print(f"    cached : {ex.stem}  ({gt_masks.shape[0]} GT masks)")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[dict] = []
    for fg_th, vote_th in product(fg_grid, vote_grid):
        per_image: list[dict] = []
        for (fg, dy, dx, gt) in cache:
            pred_inst = flowfield_instances(
                fg, dy, dx,
                fg_threshold=fg_th,
                n_steps=config.ff_n_steps,
                step_size=config.ff_step_size,
                min_distance=config.ff_min_distance,
                vote_threshold=vote_th,
                min_roi_px=config.ff_min_roi_px,
                compactness=config.watershed_compactness,   # v2 §5
            )
            per_image.append(match_instances(pred_inst, gt, config.metric_iou_threshold))

        results.append({
            "fg_th":       fg_th,
            "vote_th":     vote_th,
            "f1":          float(np.mean([x["f1"]          for x in per_image])),
            "precision":   float(np.mean([x["precision"]   for x in per_image])),
            "recall":      float(np.mean([x["recall"]      for x in per_image])),
            "count_error": float(np.mean([x["count_error"] for x in per_image])),
            "matched_iou": float(np.mean([x["matched_iou"] for x in per_image])),
        })

    results.sort(key=lambda r: (-r["f1"], r["count_error"]))

    print(f"\n  {'fg_th':>6}  {'vote_th':>7}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'CountErr':>9}  {'mIoU':>5}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*5}")
    for r in results[:10]:
        print(
            f"  {r['fg_th']:>6.2f}  {r['vote_th']:>7.2f}  "
            f"{r['f1']:>6.4f}  {r['precision']:>6.4f}  "
            f"{r['recall']:>6.4f}  {r['count_error']:>9.1f}  "
            f"{r['matched_iou']:>5.3f}"
        )

    best = results[0]
    default_f1 = next(
        (r["f1"] for r in results
         if r["fg_th"] == config.ff_fg_threshold and r["vote_th"] == config.ff_vote_threshold),
        None,
    )
    print(
        f"\n  Best     : fg_th={best['fg_th']}  vote_th={best['vote_th']}  "
        f"F1={best['f1']:.4f}  count_err={best['count_error']:.1f}"
    )
    if default_f1 is not None:
        print(f"  Defaults : fg_th={config.ff_fg_threshold}  vote_th={config.ff_vote_threshold}  F1={default_f1:.4f}")
        print(f"  Delta F1 : {best['f1'] - default_f1:+.4f}")

    sweep_path = run_dir / "threshold_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({
            "approach": "flowfield_native_v2",
            "preset":   preset,
            "fg_grid":   fg_grid,
            "vote_grid": vote_grid,
            "default":   {"fg_th": config.ff_fg_threshold, "vote_th": config.ff_vote_threshold,
                          "f1": default_f1},
            "best":      best,
            "all_results": results,
        }, f, indent=2)
    print(f"  Saved sweep -> {sweep_path.name}")

    cfg_data["ff_fg_threshold"]   = float(best["fg_th"])
    cfg_data["ff_vote_threshold"] = float(best["vote_th"])
    cfg_data["calibrated_thresholds"] = True
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    print(f"  Updated config.json with calibrated thresholds.")

    overlays_dir = run_dir / "validation_overlays_best_threshold"
    overlays_dir.mkdir(exist_ok=True)
    for vi, ((fg, dy, dx, gt), ex) in enumerate(zip(cache, val_ex)):
        if vi >= config.max_val_overlays:
            break
        pred_inst = flowfield_instances(
            fg, dy, dx,
            fg_threshold=best["fg_th"],
            n_steps=config.ff_n_steps,
            step_size=config.ff_step_size,
            min_distance=config.ff_min_distance,
            vote_threshold=best["vote_th"],
            min_roi_px=config.ff_min_roi_px,
            compactness=config.watershed_compactness,
        )
        img, _ = load_native(ex.summary_path, ex.mask_path)
        save_comparison_panel(
            img, gt, pred_inst,
            overlays_dir / f"val_{vi+1:02d}_fg{best['fg_th']}_vote{best['vote_th']}.png",
            title=(
                f"FlowField v2 | {preset} | calibrated | "
                f"GT={len(gt)} Pred={len(pred_inst)} F1={best['f1']:.3f}"
            ),
        )
    print(f"  Saved best-threshold overlays -> {overlays_dir.name}/")

    del cache
    gc.collect()
    return {"preset": preset, "best": best, "default_f1": default_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep on a Final_NN run.")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run dir under data/trained_models/<run-name>/.")
    parser.add_argument("--fg",   nargs="+", type=float, default=DEFAULT_FG_GRID)
    parser.add_argument("--vote", nargs="+", type=float, default=DEFAULT_VOTE_GRID)
    args = parser.parse_args()

    s = sweep_preset(
        preset=args.run_name,           # used only for display headers
        fg_grid=args.fg,
        vote_grid=args.vote,
        run_name_override=args.run_name,
    )
    summaries: list[dict] = [s] if s is not None else []

    if summaries:
        print(f"\n\n{'='*60}\n  Summary across presets\n{'='*60}")
        print(f"  {'preset':<18}  {'default F1':>10}  {'best F1':>8}  {'delta':>7}  {'fg':>5}  {'vote':>5}")
        for s in summaries:
            d = s["default_f1"]
            b = s["best"]["f1"]
            dlt = (b - d) if d is not None else float("nan")
            d_str = f"{d:.4f}" if d is not None else "  n/a "
            print(f"  {s['preset']:<18}  {d_str:>10}  {b:>8.4f}  {dlt:>+7.4f}  "
                  f"{s['best']['fg_th']:>5.2f}  {s['best']['vote_th']:>5.2f}")


if __name__ == "__main__":
    main()
