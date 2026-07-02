# NEURAL: Network-based Extraction (with) Unbiased ROI Analysis (in) Live-imaging

## Installation

NEURAL needs Python 3.10+. The dependency list lives in `requirements.txt`, and
`setup_env.ps1` bootstraps a ready-to-run virtualenv (installs everything,
including the correct torch build for your GPU).

```powershell
# one command: creates ..\.venv and installs all deps + torch (default cu121;
# use -Cuda cpu for CPU-only, or -Cuda cu118 for older CUDA)
powershell -ExecutionPolicy Bypass -File NEURAL\setup_env.ps1
```

Or manually:

```bash
pip install -r NEURAL/requirements.txt
# torch is GPU-specific — install the build matching your CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

torch/torchvision are intentionally not pinned in `requirements.txt` because the
correct wheel depends on your CUDA version — see the comments in that file.

## Layout

```
NEURAL/
├── peakcaller/         # convex / ALS baseline, multiplicative D, find_peaks
├── nn/                 # neural network
│   ├── scripts/        # training + inference (14 .py files)
│   └── models/         # 4 production checkpoints (see below)
├── bursting/           # chained-IEI burst grouping + 5 metrics
├── synchronicity/      # FluoroSNAPP per-recording sync (Dice) + FC
├── figure_engine/      # render trace_grid / roi_overlay / summary_panel for ONE dataset
├── io/                 # nd2 + tif loader with manual fps/pixel_size/length overrides
├── utils/              # Excel I/O + Windows long-path helpers
├── config_examples/    # 1 JSON template per NN model
├── analyze_dataset.ipynb   # one-recording-or-folder analysis
└── train_nn.ipynb          # train a new model on labeled ROIs
```

## Four NN models

| model              | path                                                 |
|--------------------|------------------------------------------------------|
| `iglusnfr`         | `nn/models/iglusnfr_production/`                     |
| `calcium_imaging`  | `nn/models/calcium_imaging_production/`              |
| `igabasnfr`        | `nn/models/igabasnfr_production/`                    |
| `human_gc8`        | `nn/models/human_gc8_production/`                    |

Each has `checkpoints/best_model.pth`, `config.json`, `history.csv`,
`summary.json`, `training_curves.png`, `PRODUCTION.json`.

## Two peak detection paths

Both live in `NEURAL/peakcaller/`:
- **convex envelope baseline** (default for positive-going indicators)
- **ALS baseline** (default for iGABASnFr / negative-going indicators)

Both feed into the same multiplicative-D + `find_peaks` event detector.

## Configuration

All knobs live in JSON. The schema (ordered by edit frequency):

```json
{
  "peak_detection": {
    "threshold": 0.03,
    "baseline":  "convex",     // "convex" | "als"
    "negate":    false         // true for negative-going indicators
  },
  "nn": {
    "model":      "iglusnfr",  // "iglusnfr" | "calcium_imaging" | "igabasnfr" | "human_gc8"
    "min_roi_px": 32,
    "fg_threshold": 0.7,
    "use_cached_inference": true
  },
  "visualization": {
    "kind":              "all",       // "trace_grid" | "roi_overlay" | "summary_panel" | "all"
    "n_rois_per_video":  10,
    "roi_selection":     "top_activity",  // "top_activity" | "random" | "all" | "by_index"
    "roi_indices":       null,
    "videos_to_show":    null
  },
  "input": {
    "path":          "/path/to/file_or_folder",
    "format":        "auto",   // "auto" | "nd2" | "tif"
    "inference_dir": null,
    "fps":           null,     // null = read from nd2 metadata; number = manual override
    "pixel_size_um": null,
    "n_frames":      null,
    "duration_s":    null
  },
  "output": {
    "dir":    "./output",
    "format": "pdf",           // "pdf" | "png"
    "dpi":    140
  }
}
```

`fps` / `pixel_size_um` / `n_frames` / `duration_s` are **auto-detected from
nd2 metadata** when set to `null`; pass an explicit number to override (required
for `.tif` input). `inference_dir` points at a cached NN output folder; if
absent or `use_cached_inference: false`, NN inference is re-run.

Templates: `config_examples/{iglusnfr, calcium_imaging, igabasnfr, human_gc8}_basic.json`.

## Quick start

```python
from NEURAL.figure_engine import render_dataset
render_dataset("NEURAL/config_examples/iglusnfr_basic.json")
```

…or open `analyze_dataset.ipynb` and edit the config dict inline.

## Train a new NN

Open `train_nn.ipynb`. Point `data.training_dir` at a folder of
`<stem>_summary_rgb.npy` + `<stem>_instance_masks.npy` pairs. New model lands
at `nn/models/<your_run_name>/`.
