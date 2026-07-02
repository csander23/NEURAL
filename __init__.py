"""NEURAL — Network-based Extraction (with) Unbiased ROI Analysis (in) Live-imaging.

Subject of the methods paper. Knows how to:
  - segment ROIs with a neural network (4 pre-trained production models supplied)
  - extract per-ROI traces from .nd2 or .tif video
  - detect events with PeakCaller (convex or ALS baseline)
  - render single-dataset trace_grid / roi_overlay / summary panels
  - train a new NN from labeled ROIs

Knows NOTHING about the manuscript's specific 6 experiments or any
manual-vs-NN comparison; that all lives in ../Methods_Paper/.

Top-level shape:
  NEURAL.peakcaller      — convex/ALS baseline + multiplicative D + find_peaks
  NEURAL.nn              — Mask R-CNN training + inference (scripts/, models/)
  NEURAL.bursting        — chained-IEI burst grouping + 5 metrics
  NEURAL.synchronicity   — FluoroSNAPP per-recording sync (Dice) + FC (Pearson)
  NEURAL.figure_engine   — render trace_grid / roi_overlay / summary for ONE dataset
  NEURAL.io              — nd2/tif loader with manual fps/pixel_size/length overrides
  NEURAL.utils           — Excel I/O + Windows long-path helpers
  NEURAL.config_examples — JSON templates for each NN model

Entry points:
  from NEURAL.figure_engine import render_dataset
  render_dataset("NEURAL/config_examples/iglusnfr_basic.json")

  # or open one of:
  NEURAL/analyze_dataset.ipynb   — run a single recording or folder
  NEURAL/train_nn.ipynb          — fit a new model on labeled ROIs
"""
__version__ = "1.1.0"
