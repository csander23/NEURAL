"""NEURAL.figure_engine - render NN-side analysis figures for a SINGLE dataset.

This is the pipeline's own renderer. It knows nothing about manuscript
experiments, comparison panels, or non-NN methods. For those, use
Paper.paper_engine.

Public entry-point:
    render_dataset(config_path_or_dict, out_dir=None) -> dict

Config schema (JSON, ordered by edit frequency):
{
  "peak_detection": {
    "threshold": 0.03,
    "baseline":  "convex" | "als",
    "negate":    false
  },
  "nn": {
    "model":      "iglusnfr" | "calcium_imaging" | "igabasnfr" | "human_gc8",
    "min_roi_px": 32,
    "fg_threshold": 0.7,
    "use_cached_inference": true
  },
  "visualization": {
    "kind":              "trace_grid" | "roi_overlay" | "summary_panel" | "all",
    "n_rois_per_video":  10,
    "roi_selection":     "top_activity" | "random" | "all" | "by_index",
    "roi_indices":       null,
    "videos_to_show":    null
  },
  "input": {
    "path":          "<file or folder>",
    "format":        "auto" | "nd2" | "tif",
    "inference_dir": null,
    "fps":           null,
    "pixel_size_um": null,
    "n_frames":      null,
    "duration_s":    null
  },
  "output": {
    "dir":    "./output",
    "format": "pdf",
    "dpi":    140
  }
}
"""
from .render import render_dataset

__all__ = ["render_dataset"]
