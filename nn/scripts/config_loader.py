"""Load parameters.yaml into a Config dataclass.

Single source of truth for all knobs lives in `Final_NN/config/parameters.yaml`.
This module reads that file and constructs a `scripts.config.Config` instance
the rest of the pipeline can use.

Usage from the notebook:
    from scripts.config_loader import load_params
    params = load_params()                    # default location
    config = params.to_config(preset_name="my_run", training_dir="data/training_data")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Config

FINAL_NN_ROOT = Path(__file__).parent.parent
DEFAULT_YAML  = FINAL_NN_ROOT / "config" / "parameters.yaml"


@dataclass
class ResolvedParams:
    """The parsed YAML, ready to construct a Config from."""
    raw: dict[str, Any]

    # Common conveniences
    paths: dict[str, str]
    training_pixel_size_um: float | None
    threshold_sweep: dict[str, list[float]] = field(default_factory=dict)
    analysis: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ResolvedParams":
        p = Path(path) if path else DEFAULT_YAML
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            raw=data,
            paths={k: str((FINAL_NN_ROOT / v).resolve()) for k, v in data.get("paths", {}).items()},
            training_pixel_size_um=data.get("training_pixel_size_um"),
            threshold_sweep=data.get("threshold_sweep", {}),
            analysis=data.get("analysis", {}),
        )

    def analysis_params(self):
        """Return an AnalysisParams instance built from the YAML 'analysis' block."""
        from scripts.analysis import AnalysisParams
        return AnalysisParams(**{k: v for k, v in self.analysis.items()
                                 if k in AnalysisParams.__dataclass_fields__})

    def to_config(
        self,
        *,
        preset_name: str = "final_nn",
        training_dir: str | None = None,
        run_name: str | None = None,
        **overrides: Any,
    ) -> Config:
        """Build a Config from the YAML. Pass overrides as kwargs."""
        d = dict(self.raw)
        # Pop containers that aren't Config fields
        d.pop("paths", None)
        d.pop("threshold_sweep", None)
        d.pop("analysis", None)
        # YAML uses scale_min / scale_max; Config wants a tuple
        if "scale_min" in d or "scale_max" in d:
            d["scale_range"] = (
                float(d.pop("scale_min", 0.7)),
                float(d.pop("scale_max", 1.5)),
            )
        # Filter to fields the Config dataclass accepts
        allowed = {k: v for k, v in d.items() if k in Config.__dataclass_fields__}
        allowed["training_dir"] = training_dir or self.paths.get("training_data_dir", "")
        allowed["preset_name"]  = preset_name
        allowed["run_name"]     = run_name
        allowed["output_root"]  = self.paths.get("trained_models_dir",
                                                 str(FINAL_NN_ROOT / "data" / "trained_models"))
        allowed.update(overrides)
        return Config(**allowed)


def load_params(path: Path | str | None = None) -> ResolvedParams:
    return ResolvedParams.load(path)
