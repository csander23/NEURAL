"""Excel I/O helpers used by the manual side of the figure engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def long_path_safe(p) -> str:
    """Windows MAX_PATH workaround: prepend \\?\ prefix when path > 250 chars."""
    s = str(p)
    if not sys.platform.startswith("win"):
        return s
    if len(s) > 250 and not s.startswith("\\\\?\\"):
        if s.startswith("\\\\"):
            return "\\\\?\\UNC\\" + s.lstrip("\\")
        return "\\\\?\\" + s
    return s


def read_traces_sheet(xlsx_path: Path, sheet_name: str) -> np.ndarray:
    """Return (T, N_neurons) float64 array from the given xlsx sheet."""
    df = pd.read_excel(long_path_safe(xlsx_path), sheet_name=sheet_name)
    cols = [c for c in df.columns if str(c).strip().lower().startswith("neuron")]
    if not cols:
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    return df[cols].to_numpy(dtype=np.float64)
