"""Inspect a microscopy file and print its pixel size in micrometres per pixel.

Supports
--------
- .nd2 via the `nd2` library (Nikon native, has voxel_size metadata)
- .tif / .tiff via tifffile (reads ImageJ + OME-TIFF resolution tags)

Usage
-----
    python detect_pixel_size.py path/to/image.nd2
    python detect_pixel_size.py path/to/image.tif

Returns nothing programmatically — prints `pixel_size_um: <float>` on stdout
so it can be captured by callers. Exits 0 on success, 2 if no calibration
metadata was found, 1 on file/format errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def detect_nd2(path: Path) -> tuple[float | None, dict]:
    """Returns (xy_um_per_px, metadata_dict). Uses the .x voxel dimension."""
    import nd2
    with nd2.ND2File(str(path)) as f:
        v = f.voxel_size()
        return float(v.x), {"x_um": float(v.x), "y_um": float(v.y), "z_um": float(v.z),
                            "shape": list(f.shape)}


def detect_tif(path: Path) -> tuple[float | None, dict]:
    """Read ImageJ-style or OME-TIFF resolution tags.

    TIFF tags 282 / 283 (XResolution / YResolution) are in pixels per unit;
    tag 296 (ResolutionUnit) tells us inch vs cm. ImageJ also encodes 'unit'
    and 'spacing' in tag 270 (ImageDescription). We honour ImageJ first when
    it's available, since most microscopy TIFFs we'd see came through Fiji.
    """
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        info: dict = {"shape": list(page.shape), "dtype": str(page.dtype)}

        # ImageJ-style metadata (preferred for biology TIFFs)
        ij = tf.imagej_metadata or {}
        if "unit" in ij and ij.get("unit") in ("micron", "um", "µm"):
            spacing = ij.get("spacing")
            if spacing:  # ImageJ "spacing" is z; "ScaleY"/"ScaleX" are xy
                info["imagej_unit"] = ij["unit"]
                info["imagej_spacing_z"] = float(spacing)
        # XResolution (pixels per unit) for xy
        tags = {t.name: t.value for t in page.tags}
        xres = tags.get("XResolution")
        yres = tags.get("YResolution")
        runit = tags.get("ResolutionUnit", 1)
        info["XResolution"] = str(xres)
        info["YResolution"] = str(yres)
        info["ResolutionUnit"] = int(runit) if runit else None

        # Convert XResolution (a rational num/den) into μm/px
        if xres is not None:
            num, den = xres if isinstance(xres, tuple) else (xres, 1)
            if num and den:
                pixels_per_unit = float(num) / float(den)
                if pixels_per_unit > 0:
                    # ResolutionUnit: 1=none, 2=inch (25400 μm), 3=cm (10000 μm)
                    unit_to_um = {2: 25400.0, 3: 10000.0}.get(int(runit) if runit else 0)
                    if unit_to_um:
                        return unit_to_um / pixels_per_unit, info

        return None, info


def detect(path: Path) -> tuple[float | None, dict]:
    suf = path.suffix.lower()
    if suf == ".nd2":
        return detect_nd2(path)
    if suf in (".tif", ".tiff"):
        return detect_tif(path)
    raise ValueError(f"unsupported extension: {suf}  (.nd2/.tif/.tiff only)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="Emit a JSON line instead of human-readable.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 1

    try:
        px, info = detect(args.path)
    except Exception as e:
        print(f"error reading {args.path.name}: {e!r}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"path": str(args.path), "pixel_size_um": px, "metadata": info}, indent=2))
    else:
        print(f"file           : {args.path.name}")
        for k, v in info.items():
            print(f"  {k:<14}: {v}")
        if px is None:
            print("pixel_size_um  : (none — no calibration metadata)")
            return 2
        print(f"pixel_size_um  : {px:.6f}")
    return 0 if px is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
