"""DICOM loading and preprocessing shared by training and inference.

Every series is turned into a fixed-size uint8 stack of shape
(n_slices, IMAGE_SIZE, IMAGE_SIZE). Training and inference must use exactly
this code, otherwise the model sees different inputs at serving time.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from .constants import IMAGE_SIZE, SERIES_ROLES, SLICES_PER_SERIES


def _slice_position(ds) -> float:
    """Position of a slice along its stack normal (falls back to InstanceNumber)."""
    try:
        orient = np.array(ds.ImageOrientationPatient, dtype=float)
        pos = np.array(ds.ImagePositionPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        return float(np.dot(pos, normal))
    except (AttributeError, ValueError, TypeError):
        return float(getattr(ds, "InstanceNumber", 0) or 0)


def sort_series_files(paths: list[Path]) -> list[Path]:
    headers = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        headers.append((_slice_position(ds), p))
    headers.sort(key=lambda t: t[0])
    return [p for _, p in headers]


def _normalize_to_uint8(stack: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(stack, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    stack = np.clip((stack - lo) / (hi - lo), 0.0, 1.0)
    return (stack * 255).astype(np.uint8)


def load_series_stack(paths: list[Path], n_slices: int) -> np.ndarray:
    """Read a series, pick n evenly spaced slices, resize and normalize."""
    if not paths:
        return np.zeros((n_slices, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    ordered = sort_series_files(paths)
    if not ordered:
        return np.zeros((n_slices, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    idx = np.linspace(0, len(ordered) - 1, n_slices).round().astype(int)
    slices = []
    for i in idx:
        arr = pydicom.dcmread(str(ordered[i])).pixel_array.astype(np.float32)
        if arr.ndim == 3:  # multi-frame / RGB edge cases: take first plane
            arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
        img = Image.fromarray(arr).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        slices.append(np.asarray(img, dtype=np.float32))
    return _normalize_to_uint8(np.stack(slices))


def load_study_stacks(series_paths: dict[str, list[Path]]) -> dict[str, np.ndarray]:
    """series_paths maps role -> list of dicom files (missing roles become zeros)."""
    return {role: load_series_stack(series_paths.get(role, []), SLICES_PER_SERIES[role]) for role in SERIES_ROLES}


# --- Series discovery for uploaded studies -------------------------------------------------------


def group_by_series(root: Path) -> dict[str, dict]:
    """Group every DICOM under root by SeriesInstanceUID and collect header hints."""
    groups: dict[str, dict] = defaultdict(lambda: {"files": [], "header": None})
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or p.parent.name)
        g = groups[uid]
        g["files"].append(p)
        if g["header"] is None:
            g["header"] = ds
    return dict(groups)


def guess_series_role(ds) -> str | None:
    """Best-effort guess of sag_t1 / sag_t2 / ax_t2 from DICOM headers."""
    desc = str(getattr(ds, "SeriesDescription", "") or "").lower()
    plane = None
    try:
        orient = np.array(ds.ImageOrientationPatient, dtype=float)
        axis = int(np.argmax(np.abs(np.cross(orient[:3], orient[3:]))))
        plane = {0: "sag", 1: "cor", 2: "ax"}[axis]
    except (AttributeError, ValueError, TypeError):
        if "sag" in desc:
            plane = "sag"
        elif "ax" in desc or "tra" in desc:
            plane = "ax"

    weighting = None
    if "stir" in desc or "t2" in desc:
        weighting = "t2"
    elif "t1" in desc:
        weighting = "t1"
    else:
        te = float(getattr(ds, "EchoTime", 0) or 0)
        tr = float(getattr(ds, "RepetitionTime", 0) or 0)
        if te >= 60:
            weighting = "t2"
        elif 0 < te < 30 and 0 < tr < 1000:
            weighting = "t1"

    if plane == "sag" and weighting in ("t1", "t2"):
        return f"sag_{weighting}"
    if plane == "ax" and weighting != "t1":
        return "ax_t2"
    return None


def describe_series(groups: dict[str, dict]) -> list[dict]:
    out = []
    for uid, g in groups.items():
        ds = g["header"]
        out.append(
            {
                "series_uid": uid,
                "description": str(getattr(ds, "SeriesDescription", "") or ""),
                "n_files": len(g["files"]),
                "guessed_role": guess_series_role(ds),
            }
        )
    out.sort(key=lambda s: (s["guessed_role"] is None, s["guessed_role"] or "", -s["n_files"]))
    return out
