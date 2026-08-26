from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
from PyQt5.QtWidgets import QMessageBox, QWidget

LARGE_SCAN_WARNING_THRESHOLD = 100_000


def generate_positions(start: float, end: float, step: float) -> list[float]:
    """Generate an inclusive list of positions from start to end (either direction)."""
    if step <= 0:
        raise ValueError("Step must be > 0.")
    if end >= start:
        n = int(np.floor((end - start) / step))
        vals = [start + i * step for i in range(n + 1)]
        if vals[-1] < end - 1e-12:
            vals.append(end)
    else:
        n = int(np.floor((start - end) / step))
        vals = [start - i * step for i in range(n + 1)]
        if vals[-1] > end + 1e-12:
            vals.append(end)
    return vals


def save_png_with_meta(folder: Path, filename: str, frame_u16: np.ndarray, meta: dict) -> Path:
    """Save a 16-bit PNG image with metadata. Requires uint16 input."""
    if frame_u16.dtype != np.uint16:
        raise TypeError(
            f"save_png_with_meta requires uint16, got {frame_u16.dtype}. "
            "Cast explicitly before calling."
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    img = Image.fromarray(frame_u16, mode="I;16")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path


def save_png_with_meta_8bit(folder: Path, filename: str, frame_u8: np.ndarray, meta: dict) -> Path:
    """Save an 8-bit grayscale PNG image with metadata. Requires uint8 input."""
    if frame_u8.dtype != np.uint8:
        raise TypeError(
            f"save_png_with_meta_8bit requires uint8, got {frame_u8.dtype}. "
            "Cast explicitly before calling."
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    img = Image.fromarray(frame_u8, mode="L")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path


def confirm_large_scan(
    parent: QWidget, total_points: int, n_detectors: int,
    threshold: int = LARGE_SCAN_WARNING_THRESHOLD,
) -> bool:
    """Warn before launching a scan whose point count looks like a parameter typo.

    Returns True if the scan should proceed (below threshold, or user confirmed).
    """
    total = total_points * max(1, n_detectors)
    if total <= threshold:
        return True
    reply = QMessageBox.question(
        parent, "Large scan",
        f"This scan will acquire {total:,} images "
        f"({total_points:,} points x {n_detectors} detector(s)).\n\n"
        "This looks unusually large — check your start/end/step values.\n\n"
        "Continue anyway?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
    )
    return reply == QMessageBox.Yes
