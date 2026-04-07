from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.image as mpimg
import dlab.hardware.drivers.SLM_driver._slm_py as slm_driver

# SLM-300 Santec
DEFAULT_SLM_SIZE: Tuple[int, int] = (1200, 1920)  # (rows, cols) = (height, width)
DEFAULT_CHIP_W = 15.36e-3
DEFAULT_CHIP_H = 9.6e-3
DEFAULT_PIXEL_SIZE = 8e-6
DEFAULT_BIT_DEPTH = 1023  # 10 bits


class SLMController:
    """Minimal controller for a Spatial Light Modulator."""

    def __init__(
        self,
        color: str,
        slm_size: Tuple[int, int] = DEFAULT_SLM_SIZE,
        chip_width: float = DEFAULT_CHIP_W,
        chip_height: float = DEFAULT_CHIP_H,
        pixel_size: float = DEFAULT_PIXEL_SIZE,
        bit_depth: int = DEFAULT_BIT_DEPTH,
    ):
        self.color = color
        self.slm_size = slm_size
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.pixel_size = pixel_size
        self.bit_depth = bit_depth

        # Hardware flatness correction. Loaded once, applied at publish().
        self.background_phase: Optional[np.ndarray] = None  # int32, raw values
        self.background_path: Optional[str] = None
        self.background_enabled: bool = False

        self.phase: Optional[np.ndarray] = None
        self.screen_num: Optional[int] = None

    # ---- background management -----------------------------------------------

    def load_background(self, filepath: str) -> None:
        """Load hardware flatness correction from disk. Stored raw, wrapped at publish()."""
        path = Path(filepath)
        if path.suffix == ".csv":
            try:
                bg = np.loadtxt(
                    path, delimiter=",", skiprows=1,
                    usecols=np.arange(self.slm_size[1]) + 1,
                )
            except Exception:
                bg = np.loadtxt(path, delimiter=",")
        else:
            bg = mpimg.imread(str(path))
            if bg.ndim == 3:
                bg = bg.sum(axis=2)

        if bg.shape != self.slm_size:
            raise ValueError(
                f"background shape {bg.shape} != slm_size {self.slm_size}"
            )

        self.background_phase = bg.astype(np.int32)
        self.background_path = str(path)
        self.background_enabled = True

    def clear_background(self) -> None:
        self.background_phase = None
        self.background_path = None
        self.background_enabled = False

    def set_background_enabled(self, enabled: bool) -> None:
        self.background_enabled = bool(enabled)

    # ---- publish -------------------------------------------------------------

    def _convert_phase(self, phase: np.ndarray) -> np.ndarray:
        """Wrap to [0, bit_depth] and cast to contiguous uint16."""
        arr = np.asarray(phase)
        arr = np.mod(arr, self.bit_depth + 1)
        arr = arr.astype(np.uint16, copy=False)
        return np.ascontiguousarray(arr)

    def publish(self, phase: np.ndarray, screen_num: int) -> None:
        """Publish logical phase + hardware background (if enabled) to the SLM."""
        if self.background_enabled and self.background_phase is not None:
            total = phase.astype(np.int32) + self.background_phase
        else:
            total = phase
        self.phase = self._convert_phase(total)
        self.screen_num = screen_num
        slm_driver.SLM_Disp_Open(self.screen_num)
        h, w = self.slm_size
        slm_driver.SLM_Disp_Data(self.screen_num, self.phase, w, h)

    def close(self) -> None:
        """Explicit close if you kept a screen open (safe no-op otherwise)."""
        if self.screen_num is not None:
            try:
                slm_driver.SLM_Disp_Close(self.screen_num)
            except Exception:
                pass
            self.screen_num = None