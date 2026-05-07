from __future__ import annotations

import logging
import os

import numpy as np

_log = logging.getLogger(__name__)

_LUCAM_DLL_CANDIDATES = [
    r"C:\Program Files\Spiricon\BeamGage Standard\x64",
    r"C:\Program Files\Spiricon\BeamGage Professional\x64",
]
for path in _LUCAM_DLL_CANDIDATES:
    if os.path.isdir(path):
        os.add_dll_directory(path)
        break
else:
    _log.warning("No lucamapi.dll location found; import will likely fail.")

import lucam

DEFAULT_EXPOSURE_US = 20_000
DEFAULT_GAIN = 1
MIN_EXPOSURE_US = 50
MAX_EXPOSURE_US = 1_000_000
MIN_GAIN = 1
MAX_GAIN = 16
SNAPSHOT_TIMEOUT_MS = 5_000.0
DEFAULT_PIXEL_FORMAT = lucam.API.LUCAM_PF_16  # 16-bit ; fallback PF_8 si non supporté


class LumeneraControllerError(Exception):
    """Raised for Lumenera camera operation errors."""


class LumeneraController:
    """Controller for a Lumenera/Ophir SP402S camera via the lucam wrapper."""

    def __init__(
        self, index: int = 1, pixel_format: int = DEFAULT_PIXEL_FORMAT
    ) -> None:
        self.index = index
        self._requested_pf = pixel_format
        self._cam: lucam.Lucam | None = None
        self._fmt: lucam.Lucam.FrameFormat | None = None
        self._framerate: float | None = None
        self._imshape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self.current_exposure: int | None = None
        self.current_gain: int | None = None

    def activate(self) -> None:
        """Initialize and configure the camera."""
        try:
            n = lucam.LucamNumCameras()
            if n < self.index:
                raise LumeneraControllerError(
                    f"Camera index {self.index} not available (found {n})"
                )

            cam = lucam.Lucam(self.index)

            fmt, framerate = cam.GetFormat()
            # Force le pixel format demandé (16-bit par défaut)
            fmt.pixelFormat = self._requested_pf
            try:
                cam.SetFormat(fmt, framerate)
                fmt, framerate = cam.GetFormat()
            except lucam.LucamError as e:
                _log.warning(
                    "Lumenera[%s] pixel format %d not supported (%s); falling back to camera default",
                    self.index,
                    self._requested_pf,
                    e,
                )
                fmt, framerate = cam.GetFormat()

            cam.SetTimeout(True, SNAPSHOT_TIMEOUT_MS)
            cam.SetTimeout(False, SNAPSHOT_TIMEOUT_MS)

            cam.set_properties(
                exposure=self._clamp_exposure(DEFAULT_EXPOSURE_US) / 1000.0,
                gain=float(self._clamp_gain(DEFAULT_GAIN)),
                brightness=1.0,
                contrast=1.0,
                gamma=1.0,
            )

            self._cam = cam
            self._fmt = fmt
            self._framerate = framerate

            snap = lucam.Lucam.Snapshot(
                exposure=self._clamp_exposure(DEFAULT_EXPOSURE_US) / 1000.0,
                gain=float(self._clamp_gain(DEFAULT_GAIN)),
                timeout=SNAPSHOT_TIMEOUT_MS,
                format=fmt,
            )
            test_frame = cam.TakeSnapshot(snap)
            if test_frame is None:
                raise LumeneraControllerError("Failed to get image on activation.")
            self._imshape = tuple(np.shape(test_frame))
            self._dtype = test_frame.dtype

            self.current_exposure = self._clamp_exposure(DEFAULT_EXPOSURE_US)
            self.current_gain = self._clamp_gain(DEFAULT_GAIN)

            _log.info(
                "Lumenera[%s] activated; shape=%s; dtype=%s; pf=%d; fr=%.2f; "
                "exposure=%dus; gain=%d",
                self.index,
                self._imshape,
                self._dtype,
                fmt.pixelFormat,
                framerate,
                self.current_exposure,
                self.current_gain,
            )
        except Exception as e:
            self._safe_close()
            raise LumeneraControllerError(f"activate failed: {e}") from e

    def deactivate(self) -> None:
        """Close the camera connection."""
        self._safe_close()
        _log.info("Lumenera[%s] deactivated", self.index)

    def _safe_close(self) -> None:
        try:
            if self._cam is not None:
                self._cam.CameraClose()
        except Exception as e:
            _log.warning("Lumenera[%s] close error: %s", self.index, e)
        finally:
            self._cam = None
            self._fmt = None
            self._framerate = None
            self._imshape = None
            self._dtype = None
            self.current_exposure = None
            self.current_gain = None

    def _clamp_exposure(self, us: int) -> int:
        if us < MIN_EXPOSURE_US or us > MAX_EXPOSURE_US:
            clamped = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, us))
            _log.warning(
                "Lumenera[%s] exposure %dus out of range [%d..%d]; clamped to %dus",
                self.index,
                us,
                MIN_EXPOSURE_US,
                MAX_EXPOSURE_US,
                clamped,
            )
            return clamped
        return us

    def _clamp_gain(self, g: int) -> int:
        if g < MIN_GAIN or g > MAX_GAIN:
            clamped = max(MIN_GAIN, min(MAX_GAIN, g))
            _log.warning(
                "Lumenera[%s] gain %d out of range [%d..%d]; clamped to %d",
                self.index,
                g,
                MIN_GAIN,
                MAX_GAIN,
                clamped,
            )
            return clamped
        return g

    def set_exposure(self, exposure_us: int) -> None:
        """Set exposure time in microseconds."""
        if not isinstance(exposure_us, int) or exposure_us <= 0:
            raise ValueError("exposure_us must be a positive integer")
        if self._cam is None:
            raise LumeneraControllerError("Camera not active; call activate() first")

        exposure_us = self._clamp_exposure(exposure_us)
        if self.current_exposure == exposure_us:
            return
        self._cam.set_properties(exposure=exposure_us / 1000.0)
        self.current_exposure = exposure_us
        _log.debug("Lumenera[%s] exposure set to %dus", self.index, exposure_us)

    def set_gain(self, gain: int) -> None:
        """Set device gain (multiplicative, 1 = unity)."""
        if not isinstance(gain, int):
            raise ValueError("gain must be an integer")
        if self._cam is None:
            raise LumeneraControllerError("Camera not active; call activate() first")

        gain = self._clamp_gain(gain)
        if self.current_gain == gain:
            return
        self._cam.set_properties(gain=float(gain))
        self.current_gain = gain
        _log.debug("Lumenera[%s] gain set to %d", self.index, gain)

    def get_image_shape(self) -> tuple[int, ...]:
        """Return the image dimensions."""
        if self._imshape is None:
            raise LumeneraControllerError("Image shape unknown; call activate() first")
        return self._imshape

    def capture_single(self, exposure_us: int, gain: int | None = None) -> np.ndarray:
        """Capture a single frame with given exposure and optional gain.
        Returns the raw frame as-is (uint16 in 16-bit mode, uint8 in 8-bit fallback)."""
        if self._cam is None or self._fmt is None:
            raise LumeneraControllerError("Camera not active; call activate() first")

        self.set_exposure(int(exposure_us))
        if gain is not None:
            self.set_gain(int(gain))

        snap = lucam.Lucam.Snapshot(
            exposure=self.current_exposure / 1000.0,
            gain=float(self.current_gain),
            timeout=SNAPSHOT_TIMEOUT_MS,
            format=self._fmt,
        )
        arr = self._cam.TakeSnapshot(snap)
        if arr is None:
            raise LumeneraControllerError("capture_single: image array is None")
        return arr

    @staticmethod
    def get_available_indices() -> list[int]:
        """Return list of available camera indices (1-based, matching lucam convention)."""
        n = lucam.LucamNumCameras()
        return list(range(1, n + 1))
