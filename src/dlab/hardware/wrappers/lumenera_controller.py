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
DEFAULT_PIXEL_FORMAT = lucam.API.LUCAM_PF_16
ROI_ALIGNMENT = 8  # camera requires width/height/offsets to be multiples of this


class LumeneraControllerError(Exception):
    """Raised for Lumenera camera operation errors."""


class LumeneraController:
    """Controller for a Lumenera/Ophir SP402S camera via the lucam wrapper.
    
    Operates at full sensor binning (1x1). ROI is exposed in sensor pixel
    coordinates. ROI dimensions are auto-aligned to ROI_ALIGNMENT.
    """

    def __init__(
        self, index: int = 1, pixel_format: int = DEFAULT_PIXEL_FORMAT
    ) -> None:
        self.index = index
        self._requested_pf = pixel_format
        self._cam: lucam.Lucam | None = None
        self._fmt: lucam.Lucam.FrameFormat | None = None
        self._framerate: float | None = None
        self._sensor_w: int | None = None
        self._sensor_h: int | None = None
        self._imshape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self.current_exposure: int | None = None
        self.current_gain: int | None = None
        self.current_roi: tuple[int, int, int, int] | None = None  # (x, y, w, h)

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
            self._sensor_w = fmt.width
            self._sensor_h = fmt.height

            fmt.pixelFormat = self._requested_pf
            try:
                cam.SetFormat(fmt, framerate)
                fmt, framerate = cam.GetFormat()
            except lucam.LucamError as e:
                _log.warning(
                    "Lumenera[%s] pixel format %d not supported (%s); falling back to camera default",
                    self.index, self._requested_pf, e,
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
            self.current_exposure = self._clamp_exposure(DEFAULT_EXPOSURE_US)
            self.current_gain = self._clamp_gain(DEFAULT_GAIN)
            self.current_roi = (0, 0, fmt.width, fmt.height)

            self._refresh_shape()

            _log.info(
                "Lumenera[%s] activated; shape=%s; dtype=%s; pf=%d; fr=%.2f; "
                "exposure=%dus; gain=%d",
                self.index, self._imshape, self._dtype, fmt.pixelFormat,
                framerate, self.current_exposure, self.current_gain,
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
            self._sensor_w = None
            self._sensor_h = None
            self._imshape = None
            self._dtype = None
            self.current_exposure = None
            self.current_gain = None
            self.current_roi = None

    def _clamp_exposure(self, us: int) -> int:
        if us < MIN_EXPOSURE_US or us > MAX_EXPOSURE_US:
            clamped = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, us))
            _log.warning(
                "Lumenera[%s] exposure %dus out of range [%d..%d]; clamped to %dus",
                self.index, us, MIN_EXPOSURE_US, MAX_EXPOSURE_US, clamped,
            )
            return clamped
        return us

    def _clamp_gain(self, g: int) -> int:
        if g < MIN_GAIN or g > MAX_GAIN:
            clamped = max(MIN_GAIN, min(MAX_GAIN, g))
            _log.warning(
                "Lumenera[%s] gain %d out of range [%d..%d]; clamped to %d",
                self.index, g, MIN_GAIN, MAX_GAIN, clamped,
            )
            return clamped
        return g

    def _refresh_shape(self) -> None:
        """Take a probe snapshot to update _imshape and _dtype after format changes."""
        if self._cam is None or self._fmt is None:
            return
        snap = lucam.Lucam.Snapshot(
            exposure=self.current_exposure / 1000.0,
            gain=float(self.current_gain),
            timeout=SNAPSHOT_TIMEOUT_MS,
            format=self._fmt,
        )
        frame = self._cam.TakeSnapshot(snap)
        if frame is None:
            raise LumeneraControllerError("Probe snapshot returned None")
        self._imshape = tuple(np.shape(frame))
        self._dtype = frame.dtype

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

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set hardware ROI in sensor pixel coordinates.
        
        Values are auto-aligned to ROI_ALIGNMENT (typically 8 px) — the actual
        applied ROI may differ slightly. Read self.current_roi after the call
        to get the actual value.
        """
        if self._cam is None or self._fmt is None or self._sensor_w is None:
            raise LumeneraControllerError("Camera not active; call activate() first")

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI offsets must be >= 0 and dimensions > 0")
        if x + width > self._sensor_w or y + height > self._sensor_h:
            raise ValueError(
                f"ROI ({x},{y},{width},{height}) exceeds sensor "
                f"{self._sensor_w}x{self._sensor_h}"
            )

        # Auto-align everything to ROI_ALIGNMENT
        ax = (x // ROI_ALIGNMENT) * ROI_ALIGNMENT
        ay = (y // ROI_ALIGNMENT) * ROI_ALIGNMENT
        aw = (width // ROI_ALIGNMENT) * ROI_ALIGNMENT
        ah = (height // ROI_ALIGNMENT) * ROI_ALIGNMENT
        if aw == 0 or ah == 0:
            raise ValueError(f"ROI too small after alignment to {ROI_ALIGNMENT} px")

        new_fmt = lucam.Lucam.FrameFormat(
            ax, ay, aw, ah,
            self._fmt.pixelFormat,
            binningX=1, flagsX=1,
            binningY=1, flagsY=1,
        )
        try:
            rates = self._cam.EnumAvailableFrameRates()
            target_fr = max(rates) if rates else self._framerate
        except Exception:
            target_fr = self._framerate

        self._cam.SetFormat(new_fmt, target_fr)
        applied_fmt, applied_fr = self._cam.GetFormat()
        self._fmt = applied_fmt
        self._framerate = applied_fr
        self._refresh_shape()

        self.current_roi = (
            applied_fmt.xOffset, applied_fmt.yOffset,
            applied_fmt.width, applied_fmt.height,
        )
        _log.info(
            "Lumenera[%s] ROI=%s; shape=%s; framerate=%.2f",
            self.index, self.current_roi, self._imshape, self._framerate,
        )

    def reset_roi(self) -> None:
        """Reset ROI to full sensor."""
        if self._sensor_w is None or self._sensor_h is None:
            raise LumeneraControllerError("Camera not active; call activate() first")
        self.set_roi(0, 0, self._sensor_w, self._sensor_h)

    def get_image_shape(self) -> tuple[int, ...]:
        """Return the image dimensions."""
        if self._imshape is None:
            raise LumeneraControllerError("Image shape unknown; call activate() first")
        return self._imshape

    def get_sensor_size(self) -> tuple[int, int]:
        """Return (width, height) of the full sensor in pixels."""
        if self._sensor_w is None or self._sensor_h is None:
            raise LumeneraControllerError("Camera not active; call activate() first")
        return self._sensor_w, self._sensor_h

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