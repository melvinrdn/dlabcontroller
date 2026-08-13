from __future__ import annotations

import logging

import numpy as np
from pylablib.devices import Andor
import pylablib

from dlab.boot import ROOT
from dlab.utils.config_utils import cfg_get


_log = logging.getLogger(__name__)

DEFAULT_EXPOSURE_US = 500_000
MIN_EXPOSURE_US = 1_000
MAX_EXPOSURE_US = 10_000_000


class AndorControllerError(Exception):
    """Raised for Andor camera operation errors."""


class AndorController:
    """Controller for an Andor SDK2 camera via pylablib."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cam: Andor.AndorSDK2Camera | None = None
        self._image_shape: tuple[int, ...] | None = None
        self._current_exposure: int | None = None

    def is_active(self) -> bool:
        """Check if camera is activated."""
        return self._cam is not None

    def activate(self) -> None:
        """Initialize and configure the camera."""
        if self._cam is not None:
            return

        try:
            driver_path = str(cfg_get("paths.drivers_andor", "src/dlab/hardware/drivers/andor_driver"))
            pylablib.par["devices/dlls/andor_sdk2"] = str((ROOT / driver_path).resolve())

            cam = Andor.AndorSDK2Camera()
            exp_us = self._clamp_exposure(DEFAULT_EXPOSURE_US)
            cam.set_exposure(exp_us / 1e6)

            try:
                cam.setup_shutter("open")
            except Exception:
                pass

            cam.start_acquisition()
            try:
                cam.wait_for_frame(timeout=20)
                frame = cam.read_oldest_image()
            finally:
                cam.stop_acquisition()

            self._cam = cam
            self._image_shape = np.shape(frame)
            self._current_exposure = exp_us

            _log.info(
                "Andor[%s] activated; shape=%s; exposure=%dus",
                self.device_index, self._image_shape, self._current_exposure
            )
        except Exception as e:
            try:
                cam.close()
            except Exception:
                pass
            self._cam = None
            self._image_shape = None
            raise AndorControllerError(f"activate failed: {e}") from e

    def deactivate(self) -> None:
        """Close the camera connection."""
        if self._cam:
            try:
                self._cam.close()
                _log.info("Andor[%s] deactivated", self.device_index)
            except Exception as e:
                _log.error("Andor[%s] deactivate error: %s", self.device_index, e)
            finally:
                self._cam = None
                self._image_shape = None
                self._current_exposure = None

    def __enter__(self) -> AndorController:
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.deactivate()

    def _clamp_exposure(self, exposure_us: int) -> int:
        if exposure_us < MIN_EXPOSURE_US or exposure_us > MAX_EXPOSURE_US:
            clamped = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, exposure_us))
            _log.warning(
                "Andor[%s] exposure %dus out of range [%d..%d]; clamped to %dus",
                self.device_index, exposure_us, MIN_EXPOSURE_US, MAX_EXPOSURE_US, clamped
            )
            return clamped
        return exposure_us

    def set_exposure(self, exposure_us: int) -> None:
        """Set exposure time in microseconds."""
        if not isinstance(exposure_us, int) or exposure_us <= 0:
            raise ValueError("exposure_us must be a positive integer")
        if self._cam is None:
            raise AndorControllerError("Camera not active; call activate() first")

        exposure_us = self._clamp_exposure(exposure_us)
        if self._current_exposure == exposure_us:
            return

        try:
            self._cam.set_exposure(exposure_us / 1e6)
            self._current_exposure = exposure_us
            _log.debug("Andor[%s] exposure set to %dus", self.device_index, exposure_us)
        except Exception as e:
            raise AndorControllerError(f"set_exposure failed: {e}") from e

    def get_image_shape(self) -> tuple[int, ...]:
        """Return the image dimensions."""
        if self._image_shape is None:
            raise AndorControllerError("Image shape unknown; call activate() first")
        return self._image_shape

    @property
    def current_exposure(self) -> int | None:
        """Current exposure time in microseconds."""
        return self._current_exposure

    def capture_single(self, exposure_us: int | None = None, timeout_s: float = 20.0) -> np.ndarray:
        """Capture a single frame with optional exposure override."""
        if self._cam is None or self._image_shape is None:
            raise AndorControllerError("Camera not active; call activate() first")

        if exposure_us is not None:
            self.set_exposure(int(exposure_us))

        self._cam.start_acquisition()
        try:
            self._cam.wait_for_frame(timeout=timeout_s)
            frame = self._cam.read_oldest_image()
        finally:
            self._cam.stop_acquisition()

        return frame.astype(np.float64, copy=False)