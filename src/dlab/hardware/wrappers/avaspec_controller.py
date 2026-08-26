from __future__ import annotations

import time
import threading

import numpy as np

import dlab.hardware.drivers.avaspec_driver._avs_py as avs
import dlab.hardware.drivers.avaspec_driver._avs_win as avs_win
from dlab.utils.config_utils import cfg_get
from dlab.utils.paths_utils import ressources_dir


class AvaspecError(Exception):
    """Raised for Avaspec spectrometer operation errors."""


def _avaspec_calibration_path():
    """Get calibration file path from config."""
    rel = cfg_get("avaspec.calibration_file_1030nm")
    if not rel:
        return None
    p = (ressources_dir() / str(rel)).resolve()
    return p if p.exists() else None


def _is_pending(exc: Exception) -> bool:
    return "ERR_OPERATION_PENDING" in str(exc)


class AvaspecController:
    """Controller for Avantes spectrometer via AVS driver."""

    def __init__(self, spec_handle):
        self._orig_handle = spec_handle
        self._h = None
        self._wl: np.ndarray | None = None
        self._npx: int | None = None
        self._bg_wl: np.ndarray | None = None
        self._bg_counts: np.ndarray | None = None
        self._cal_wl: np.ndarray | None = None
        self._cal_vals: np.ndarray | None = None
        self._cal_enabled = False
        self._int_ms = 100.0
        self._avg = 1
        self._io_lock = threading.Lock()

    def activate(self) -> None:
        """Initialize and activate the spectrometer."""
        try:
            avs.AVS_Init()
            self._h = avs.AVS_Activate(self._orig_handle)
            if self._h is None:
                raise AvaspecError("AVS_Activate returned None")
            try:
                avs_win.AVS_UseHighResAdc(self._h, True)
            except Exception:
                pass
            wl = np.asarray(avs.AVS_GetLambda(self._h), dtype=float)
            if wl.size == 0:
                raise AvaspecError("Empty wavelength array")
            self._wl = wl
            self._npx = int(wl.size)
        except Exception as e:
            self._h = None
            raise AvaspecError(f"Activate failed: {e}") from e

    def deactivate(self) -> None:
        """Deactivate and close the spectrometer."""
        with self._io_lock:
            try:
                if self._h is not None:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.AVS_Deactivate(self._h)
            finally:
                self._h = None

    @classmethod
    def list_spectrometers(cls) -> list:
        """Return list of available spectrometers."""
        try:
            avs.AVS_Init()
            return avs.AVS_GetList() or []
        except Exception:
            return []

    def set_params(self, int_time_ms: float, averages: int) -> None:
        """Set integration time and number of averages."""
        if self._h is None:
            raise AvaspecError("Not activated")
        int_time_ms = float(max(1.0, float(int_time_ms)))
        averages = int(max(1, int(averages)))
        with self._io_lock:
            tries = 0
            while True:
                try:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.set_measure_params(self._h, int_time_ms, averages)
                    self._int_ms = int_time_ms
                    self._avg = averages
                    return
                except Exception as e:
                    if _is_pending(e) and tries < 100:
                        tries += 1
                        time.sleep(0.01)
                        continue
                    raise

    def set_background(self, wl_nm: np.ndarray, counts: np.ndarray) -> None:
        """Set background spectrum for subtraction."""
        self._bg_wl = np.asarray(wl_nm, float).ravel()
        self._bg_counts = np.asarray(counts, float).ravel()

    def clear_background(self) -> None:
        """Clear background spectrum."""
        self._bg_wl = None
        self._bg_counts = None

    def enable_calibration(self, enabled: bool) -> None:
        """Enable or disable calibration correction."""
        self._cal_enabled = bool(enabled)
        if enabled and (self._cal_wl is None or self._cal_vals is None):
            p = _avaspec_calibration_path()
            if p:
                self._load_calibration_file(p)

    def _load_calibration_file(self, path) -> None:
        arr = np.loadtxt(str(path), dtype=float, delimiter=",")
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise AvaspecError("Calibration file must have 2 columns: wavelength,value")
        wl = np.asarray(arr[:, 0], float)
        vals = np.asarray(arr[:, 1], float)
        eps = np.finfo(float).tiny
        vals = np.where(np.isfinite(vals) & (np.abs(vals) > 0), vals, eps)
        self._cal_wl = wl
        self._cal_vals = vals

    def _apply_background(self, wl_nm: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if self._bg_wl is None or self._bg_counts is None:
            return counts
        bg = np.interp(wl_nm, self._bg_wl, self._bg_counts, left=np.nan, right=np.nan)
        self._fill_nan_edges(bg)
        return counts - bg

    def _apply_calibration(self, wl_nm: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if not self._cal_enabled or self._cal_wl is None or self._cal_vals is None:
            return counts
        cal = np.interp(wl_nm, self._cal_wl, self._cal_vals, left=np.nan, right=np.nan)
        self._fill_nan_edges(cal)
        eps = np.finfo(float).tiny
        cal = np.where(np.isfinite(cal) & (np.abs(cal) > 0), cal, eps)
        return counts / cal

    @staticmethod
    def _fill_nan_edges(arr: np.ndarray) -> None:
        """Fill NaN values at edges with nearest valid value."""
        if np.isnan(arr[0]):
            j = np.flatnonzero(~np.isnan(arr))
            if j.size:
                arr[:j[0]] = arr[j[0]]
        if np.isnan(arr[-1]):
            j = np.flatnonzero(~np.isnan(arr))
            if j.size:
                arr[j[-1]:] = arr[j[-1]]

    def measure_once(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Perform a single measurement and return (timestamp, wavelengths, counts)."""
        if self._h is None:
            raise AvaspecError("Not activated")
        with self._io_lock:
            tries = 0
            while True:
                try:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.set_measure_params(self._h, self._int_ms, self._avg)
                    avs.AVS_Measure(self._h)
                    wait_s = max(0.0, 0.001 * max(1, self._avg) + 0.001)
                    time.sleep(wait_s)
                    ts, data = avs.get_spectrum(self._h)
                    counts = np.asarray(data, float).ravel()
                    if self._npx and counts.size != self._npx:
                        n = self._npx
                        out = np.zeros(n, float)
                        out[:min(n, counts.size)] = counts[:min(n, counts.size)]
                        counts = out
                    wl = np.asarray(self._wl, float).ravel()
                    return float(ts), wl, counts
                except Exception as e:
                    if _is_pending(e) and tries < 100:
                        tries += 1
                        time.sleep(0.01)
                        continue
                    raise

    def process_counts(self, wl_nm: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Apply background subtraction and calibration to counts."""
        y = self._apply_background(wl_nm, counts)
        y = self._apply_calibration(wl_nm, y)
        return y