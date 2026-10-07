from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QDoubleSpinBox,
    QProgressBar,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QLineEdit,
    QCheckBox,
)

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.phase_settings import PhaseSettings
from dlab.utils.log_panel import LogPanel
from dlab.utils.paths_utils import data_dir, cfg_get
from dlab.diagnostics.ui.scans.scan_utils import (
    confirm_large_scan,
    generate_positions,
    save_png_with_meta,
    save_png_with_meta_8bit,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

NUM_WAVEPLATES = int(cfg_get("waveplates.num_waveplates", 7))
POWER_MODE_SYNC_INTERVAL_MS = 400
SPECTRUM_MEASUREMENT_DELAY_S = 0.01
# Virtual axis: global rotation [deg] of every rotatable phase of the red SLM
# (SlmWindow rotation field), e.g. for XUV far-field tomography.
SLM_ROTATION_AXIS = "slm:Rotation"


# -----------------------------------------------------------------------------
# Helper functions - Waveplate calibration
# -----------------------------------------------------------------------------


def power_to_angle(power_fraction: float, phase_deg: float) -> float:
    """Convert power fraction (0-1) to waveplate angle using calibration phase."""
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0


def angle_to_power(angle_deg: float, phase_deg: float) -> float:
    """Convert waveplate angle to power fraction using calibration phase."""
    y = 0.5 * (1.0 + float(np.cos(2.0 * np.pi / 90.0 * (float(angle_deg) - float(phase_deg)))))
    return float(np.clip(y, 0.0, 1.0))


def _wp_index_from_stage_key(stage_key: str) -> int | None:
    """Extract waveplate index from stage key like 'stage:3'."""
    try:
        if not stage_key.startswith("stage:"):
            return None
        n = int(stage_key.split(":")[1])
        if 1 <= n <= NUM_WAVEPLATES:
            return n
    except (ValueError, IndexError):
        pass
    return None


# -----------------------------------------------------------------------------
# Helper functions - Registry keys
# -----------------------------------------------------------------------------


def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"


def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"


def _reg_key_calib_path(wp_index: int) -> str:
    return f"waveplate:calib_path:{wp_index}"


def _reg_key_maxvalue(wp_index: int) -> str:
    return f"waveplate:max_value:{wp_index}"


def _detector_display_name(det_key: str, dev, meta: dict | None) -> str:
    """Get a human-readable name for a detector."""
    if meta and str(meta.get("DeviceName", "")).strip():
        return str(meta["DeviceName"]).strip()
    for attr in ("name", "camera_name", "model_name"):
        v = getattr(dev, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    suffix = det_key.split(":")[-1]
    base, *rest = suffix.split("_")
    if base.lower().endswith("cam"):
        vendor = base[:-3]
        camel = (vendor[:1].upper() + vendor[1:]) + "Cam"
    else:
        camel = base[:1].upper() + base[1:]
    return camel + (("_" + "_".join(rest)) if rest else "")


# -----------------------------------------------------------------------------
# Helper functions - Scan time estimate
# -----------------------------------------------------------------------------


def _detector_time_estimate_s(det_key: str, params: tuple) -> float:
    """Lower-bound capture time for one detector acquisition, mirroring the
    worker's _capture_* methods. Ignores per-frame readout/transfer overhead."""
    if det_key.startswith("powermeter:"):
        period_ms = float(params[0]) if len(params) >= 1 else 100.0
        averages = int(params[1]) if len(params) >= 2 else 1
        return max(0, averages - 1) * period_ms / 1000.0
    if det_key.startswith("spectrometer:"):
        int_ms = float(params[0]) if len(params) >= 1 else 0.0
        averages = int(params[1]) if len(params) >= 2 else 1
        return (int_ms / 1000.0 + SPECTRUM_MEASUREMENT_DELAY_S) * averages
    exposure_us = float(params[0]) if len(params) >= 1 else 0.0
    averages = int(params[1]) if len(params) >= 2 else 1
    return (exposure_us / 1e6) * averages


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------


class GridScanWorker(QObject):
    """Worker for multi-axis grid scan.

    By default the scan visits the Cartesian product of the per-axis position
    lists in ``axes``. Pass ``points`` to visit an explicit list of positions
    instead: each point is a list of values, one per axis in ``axes`` order.
    For waveplate axes with ``axes_meta[ax]["pm"]`` set, explicit values are
    power fractions in [0, 1], converted to angles with the waveplate's
    calibration when the scan starts.
    ``point_extras`` optionally gives one dict per point whose values are
    appended as extra log columns (keys of the first dict name the columns).
    ``header_notes`` are extra lines written as comments in the log header.
    """

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        axes: list[tuple[str, list[float]]],
        camera_params: dict[str, tuple],
        settle_s: float,
        scan_name: str,
        comment: str,
        mcp_voltage: str,
        background: bool = False,
        existing_scan_log: str | None = None,
        axes_meta: dict | None = None,
        single_shot: bool = False,
        points: list[list[float]] | None = None,
        point_extras: list[dict] | None = None,
        header_notes: list[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if points is not None:
            n_axes = len(axes)
            for i, pt in enumerate(points):
                if len(pt) != n_axes:
                    raise ValueError(f"Point {i} has {len(pt)} values, expected {n_axes} (one per axis).")
        if point_extras is not None:
            if points is None:
                raise ValueError("point_extras requires an explicit points list.")
            if len(point_extras) != len(points):
                raise ValueError("point_extras must have one entry per point.")
        self.points_input = points  # as given (power fractions), kept for the log header
        self.points = points  # move targets; fractions become angles in _resolve_explicit_points
        self.point_extras = point_extras
        self.extra_columns = list(point_extras[0].keys()) if point_extras else []
        self.header_notes = list(header_notes or [])
        self.axes = axes
        self.camera_params = camera_params
        self.settle_s = float(settle_s)
        self.scan_name = scan_name
        self.comment = comment
        self.mcp_voltage = mcp_voltage
        self.background = bool(background)
        self.existing_scan_log = existing_scan_log
        self.abort = False
        self.axes_meta = axes_meta or {}
        self.single_shot = bool(single_shot)
        self.data_root = data_dir()
        self.timestamp = datetime.datetime.now()

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)

    # -------------------------------------------------------------------------
    # Cartesian product iteration
    # -------------------------------------------------------------------------

    def _cartesian_indices(self):
        """Yield all index combinations for the grid scan.

        In single_shot mode, yield only the last point (current axis positions).
        """
        lengths = [len(pos) for _, pos in self.axes]

        if self.single_shot:
            # Only the final point of each axis — this matches where the stages
            # were left at the end of the previous scan.
            yield [L - 1 for L in lengths]
            return

        def rec(level, idxs):
            if level == len(lengths):
                yield list(idxs)
                return
            for i in range(lengths[level]):
                idxs.append(i)
                yield from rec(level + 1, idxs)
                idxs.pop()

        yield from rec(0, [])

    def _total_points(self) -> int:
        if self.single_shot:
            return 1
        if self.points is not None:
            return len(self.points)
        total = 1
        for _, pos in self.axes:
            total *= max(1, len(pos))
        return total

    def _resolve_explicit_points(self) -> None:
        """Convert power-fraction values on power-mode axes to waveplate angles.

        Runs once before any move so a missing calibration or an out-of-range
        fraction aborts the scan before the stages move.
        """
        if self.points is None:
            return
        converters = {}
        for k, (ax, _) in enumerate(self.axes):
            wp = _wp_index_from_stage_key(ax)
            if wp is None or not self.axes_meta.get(ax, {}).get("pm", False):
                continue
            amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
            if amp_off[1] is None:
                raise ValueError(f"{ax}: Power mode ON but no calibration.")
            converters[k] = float(amp_off[1])
        if not converters:
            return
        resolved = []
        for i, pt in enumerate(self.points):
            new_pt = list(pt)
            for k, phase in converters.items():
                frac = float(pt[k])
                if not 0.0 <= frac <= 1.0:
                    raise ValueError(f"Point {i}: {self.axes[k][0]} power fraction {frac} outside [0, 1].")
                new_pt[k] = power_to_angle(frac, phase)
            resolved.append(new_pt)
        self.points = resolved

    def _iter_points(self):
        """Yield (point_index, ui_combo) for every point to visit.

        point_index indexes ``point_extras`` in explicit mode, else None.
        """
        if self.points is not None:
            if not self.points:
                return
            # Single-shot re-measures the last point, where the stages were left.
            indices = [len(self.points) - 1] if self.single_shot else range(len(self.points))
            for i in indices:
                yield i, [(self.axes[k][0], float(v)) for k, v in enumerate(self.points[i])]
            return

        for idxs in self._cartesian_indices():
            yield None, [(self.axes[k][0], self.axes[k][1][idxs[k]]) for k in range(len(self.axes))]

    # -------------------------------------------------------------------------
    # Scan log management
    # -------------------------------------------------------------------------

    def _write_scan_log_header(self, scan_log: Path) -> None:
        """Write the header for the scan log file."""
        header_cols = []
        for i, (ax, _) in enumerate(self.axes, 1):
            header_cols += [f"Stage_{i}", f"pos_{i}", f"power_{i}"]
        header_cols += [
            "DetectorKey",
            "ImageFile",
            "Exposure_or_IntTime_or_Period",
            "Averages_or_None",
            "MCP_Voltage",
        ]
        header_cols += self.extra_columns

        with open(scan_log, "w", encoding="utf-8") as f:
            f.write("\t".join(header_cols) + "\n")
            f.write(f"# {self.comment}\n")
            for note in self.header_notes:
                f.write(f"# {note}\n")

            if self.points_input is not None:
                f.write(f"# Explicit point list: {len(self.points_input)} points (not a Cartesian grid)\n")

            for k, (ax, _) in enumerate(self.axes):
                wp = _wp_index_from_stage_key(ax)
                meta = self.axes_meta.get(ax, {})
                pm_on = bool(meta.get("pm", False))

                if self.points_input is not None:
                    vals = [float(pt[k]) for pt in self.points_input]
                    kind = "power fraction" if (wp is not None and pm_on) else "value"
                    rng = f"[{min(vals):.6g}, {max(vals):.6g}]" if vals else "[]"
                    f.write(f"#   {ax}: {len(set(vals))} distinct {kind}s in {rng}\n")

                if wp is not None and pm_on:
                    calib_path = meta.get("calib_path", REGISTRY.get(_reg_key_calib_path(wp)) or "unknown")
                    mv = meta.get("max_value_W", REGISTRY.get(_reg_key_maxvalue(wp)))
                    mv_txt = "none" if mv is None else f"{float(mv):.6g} W"
                    f.write(
                        f"# PowerMode ON for {ax} (WP{wp}) | calib={calib_path} | max_value={mv_txt}\n"
                    )
                    if self.points_input is not None:
                        continue
                    f.write(
                        f"#   Start fraction={float(meta.get('start_fraction', float('nan'))):.6f} | "
                        f"Start angle={float(meta.get('start_angle_deg', float('nan'))):.3f} deg | "
                        f"Rotation={float(meta.get('delta_deg', float('nan'))):.3f} deg | "
                        f"Step={float(meta.get('step_deg', float('nan'))):.3f} deg\n"
                    )
                elif self.points_input is None:
                    f.write(
                        f"# PowerMode OFF for {ax} | "
                        f"Start={float(meta.get('start', float('nan'))):.6g} | "
                        f"End={float(meta.get('end', float('nan'))):.6g} | "
                        f"Step={float(meta.get('step', float('nan'))):.6g}\n"
                    )

    def _create_scan_log(self) -> Path:
        """Create or reuse scan log file."""
        scan_dir = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        if self.existing_scan_log:
            return Path(self.existing_scan_log)

        date_str = f"{self.timestamp:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1

        self._write_scan_log_header(candidate)
        return candidate

    # -------------------------------------------------------------------------
    # Image/spectrum saving
    # -------------------------------------------------------------------------

    def _save_image(
        self, det_key: str, dev, frame: np.ndarray, exposure_us: int, tag: str,
        is_8bit: bool = False, meta: dict | None = None,
    ) -> str:
        """Save an image and return the filename."""
        det_name = _detector_display_name(det_key, dev, meta)
        det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / det_name
        ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fn = f"{det_name}_{tag}_{ts_ms}.png"

        # Merge incoming meta from grab_frame_for_scan (DarkSubtracted, ROI_px,
        # CameraName, etc.) with scan-level fields. Local fields take precedence.
        file_meta = dict(meta) if meta else {}
        file_meta.update({
            "Exposure_us": exposure_us,
            "Gain": file_meta.get("Gain", ""),
            "Comment": self.comment,
        })

        if is_8bit:
            frame_out = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
            save_png_with_meta_8bit(det_day, fn, frame_out, file_meta)
        else:
            frame_out = np.clip(frame, 0, 65535).astype(np.uint16, copy=False)
            save_png_with_meta(det_day, fn, frame_out, file_meta)
        return fn

    def _save_spectrum(
        self, det_key: str, dev, wl_nm: np.ndarray, counts: np.ndarray, int_ms: float, averages: int
    ) -> str:
        """Save a spectrum and return the filename."""
        det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "Avaspec"
        safe_name = _detector_display_name(det_key, dev, None).replace(" ", "")
        ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = "Background" if self.background else "Spectrum"
        fn = f"{safe_name}_{tag}_{ts_ms}.txt"

        header = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IntegrationTime_ms": int_ms,
            "Averages": averages,
            "Comment": self.comment,
            "CalibrationApplied": bool(getattr(dev, "has_calibration", lambda: False)()),
        }

        det_day.mkdir(parents=True, exist_ok=True)
        path = det_day / fn
        lines = [f"# {k}: {v}" for k, v in header.items()]
        lines.append("Wavelength_nm;Counts")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            for xv, yv in zip(wl_nm, counts):
                f.write(f"{float(xv):.6f};{float(yv):.6f}\n")
        return fn

    # -------------------------------------------------------------------------
    # Detector capture methods
    # -------------------------------------------------------------------------

    def _capture_camera(self, det_key: str, dev, params: tuple) -> tuple[str, str]:
        """Capture from a camera detector."""
        exposure_or_int = int(params[0]) if len(params) >= 1 else 0
        averages = int(params[1]) if len(params) >= 2 else 1
        is_daheng = "daheng" in det_key.lower()

        try:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
                exposure_us=int(exposure_or_int),
            )
        except TypeError:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
            )

        exp_meta = int((meta or {}).get("Exposure_us", exposure_or_int))
        tag = "Background" if self.background else "Image"
        data_fn = self._save_image(det_key, dev, frame, exp_meta, tag, is_8bit=is_daheng, meta=meta)
        saved_label = f"exp {exp_meta} µs"

        return data_fn, saved_label

    def _capture_spectrometer(self, det_key: str, dev, params: tuple) -> tuple[str, str]:
        """Capture from a spectrometer."""
        exposure_or_int = float(params[0]) if len(params) >= 1 else 0.0
        averages = int(params[1]) if len(params) >= 2 else 1

        if hasattr(dev, "get_wavelengths"):
            wl = np.asarray(dev.get_wavelengths(), dtype=float)
        else:
            wl = np.asarray(getattr(dev, "wavelength", None), dtype=float)

        if wl is None or wl.size == 0:
            raise ValueError(f"{det_key}: wavelength array empty")

        if hasattr(dev, "grab_spectrum_for_scan"):
            counts, meta = dev.grab_spectrum_for_scan(int_ms=float(exposure_or_int), averages=int(averages))
            counts = np.asarray(counts, dtype=float)
            int_ms = float((meta or {}).get("Integration_ms", float(exposure_or_int)))
        else:
            buf = []
            for _ in range(int(averages)):
                _ts, _data = dev.measure_spectrum(float(exposure_or_int), 1)
                buf.append(np.asarray(_data, dtype=float))
                time.sleep(SPECTRUM_MEASUREMENT_DELAY_S)
            counts = np.mean(np.stack(buf, axis=0), axis=0)
            int_ms = float(exposure_or_int)

        if counts.size != wl.size:
            raise ValueError(f"{det_key}: spectrum length mismatch")

        data_fn = self._save_spectrum(det_key, dev, wl, counts, int_ms, averages)
        saved_label = f"int {int_ms:.0f} ms"

        return data_fn, saved_label

    def _capture_powermeter(self, det_key: str, dev, params: tuple) -> tuple[str, str]:
        """Capture from a power meter."""
        period_ms = float(params[0]) if len(params) >= 1 else 100.0
        averages = int(params[1]) if len(params) >= 2 else 1
        wavelength_nm = float(params[2]) if len(params) >= 3 else None

        if wavelength_nm is not None and hasattr(dev, "set_wavelength"):
            try:
                dev.set_wavelength(float(wavelength_nm))
            except Exception:
                pass

        vals = []
        n_avg = max(1, int(averages))
        for i in range(n_avg):
            v = float(dev.read_power())
            vals.append(v)
            if i + 1 < n_avg:
                time.sleep(period_ms / 1000.0)

        power = float(np.mean(vals)) if vals else float("nan")
        data_fn = f"{power:.9f}"
        saved_label = f"P={power:.3e} W"

        return data_fn, saved_label

    # -------------------------------------------------------------------------
    # Stage movement
    # -------------------------------------------------------------------------

    def _move_slm_axis(self, ax: str, pos: float) -> None:
        """Move an SLM virtual axis (slm:ClassName:FieldName or slm:Rotation)."""
        slm_window = REGISTRY.get("slm:red:window")
        if slm_window is None:
            raise RuntimeError("SLM window not registered.")

        if ax == SLM_ROTATION_AXIS:
            slm_window.set_rotation("red", float(pos))
            label = "rotation [deg]"
        else:
            parts = ax.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid SLM axis format '{ax}'. Expected slm:ClassName:FieldName")

            _, class_name, field_name = parts

            active_classes = REGISTRY.get("slm:red:active_classes") or []
            widgets = REGISTRY.get("slm:red:widgets") or []

            if class_name not in active_classes:
                raise ValueError(f"SLM class '{class_name}' is not active on the red SLM.")

            phase_widget = None
            for w in widgets:
                if getattr(w, "name_", lambda: "")() == class_name:
                    phase_widget = w
                    break

            if phase_widget is None:
                raise ValueError(f"SLM widget for '{class_name}' not found in registry.")

            if not hasattr(phase_widget, field_name):
                raise ValueError(f"Field '{field_name}' does not exist in SLM class '{class_name}'.")

            widget = getattr(phase_widget, field_name)
            widget.setText(str(pos))
            label = f"{class_name}:{field_name}"

        levels = slm_window.compose_levels()

        slm_red = REGISTRY.get("slm:red:controller")
        if slm_red is None:
            raise RuntimeError("Red SLM is not active.")

        screen_num = self.axes_meta[ax].get("screen", 3)
        slm_red.publish(levels, screen_num=screen_num)
        self._emit(f"SLM {label} = {pos}")

    def _prepare_move_targets(self, ui_combo: list[tuple[str, float]]) -> tuple[list, list]:
        """Prepare move targets and log entries for a grid point."""
        move_targets = []
        log_combo = []

        for ax, pos in ui_combo:
            wp = _wp_index_from_stage_key(ax)
            meta = self.axes_meta.get(ax, {})
            pm_on = bool(meta.get("pm", False))

            if wp is not None and pm_on:
                angle = float(pos)
                amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                if amp_off[1] is None:
                    raise ValueError(f"{ax}: Power Mode ON but no calibration phase")
                phase = float(amp_off[1])
                frac = angle_to_power(angle, phase)
                mv = REGISTRY.get(_reg_key_maxvalue(wp))
                power_val = frac if mv is None else frac * float(mv)

                move_targets.append((ax, angle))
                log_combo.append((ax, angle, power_val))
            else:
                move_targets.append((ax, float(pos)))
                log_combo.append((ax, float(pos), ""))

        return move_targets, log_combo

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _initialize_stages(self) -> dict:
        """Initialize all stages."""
        stages = {}
        for stage_key, _ in self.axes:
            if stage_key.startswith("slm:"):
                stages[stage_key] = "VIRTUAL_SLM"
            else:
                stg = REGISTRY.get(stage_key)
                if stg is None:
                    raise ValueError(f"Stage '{stage_key}' not found")
                stages[stage_key] = stg
        return stages

    def _initialize_detectors(self) -> dict:
        """Initialize all detectors."""
        detectors = {}
        for det_key, params in self.camera_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                raise ValueError(f"Detector '{det_key}' not found")

            is_camera = hasattr(dev, "grab_frame_for_scan")
            is_spectro = hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan")
            is_pow = hasattr(dev, "fetch_power")

            if not (is_camera or is_spectro or is_pow):
                raise ValueError(f"Detector '{det_key}' doesn't expose a scan API")

            # Pre-configure detector
            try:
                if is_camera:
                    exposure = int(params[0])
                    if hasattr(dev, "set_exposure_us"):
                        dev.set_exposure_us(exposure)
                    elif hasattr(dev, "setExposureUS"):
                        dev.setExposureUS(exposure)
                    elif hasattr(dev, "set_exposure"):
                        dev.set_exposure(exposure)
                elif is_pow:
                    if len(params) >= 2 and hasattr(dev, "set_avg"):
                        try:
                            dev.set_avg(int(params[1]))
                        except Exception:
                            pass
                    if len(params) >= 3 and hasattr(dev, "set_wavelength"):
                        try:
                            dev.set_wavelength(float(params[2]))
                        except Exception:
                            pass
            except Exception as e:
                self._emit(f"Warning: failed to preset on '{det_key}': {e}")

            detectors[det_key] = dev
        return detectors

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    def _format_position_log(self, log_combo: list) -> str:
        """Format position info for logging."""
        return ", ".join(
            [
                f"{ax}: pos={float(pv):.6f}" + ("" if (powv == "") else f", power={float(powv):.6f}")
                for ax, pv, powv in log_combo
            ]
        )

    def run(self) -> None:
        try:
            self._resolve_explicit_points()
            stages = self._initialize_stages()
            detectors = self._initialize_detectors()
            scan_log = self._create_scan_log()
        except ValueError as e:
            self._emit(str(e))
            self.finished.emit("")
            return

        total_images = self._total_points() * max(1, len(self.camera_params))
        done = 0
        last_targets: dict[str, float] = {}

        try:
            for point_idx, ui_combo in self._iter_points():
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return

                extra_vals = []
                if point_idx is not None and self.point_extras is not None:
                    extras = self.point_extras[point_idx]
                    extra_vals = [str(extras.get(c, "")) for c in self.extra_columns]

                try:
                    move_targets, log_combo = self._prepare_move_targets(ui_combo)
                except ValueError as e:
                    self._emit(str(e))
                    self.finished.emit("")
                    return

                # Move all axes (skip in single-shot mode — stages already at final position)
                if not self.single_shot:
                    move_ok = True
                    for ax, move_val in move_targets:
                        # Outer axes repeat their value over many points; re-sending it only
                        # costs time (an extra SLM publish or a no-op stage move).
                        if last_targets.get(ax) == move_val:
                            continue
                        try:
                            if ax.startswith("slm:"):
                                self._move_slm_axis(ax, move_val)
                            else:
                                stages[ax].move_to(float(move_val), blocking=True)
                            last_targets[ax] = move_val
                        except Exception as e:
                            last_targets.pop(ax, None)
                            self._emit(f"Move {ax} -> {move_val:.6f} failed: {e}")
                            move_ok = False
                            break

                    if not move_ok:
                        done += len(detectors)
                        self.progress.emit(done, total_images)
                        continue

                    time.sleep(float(self.settle_s))

                # Capture from all detectors
                for det_key, dev in detectors.items():
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    params = self.camera_params.get(det_key, (0, 1))

                    try:
                        if hasattr(dev, "grab_frame_for_scan"):
                            data_fn, saved_label = self._capture_camera(det_key, dev, params)
                        elif hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"):
                            data_fn, saved_label = self._capture_spectrometer(det_key, dev, params)
                        else:
                            data_fn, saved_label = self._capture_powermeter(det_key, dev, params)

                        # Write log row
                        row = []
                        for ax, pos_val, power_val in log_combo:
                            row += [
                                ax,
                                f"{float(pos_val):.9f}",
                                ("" if power_val == "" else f"{float(power_val):.9f}"),
                            ]

                        row += [
                            det_key,
                            data_fn,
                            str(params[0] if len(params) >= 1 else ""),
                            str(params[1] if len(params) >= 2 else ""),
                            str(self.mcp_voltage),
                        ]
                        row += extra_vals

                        with open(scan_log, "a", encoding="utf-8") as f:
                            f.write("\t".join(row) + "\n")

                        pos_log = self._format_position_log(log_combo)
                        avg = int(params[1] if len(params) >= 2 else 1)
                        self._emit(f"Saved {data_fn} @ {pos_log} on {det_key} ({saved_label}, avg {avg})")

                    except Exception as e:
                        pos_log = self._format_position_log(log_combo)
                        self._emit(f"Capture failed @ {pos_log} on {det_key}: {e}")

                    done += 1
                    self.progress.emit(done, total_images)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


# -----------------------------------------------------------------------------
# GridScanTab
# -----------------------------------------------------------------------------


class GridScanTab(QWidget):
    """Tab for multi-axis grid scan with multiple detectors."""

    _log_source = "GridScan"

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self._log = log_panel
        self._thread: QThread | None = None
        self._worker: GridScanWorker | None = None
        self._doing_background = False
        self._cached_params: dict | None = None
        self._last_scan_log_path: str | None = None

        self._init_ui()
        self._refresh_devices()

        # Power mode sync timer
        self._pm_sync = QTimer(self)
        self._pm_sync.setInterval(POWER_MODE_SYNC_INTERVAL_MS)
        self._pm_sync.timeout.connect(self._sync_power_mode_from_registry)
        self._pm_sync.start()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Axes group
        main.addWidget(self._create_axes_group())

        # Detectors group
        main.addWidget(self._create_detectors_group())

        # Parameters group
        main.addWidget(self._create_parameters_group())

        # Controls row
        main.addLayout(self._create_controls_row())

        # Refresh button
        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        btn_refresh = QPushButton("Refresh Devices")
        btn_refresh.clicked.connect(self._refresh_devices)
        refresh_row.addWidget(btn_refresh)
        main.addLayout(refresh_row)

    def _create_axes_group(self) -> QGroupBox:
        group = QGroupBox("Axes")
        layout = QVBoxLayout(group)

        # Axis picker row
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Stage or slm:<Class>:"))
        self._stage_picker = QComboBox()
        picker.addWidget(self._stage_picker, 1)
        btn_add = QPushButton("Add Axis")
        btn_add.clicked.connect(self._on_add_axis)
        picker.addWidget(btn_add)
        layout.addLayout(picker)

        # Axes table
        self._axes_tbl = QTableWidget(0, 8)
        self._axes_tbl.setHorizontalHeaderLabels(
            ["Stage", "Param", "Start", "End", "Step", "Screen", "Power Mode", "Max Value (W)"]
        )
        self._axes_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._axes_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        layout.addWidget(self._axes_tbl)

        # Move buttons
        move_row = QHBoxLayout()
        move_row.addStretch(1)
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(lambda: self._move_axis_row(-1))
        move_row.addWidget(btn_up)
        btn_down = QPushButton("Down")
        btn_down.clicked.connect(lambda: self._move_axis_row(+1))
        move_row.addWidget(btn_down)
        layout.addLayout(move_row)

        # Remove button
        rm_row = QHBoxLayout()
        rm_row.addStretch(1)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._on_remove_axis)
        rm_row.addWidget(btn_remove)
        layout.addLayout(rm_row)

        return group

    def _create_detectors_group(self) -> QGroupBox:
        group = QGroupBox("Detectors")
        layout = QVBoxLayout(group)

        # Detector picker row
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Detector:"))
        self._cam_picker = QComboBox()
        picker.addWidget(self._cam_picker, 1)
        btn_add = QPushButton("Add Detector")
        btn_add.clicked.connect(self._on_add_detector)
        picker.addWidget(btn_add)
        layout.addLayout(picker)

        # Detectors table
        self._cam_tbl = QTableWidget(0, 4)
        self._cam_tbl.setHorizontalHeaderLabels(
            ["Detector Key", "Exposure_us / Int_ms", "Wavelength_nm", "Averages"]
        )
        self._cam_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._cam_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        layout.addWidget(self._cam_tbl)

        # Remove button
        rm_row = QHBoxLayout()
        rm_row.addStretch(1)
        btn_remove = QPushButton("Remove Selected Detector")
        btn_remove.clicked.connect(self._on_remove_detector)
        rm_row.addWidget(btn_remove)
        layout.addLayout(rm_row)

        return group

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Scan Parameters")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Settle (s)"))
        self._settle_sb = QDoubleSpinBox()
        self._settle_sb.setDecimals(2)
        self._settle_sb.setRange(0.0, 60.0)
        self._settle_sb.setValue(0.5)
        layout.addWidget(self._settle_sb)

        layout.addWidget(QLabel("Scan Name"))
        self._scan_name_edit = QLineEdit("")
        layout.addWidget(self._scan_name_edit, 1)

        layout.addWidget(QLabel("Comment"))
        self._comment_edit = QLineEdit("")
        layout.addWidget(self._comment_edit, 2)

        layout.addWidget(QLabel("MCP Voltage"))
        self._mcp_edit = QLineEdit("")
        layout.addWidget(self._mcp_edit, 1)

        return group

    def _create_controls_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self._estimate_btn = QPushButton("Estimate Scan Time")
        self._estimate_btn.clicked.connect(self._on_estimate_time)
        layout.addWidget(self._estimate_btn)

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        layout.addWidget(self._abort_btn)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)

        return layout

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source=self._log_source)

    # -------------------------------------------------------------------------
    # Device management
    # -------------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._stage_picker.clear()
        for k in REGISTRY.keys("stage:"):
            if not k.startswith("stage:serial:"):
                self._stage_picker.addItem(k)
        for t in PhaseSettings.types:
            self._stage_picker.addItem(f"slm:{t}")
        self._stage_picker.addItem(SLM_ROTATION_AXIS)

        self._cam_picker.clear()
        for prefix in (
            "camera:daheng:",
            "camera:andor:",
            "camera:lumenera:",
            "spectrometer:avaspec:",
            "powermeter:",
        ):
            for k in REGISTRY.keys(prefix):
                if ":index:" not in k:
                    self._cam_picker.addItem(k)

    def _sync_power_mode_from_registry(self) -> None:
        """Sync power mode checkboxes with registry values."""
        for r in range(self._axes_tbl.rowCount()):
            ax = (self._axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if ax.startswith("slm:"):
                continue
            wp = _wp_index_from_stage_key(ax)
            if wp is None:
                continue
            w = self._axes_tbl.cellWidget(r, 6)
            if not hasattr(w, "setChecked"):
                continue
            val = REGISTRY.get(_reg_key_powermode(wp))
            if isinstance(val, bool) and val != w.isChecked():
                w.blockSignals(True)
                w.setChecked(val)
                w.blockSignals(False)

    # -------------------------------------------------------------------------
    # Axis table management
    # -------------------------------------------------------------------------

    def _on_add_axis(self) -> None:
        ax = self._stage_picker.currentText().strip()
        if not ax:
            QMessageBox.warning(self, "Pick an axis", "Select a stage or slm:<Class>.")
            return

        r = self._axes_tbl.rowCount()
        self._axes_tbl.insertRow(r)
        self._axes_tbl.setItem(r, 0, QTableWidgetItem(ax))

        if ax.startswith("slm:"):
            self._setup_slm_axis_row(r, ax)
        else:
            self._setup_stage_axis_row(r)

    def _setup_slm_axis_row(self, r: int, ax: str) -> None:
        """Setup a row for an SLM axis."""
        if ax == SLM_ROTATION_AXIS:
            param = QTableWidgetItem("")
            param.setFlags(param.flags() & ~Qt.ItemIsEditable)
            param.setToolTip("Rotates every rotatable phase (SLM window rotation field)")
            self._axes_tbl.setItem(r, 1, param)
            for col, text in ((2, "0"), (3, "178"), (4, "2"), (5, "1"), (7, "")):
                self._axes_tbl.setItem(r, col, QTableWidgetItem(text))
            pm = QCheckBox()
            pm.setEnabled(False)
            self._axes_tbl.setCellWidget(r, 6, pm)
            return

        parts = ax.split(":")
        if len(parts) < 2:
            QMessageBox.critical(self, "Invalid SLM axis", "SLM axis must be slm:ClassName")
            return

        class_name = parts[1]

        try:
            phase_ref = PhaseSettings.new_type(None, class_name)
        except Exception:
            valid = ", ".join(sorted(PhaseSettings.types.keys()))
            QMessageBox.critical(
                self,
                "Unknown SLM class",
                f"Class '{class_name}' not found.\nValid classes:\n{valid}",
            )
            return

        self._axes_tbl.setItem(r, 1, QTableWidgetItem(""))
        self._axes_tbl.setItem(r, 2, QTableWidgetItem("0.0"))
        self._axes_tbl.setItem(r, 3, QTableWidgetItem("1.0"))
        self._axes_tbl.setItem(r, 4, QTableWidgetItem("0.1"))
        self._axes_tbl.setItem(r, 5, QTableWidgetItem("1"))

        pm = QCheckBox()
        pm.setEnabled(False)
        self._axes_tbl.setCellWidget(r, 6, pm)

        self._axes_tbl.setItem(r, 7, QTableWidgetItem(""))

        def on_item_changed(item):
            if item.row() != r or item.column() != 1:
                return
            param = item.text().strip()
            if not param:
                return
            if not hasattr(phase_ref, param):
                valid = sorted([k for k in dir(phase_ref) if k.startswith("le_")])
                QMessageBox.critical(
                    self,
                    "Invalid SLM parameter",
                    f"Parameter '{param}' does not exist for class '{class_name}'.\n"
                    f"Valid parameters:\n" + "\n".join(valid),
                )
                return
            new_name = f"slm:{class_name}:{param}"
            self._axes_tbl.item(r, 0).setText(new_name)

        self._axes_tbl.itemChanged.connect(on_item_changed)

    def _setup_stage_axis_row(self, r: int) -> None:
        """Setup a row for a regular stage axis."""
        self._axes_tbl.setItem(r, 1, QTableWidgetItem(""))
        self._axes_tbl.setItem(r, 2, QTableWidgetItem("0.0"))
        self._axes_tbl.setItem(r, 3, QTableWidgetItem("1.0"))
        self._axes_tbl.setItem(r, 4, QTableWidgetItem("0.1"))
        self._axes_tbl.setItem(r, 5, QTableWidgetItem(""))

        pm = QCheckBox()
        self._axes_tbl.setCellWidget(r, 6, pm)

        self._axes_tbl.setItem(r, 7, QTableWidgetItem(""))

    def _on_remove_axis(self) -> None:
        rows = sorted({idx.row() for idx in self._axes_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self._axes_tbl.removeRow(r)

    def _move_axis_row(self, delta: int) -> None:
        sel = self._axes_tbl.selectedIndexes()
        if not sel:
            return
        rows = sorted({i.row() for i in sel})
        if len(rows) != 1:
            return
        r = rows[0]
        d = r + delta
        if 0 <= d < self._axes_tbl.rowCount():
            self._swap_axis_rows(r, d)
            self._axes_tbl.selectRow(d)

    def _swap_axis_rows(self, r1: int, r2: int) -> None:
        for c in range(self._axes_tbl.columnCount()):
            w1 = self._axes_tbl.cellWidget(r1, c)
            w2 = self._axes_tbl.cellWidget(r2, c)
            if hasattr(w1, "isChecked") and hasattr(w2, "isChecked"):
                checked1, checked2 = w1.isChecked(), w2.isChecked()
                w1.setChecked(checked2)
                w2.setChecked(checked1)
                continue

            x = self._axes_tbl.item(r1, c)
            y = self._axes_tbl.item(r2, c)
            t1 = x.text() if x else ""
            t2 = y.text() if y else ""
            if x:
                x.setText(t2)
            else:
                self._axes_tbl.setItem(r1, c, QTableWidgetItem(t2))
            if y:
                y.setText(t1)
            else:
                self._axes_tbl.setItem(r2, c, QTableWidgetItem(t1))

    # -------------------------------------------------------------------------
    # Detector table management
    # -------------------------------------------------------------------------

    def _on_add_detector(self) -> None:
        cam_key = self._cam_picker.currentText().strip()
        if not cam_key:
            QMessageBox.warning(self, "Pick a detector", "Select a detector to add.")
            return
        r = self._cam_tbl.rowCount()
        self._cam_tbl.insertRow(r)
        self._cam_tbl.setItem(r, 0, QTableWidgetItem(cam_key))
        self._cam_tbl.setItem(r, 1, QTableWidgetItem("5000"))
        self._cam_tbl.setItem(r, 2, QTableWidgetItem(""))
        self._cam_tbl.setItem(r, 3, QTableWidgetItem("1"))

    def _on_remove_detector(self) -> None:
        rows = sorted({i.row() for i in self._cam_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self._cam_tbl.removeRow(r)

    # -------------------------------------------------------------------------
    # Parameter collection
    # -------------------------------------------------------------------------

    def _collect_params(self) -> dict:
        """Collect all scan parameters from UI."""
        axes = []
        axes_meta = {}

        if self._axes_tbl.rowCount() == 0:
            raise ValueError("Add at least one axis.")

        for r in range(self._axes_tbl.rowCount()):
            ax = (self._axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            param = (self._axes_tbl.item(r, 1) or QTableWidgetItem("")).text().strip()

            try:
                start = float((self._axes_tbl.item(r, 2) or QTableWidgetItem("0")).text())
                end = float((self._axes_tbl.item(r, 3) or QTableWidgetItem("0")).text())
                step = float((self._axes_tbl.item(r, 4) or QTableWidgetItem("1")).text())
            except ValueError:
                raise ValueError(f"Invalid numeric value in axis row {r + 1}.")

            if ax.startswith("slm:"):
                if param == "" and ax != SLM_ROTATION_AXIS:
                    raise ValueError(f"SLM axis {ax}: missing parameter name.")
                vals = generate_positions(start, end, step)
                axes.append((ax, vals))
                axes_meta[ax] = {
                    "param": param,
                    "screen": int((self._axes_tbl.item(r, 5) or QTableWidgetItem("1")).text()),
                }
                continue

            wp = _wp_index_from_stage_key(ax)
            pm = False
            if wp is not None:
                w = self._axes_tbl.cellWidget(r, 6)
                pm = bool(w.isChecked()) if w else False

            if pm and wp is not None:
                sf = max(0.0, min(1.0, start))
                amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                if amp_off[1] is None:
                    raise ValueError(f"{ax}: Power mode ON but no calibration.")
                phase = float(amp_off[1])
                start_angle = power_to_angle(sf, phase)
                end_angle_abs = start_angle + end
                pos = generate_positions(start_angle, end_angle_abs, step)
                axes.append((ax, pos))

                max_item = self._axes_tbl.item(r, 7)
                try:
                    mv = float((max_item.text() if max_item else "").strip())
                except Exception:
                    mv = float("nan")
                if not (np.isfinite(mv) and mv > 0):
                    raise ValueError(f"{ax}: invalid max power value.")
                REGISTRY.register(_reg_key_maxvalue(wp), float(mv))

                axes_meta[ax] = {
                    "pm": True,
                    "start_fraction": float(sf),
                    "start_angle_deg": float(start_angle),
                    "delta_deg": float(end),
                    "step_deg": float(step),
                    "max_value_W": float(mv),
                }
            else:
                pos = generate_positions(start, end, step)
                axes.append((ax, pos))
                axes_meta[ax] = {
                    "pm": False,
                    "start": float(start),
                    "end": float(end),
                    "step": float(step),
                }

        return {"axes": axes, "axes_meta": axes_meta, **self._collect_common_params()}

    def _collect_common_params(self) -> dict:
        """Detector table and scan-name/settle/comment/MCP fields."""
        cam_params = {}
        if self._cam_tbl.rowCount() == 0:
            raise ValueError("Add at least one detector.")

        for r in range(self._cam_tbl.rowCount()):
            cam = (self._cam_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not cam:
                raise ValueError(f"Empty detector key row {r + 1}.")
            p1 = (self._cam_tbl.item(r, 1) or QTableWidgetItem("0")).text()
            p2 = (self._cam_tbl.item(r, 2) or QTableWidgetItem("")).text()
            p3 = (self._cam_tbl.item(r, 3) or QTableWidgetItem("1")).text()

            if cam.startswith("powermeter:"):
                cam_params[cam] = (float(p1), int(p3), float(p2 or 1030))
            else:
                cam_params[cam] = (int(float(p1)), int(float(p3)))

        settle = float(self._settle_sb.value())
        name = self._scan_name_edit.text().strip()
        if not name:
            raise ValueError("Missing scan name.")
        comment = self._comment_edit.text()
        mcp = self._mcp_edit.text().strip()

        return {
            "camera_params": cam_params,
            "settle": settle,
            "scan_name": name,
            "comment": comment,
            "mcp_voltage": mcp,
        }

    # -------------------------------------------------------------------------
    # Scan control
    # -------------------------------------------------------------------------

    def _on_estimate_time(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        total_points = self._total_points(p)
        n_detectors = max(1, len(p["camera_params"]))
        total_acquisitions = total_points * n_detectors

        detector_time = sum(
            _detector_time_estimate_s(det_key, params)
            for det_key, params in p["camera_params"].items()
        )
        time_per_point = p["settle"] + detector_time
        total = total_points * time_per_point
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = int(total % 60)

        msg = (
            f"{self._points_summary(p)}\n\n"
            f"Total points: {total_points}\n"
            f"Detectors: {n_detectors}\n"
            f"Total acquisitions: {total_acquisitions}\n\n"
            f"Settle per point: {p['settle']:.2f} s\n"
            f"Detector acquisition per point: {detector_time:.2f} s\n"
            f"Min per point: {time_per_point:.2f} s\n\n"
            f"Estimated MINIMUM total: {hours}h {minutes}min {seconds}s\n\n"
            f"(Lower bound — excludes stage transit time between points.)"
        )
        QMessageBox.information(self, "Scan Time Estimate", msg)
        self._log_message(f"Estimated minimum scan time: {hours}h {minutes}min {seconds}s")

    @staticmethod
    def _total_points(p: dict) -> int:
        if p.get("points") is not None:
            return len(p["points"])
        total = 1
        for _, pos in p["axes"]:
            total *= max(1, len(pos))
        return total

    def _points_summary(self, p: dict) -> str:
        axes_desc = "\n".join(f"  {ax}: {len(pos)} points" for ax, pos in p["axes"])
        return f"Axes:\n{axes_desc}"

    def _on_start(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        if not confirm_large_scan(self, self._total_points(p), len(p["camera_params"])):
            return

        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(background=False, existing=None)
        self._log_message("Scan started…")

    def _launch(self, background: bool, existing: str | None, single_shot: bool = False) -> None:
        p = self._cached_params
        if not p:
            return

        self._thread = QThread(self)
        self._worker = GridScanWorker(
            axes=p["axes"],
            camera_params=p["camera_params"],
            settle_s=p["settle"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            mcp_voltage=p["mcp_voltage"],
            background=background,
            existing_scan_log=existing,
            axes_meta=p.get("axes_meta", {}),
            single_shot=single_shot,
            points=p.get("points"),
            point_extras=p.get("point_extras"),
            header_notes=p.get("header_notes"),
        )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)

        # Calculate total points
        n_points = 1 if single_shot else self._total_points(p)
        total = n_points * max(1, len(p["camera_params"]))
        self._progress.setMaximum(total)
        self._progress.setValue(0)

        self._thread.start()

    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self._progress.setMaximum(n)
        self._progress.setValue(i)

    def _on_finished(self, log_path: str) -> None:
        if log_path:
            self._last_scan_log_path = log_path
            self._log_message(f"Scan finished: {log_path}")
        else:
            self._log_message("Scan finished with errors.")
            self._last_scan_log_path = None

        self._abort_btn.setEnabled(False)
        self._start_btn.setEnabled(True)

        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

        # Offer background options
        if not self._doing_background and self._last_scan_log_path is not None:
            box = QMessageBox(self)
            box.setWindowTitle("Background?")
            box.setText(
                "The scan finished.\n\nDo you want to acquire background?\n"
                "Cut the gas and wait 3-5 min before continuing."
            )
            btn_full = box.addButton("Full background scan", QMessageBox.AcceptRole)
            btn_single = box.addButton("Single background image", QMessageBox.AcceptRole)
            btn_no = box.addButton("No", QMessageBox.RejectRole)
            box.setDefaultButton(btn_no)
            box.exec_()

            clicked = box.clickedButton()
            if clicked is btn_full:
                self._doing_background = True
                self._log_message("Launching full background scan…")
                self._launch(background=True, existing=self._last_scan_log_path)
                return
            elif clicked is btn_single:
                self._doing_background = True
                self._log_message("Acquiring single background image at last scan position…")
                self._launch(
                    background=True,
                    existing=self._last_scan_log_path,
                    single_shot=True,
                )
                return

        self._doing_background = False
        
        