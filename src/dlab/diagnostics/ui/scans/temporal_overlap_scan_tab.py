from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np
import cmasher as cmr

from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QCheckBox,
    QProgressBar,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.paths_utils import data_dir
from dlab.diagnostics.ui.scans.scan_utils import confirm_large_scan, generate_positions


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_avaspec_log(
    folder: Path, spec_name: str, fn: str, int_ms: float, averages: int, comment: str
) -> None:
    """Append entry to daily avaspec log file."""
    log_path = folder / f"{spec_name}_log_{datetime.datetime.now():%Y-%m-%d}.log"
    exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("File Name\tIntegration_ms\tAverages\tComment\n")
        f.write(f"{fn}\t{int_ms}\t{averages}\t{comment}\n")


# -----------------------------------------------------------------------------
# Live view window
# -----------------------------------------------------------------------------


class TOverlapLiveView(QWidget):
    """Live view window showing spectra vs position during scan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(None)
        self.setWindowTitle("Temporal Overlap — Live View")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(800, 500)
        self.resize(900, 550)

        self._wl: np.ndarray | None = None
        self._stack: np.ndarray | None = None
        self._pos_list: list[float] = []
        self._xlim_enabled = False
        self._xlim_center = 515.0
        self._xlim_half = 40.0

        self._init_ui()
        self._center_on_screen()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(8.8, 5.0), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("Wavelength (nm)")
        self._ax.set_ylabel("Position (mm)")
        self._im = None
        self._canvas.draw_idle()

    def _center_on_screen(self) -> None:
        try:
            from PyQt5.QtWidgets import QApplication
            screen_geo = QApplication.primaryScreen().availableGeometry()
            geo = self.frameGeometry()
            geo.moveCenter(screen_geo.center())
            self.move(geo.topLeft())
        except Exception:
            pass

    def set_xlim_params(self, enabled: bool, center: float, half: float) -> None:
        """Set wavelength axis limit parameters."""
        self._xlim_enabled = enabled
        self._xlim_center = center
        self._xlim_half = half

    def reset(self) -> None:
        """Clear all data."""
        self._wl = None
        self._stack = None
        self._pos_list = []
        self._ax.cla()
        self._ax.set_xlabel("Wavelength (nm)")
        self._ax.set_ylabel("Position (mm)")
        self._im = None
        self._canvas.draw_idle()

    def add_spectrum(self, pos_mm: float, wl: np.ndarray, y: np.ndarray) -> None:
        """Add a spectrum to the plot."""
        wl = np.asarray(wl, float).ravel()
        y = np.asarray(y, float).ravel()

        if self._wl is None:
            self._wl = wl
            self._stack = np.zeros((0, wl.size), dtype=float)

        self._stack = np.vstack([self._stack, y])
        self._pos_list.append(float(pos_mm))
        self._update_image()

    def _update_image(self) -> None:
        if self._wl is None or self._stack is None or self._stack.size == 0 or not self._pos_list:
            return

        wl_min = float(self._wl[0])
        wl_max = float(self._wl[-1])
        y_min = float(min(self._pos_list))
        y_max = float(max(self._pos_list))

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if abs(y_max - y_min) < 1e-12:
            y_max = y_min + 1e-9

        self._ax.cla()
        self._ax.set_xlabel("Wavelength (nm)")
        self._ax.set_ylabel("Position (mm)")

        extent = [wl_min, wl_max, y_min, y_max]
        self._im = self._ax.imshow(
            self._stack,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="cmr.amethyst_r",
        )

        if self._xlim_enabled:
            self._ax.set_xlim(self._xlim_center - self._xlim_half, self._xlim_center + self._xlim_half)

        self._canvas.draw_idle()


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------


class TOverlapWorker(QObject):
    """Worker for temporal overlap scan."""

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    partial = pyqtSignal(float, object, object)
    finished = pyqtSignal(str)

    def __init__(
        self,
        stage_key: str,
        spec_key: str,
        positions: list[float],
        settle_s: float,
        int_ms: float,
        averages: int,
        use_processed: bool,
        scan_name: str,
        comment: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.spec_key = spec_key
        self.positions = [float(p) for p in positions]
        self.settle_s = float(settle_s)
        self.int_ms = float(max(1.0, float(int_ms)))
        self.averages = int(max(1, int(averages)))
        self.use_processed = bool(use_processed)
        self.scan_name = (scan_name or "temporal_overlap").strip()
        self.comment = comment or ""
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)

    def _save_spectrum_like_ui(self, wl_nm: np.ndarray, counts_raw: np.ndarray, ctrl) -> Path:
        """Save spectrum in the same format as the UI."""
        now = datetime.datetime.now()
        avaspec_folder = _ensure_dir(data_dir() / f"{now:%Y-%m-%d}" / "avaspec")
        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = "Avaspec"
        fn = avaspec_folder / f"{base}_Spectrum_{safe_ts}.txt"

        wl = np.asarray(wl_nm, float).ravel()
        raw = np.asarray(counts_raw, float).ravel()
        bgsub = ctrl._apply_background(wl, raw) if hasattr(ctrl, "_apply_background") else raw

        use_cal = bool(getattr(ctrl, "_cal_enabled", False))
        divcal = None
        cal_applied = False
        cal_path_str = ""

        if use_cal:
            cal_applied = (
                getattr(ctrl, "_cal_wl", None) is not None
                and getattr(ctrl, "_cal_vals", None) is not None
            )
            if cal_applied:
                cal_path = (get_config() or {}).get("avaspec", {}).get("calibration_file", "")
                cal_path_str = str((ROOT / cal_path).resolve()) if cal_path else ""
                if hasattr(ctrl, "_apply_calibration"):
                    divcal = ctrl._apply_calibration(wl, bgsub)

        display_thr = 0.02
        with open(fn, "w", encoding="utf-8") as f:
            if self.comment:
                f.write(f"# Comment: {self.comment}\n")
            f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"# IntegrationTime_ms: {self.int_ms}\n")
            f.write(f"# Averages: {self.averages}\n")
            f.write(f"# BackgroundApplied: {getattr(ctrl, '_bg_counts', None) is not None}\n")
            f.write(f"# CalibrationApplied: {bool(cal_applied)}\n")
            if cal_applied and cal_path_str:
                f.write(f"# CalibrationFile: {cal_path_str}\n")
            f.write(f"# DisplayThreshold: {display_thr:.4f}\n")

            if cal_applied and divcal is not None:
                f.write("Wavelength_nm;Counts_raw;Counts_bgsub;Counts_bgsub_divcal\n")
                for x, y_raw, y_bg, y_cal in zip(wl, raw, bgsub, divcal):
                    f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f};{float(y_cal):.6f}\n")
            else:
                f.write("Wavelength_nm;Counts_raw;Counts_bgsub\n")
                for x, y_raw, y_bg in zip(wl, raw, bgsub):
                    f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f}\n")

        _append_avaspec_log(avaspec_folder, base, fn.name, self.int_ms, self.averages, self.comment)
        return fn

    def run(self) -> None:
        stage = REGISTRY.get(self.stage_key)
        ctrl = REGISTRY.get(self.spec_key)

        if stage is None:
            self._emit(f"Stage '{self.stage_key}' not found in registry.")
            self.finished.emit("")
            return
        if ctrl is None or not hasattr(ctrl, "measure_once"):
            self._emit(f"Spectrometer '{self.spec_key}' not found or invalid.")
            self.finished.emit("")
            return

        try:
            if hasattr(ctrl, "set_params"):
                ctrl.set_params(self.int_ms, self.averages)
        except Exception as e:
            self._emit(f"Failed to set spectrometer params: {e}")

        now = datetime.datetime.now()
        scan_dir = _ensure_dir(data_dir() / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name)

        # Create scan log
        date_str = f"{now:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        scan_log = candidate

        with open(scan_log, "w", encoding="utf-8") as lf:
            lf.write("SpecFile\tStageKey\tPosition_mm\tIntegration_ms\tAverages\tComment\n")
            lf.write(f"# {self.comment}\n")

        wl_ref = None
        n = len(self.positions)

        for i, pos in enumerate(self.positions, 1):
            if self.abort:
                self._emit("Scan aborted.")
                break

            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f} mm.")
            except Exception as e:
                self._emit(f"Move failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue

            time.sleep(self.settle_s)

            try:
                ts, wl, counts = ctrl.measure_once()
                wl = np.asarray(wl, float).ravel()
                y_raw = np.asarray(counts, float).ravel()
                if self.use_processed and hasattr(ctrl, "process_counts"):
                    y_for_display = np.asarray(ctrl.process_counts(wl, y_raw), float).ravel()
                else:
                    y_for_display = y_raw
            except Exception as e:
                self._emit(f"Acquisition failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue

            if wl_ref is None:
                wl_ref = wl.copy()
                y_disp = y_for_display
                wl_disp = wl
            else:
                if wl.shape != wl_ref.shape or np.max(np.abs(wl - wl_ref)) > 1e-9:
                    y_disp = np.interp(wl_ref, wl, y_for_display)
                    wl_disp = wl_ref
                else:
                    y_disp = y_for_display
                    wl_disp = wl

            try:
                out_file = self._save_spectrum_like_ui(wl, y_raw, ctrl)
            except Exception as e:
                self._emit(f"Save spectrum failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue

            try:
                with open(scan_log, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"{Path(out_file).name}\t{self.stage_key}\t{pos:.6f}\t"
                        f"{self.int_ms:.3f}\t{self.averages}\t{self.comment}\n"
                    )
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")

            self.partial.emit(float(pos), wl_disp.copy(), y_disp.copy())
            self.progress.emit(i, n)
            self._emit(f"Saved {Path(out_file).name} @ {pos:.3f} mm.")

        self.finished.emit(scan_log.as_posix())


# -----------------------------------------------------------------------------
# TOverlapTab
# -----------------------------------------------------------------------------


class TOverlapTab(QWidget):
    """Tab for temporal overlap scan using spectrometer."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self._log = log_panel
        self._thread: QThread | None = None
        self._worker: TOverlapWorker | None = None
        self._live_view: TOverlapLiveView | None = None

        self._init_ui()
        self._refresh_devices()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Devices group
        main.addWidget(self._create_devices_group())

        # Parameters group
        main.addWidget(self._create_parameters_group())

        # Display options group
        main.addWidget(self._create_display_options_group())

        # Controls row
        main.addLayout(self._create_controls_row())

    def _create_devices_group(self) -> QGroupBox:
        group = QGroupBox("Devices")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        layout.addWidget(self._stage_combo, 1)

        layout.addWidget(QLabel("Spectrometer:"))
        self._spec_combo = QComboBox()
        layout.addWidget(self._spec_combo, 1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_devices)
        layout.addWidget(btn_refresh)

        return group

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Scan Parameters")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Start (mm)"))
        self._start_sb = QDoubleSpinBox()
        self._start_sb.setDecimals(3)
        self._start_sb.setRange(-1e6, 1e6)
        self._start_sb.setValue(0.0)
        layout.addWidget(self._start_sb)

        layout.addWidget(QLabel("End (mm)"))
        self._end_sb = QDoubleSpinBox()
        self._end_sb.setDecimals(3)
        self._end_sb.setRange(-1e6, 1e6)
        self._end_sb.setValue(6.0)
        layout.addWidget(self._end_sb)

        layout.addWidget(QLabel("Step (mm)"))
        self._step_sb = QDoubleSpinBox()
        self._step_sb.setDecimals(3)
        self._step_sb.setRange(1e-6, 1e6)
        self._step_sb.setValue(0.05)
        layout.addWidget(self._step_sb)

        layout.addWidget(QLabel("Settle (s)"))
        self._settle_sb = QDoubleSpinBox()
        self._settle_sb.setDecimals(2)
        self._settle_sb.setRange(0.0, 60.0)
        self._settle_sb.setValue(0.20)
        layout.addWidget(self._settle_sb)

        layout.addWidget(QLabel("Int (ms)"))
        self._int_sb = QDoubleSpinBox()
        self._int_sb.setDecimals(1)
        self._int_sb.setRange(1.0, 600000.0)
        self._int_sb.setValue(50.0)
        layout.addWidget(self._int_sb)

        layout.addWidget(QLabel("Avg"))
        self._avg_sb = QDoubleSpinBox()
        self._avg_sb.setDecimals(0)
        self._avg_sb.setRange(1, 1000)
        self._avg_sb.setValue(1)
        layout.addWidget(self._avg_sb)

        self._proc_checkbox = QCheckBox("Use processed counts (display only)")
        self._proc_checkbox.setChecked(True)
        layout.addWidget(self._proc_checkbox)

        return group

    def _create_display_options_group(self) -> QGroupBox:
        group = QGroupBox("Display Options")
        layout = QHBoxLayout(group)

        self._xlim_checkbox = QCheckBox("Limit wavelength axis")
        layout.addWidget(self._xlim_checkbox)

        layout.addWidget(QLabel("Center (nm)"))
        self._xlim_center_sb = QDoubleSpinBox()
        self._xlim_center_sb.setDecimals(2)
        self._xlim_center_sb.setRange(0.0, 1e6)
        self._xlim_center_sb.setValue(515.0)
        layout.addWidget(self._xlim_center_sb)

        layout.addWidget(QLabel("± Halfwidth (nm)"))
        self._xlim_half_sb = QDoubleSpinBox()
        self._xlim_half_sb.setDecimals(2)
        self._xlim_half_sb.setRange(0.0, 1e6)
        self._xlim_half_sb.setValue(40.0)
        layout.addWidget(self._xlim_half_sb)

        layout.addStretch(1)
        return group

    def _create_controls_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Comment:"))
        self._comment_edit = QLineEdit("")
        layout.addWidget(self._comment_edit, 1)

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

        self._live_btn = QPushButton("Live View")
        self._live_btn.setCheckable(True)
        self._live_btn.toggled.connect(self._on_toggle_live_view)
        layout.addWidget(self._live_btn)

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
            self._log.log(msg, source="TOverlap")

    # -------------------------------------------------------------------------
    # Device management
    # -------------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._stage_combo.clear()
        for k in REGISTRY.keys("stage:"):
            if k.startswith("stage:serial:"):
                continue
            self._stage_combo.addItem(k)

        self._spec_combo.clear()
        for k in REGISTRY.keys("spectrometer:avaspec:"):
            self._spec_combo.addItem(k)

    # -------------------------------------------------------------------------
    # Scan time estimate
    # -------------------------------------------------------------------------

    def _on_estimate_time(self) -> None:
        try:
            a0 = float(self._start_sb.value())
            a1 = float(self._end_sb.value())
            step = float(self._step_sb.value())
            positions = generate_positions(a0, a1, step)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        settle = float(self._settle_sb.value())
        int_s = float(self._int_sb.value()) / 1000.0
        avg = int(self._avg_sb.value())
        time_per_point = settle + int_s * avg
        total = len(positions) * time_per_point
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = int(total % 60)

        msg = (
            f"Positions: {len(positions)}\n\n"
            f"Settle per point: {settle:.2f} s\n"
            f"Integration x averages: {int_s * avg:.2f} s\n"
            f"Min per point: {time_per_point:.2f} s\n\n"
            f"Estimated MINIMUM total: {hours}h {minutes}min {seconds}s\n\n"
            f"(Lower bound — excludes stage transit time.)"
        )
        QMessageBox.information(self, "Scan Time Estimate", msg)
        self._log_message(f"Estimated minimum scan time: {hours}h {minutes}min {seconds}s")

    # -------------------------------------------------------------------------
    # Scan control
    # -------------------------------------------------------------------------

    def _on_start(self) -> None:
        try:
            stage_key = self._stage_combo.currentText().strip()
            spec_key = self._spec_combo.currentText().strip()
            if not stage_key or not spec_key:
                raise ValueError("Select a stage and a spectrometer.")

            a0 = float(self._start_sb.value())
            a1 = float(self._end_sb.value())
            step = float(self._step_sb.value())
            positions = generate_positions(a0, a1, step)

            settle = float(self._settle_sb.value())
            int_ms = float(self._int_sb.value())
            avg = int(self._avg_sb.value())
            use_processed = bool(self._proc_checkbox.isChecked())
            scan_name = "temporal_overlap"
            comment = self._comment_edit.text()

        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        if not confirm_large_scan(self, len(positions), 1):
            return

        self._thread = QThread(self)
        self._worker = TOverlapWorker(
            stage_key=stage_key,
            spec_key=spec_key,
            positions=positions,
            settle_s=settle,
            int_ms=int_ms,
            averages=avg,
            use_processed=use_processed,
            scan_name=scan_name,
            comment=comment,
        )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._thread.finished.connect(self._thread.deleteLater)

        # Connect live view if open
        if self._live_view:
            self._live_view.reset()
            self._live_view.set_xlim_params(
                self._xlim_checkbox.isChecked(),
                float(self._xlim_center_sb.value()),
                float(self._xlim_half_sb.value()),
            )
            self._worker.partial.connect(self._live_view.add_spectrum, Qt.QueuedConnection)

        self._progress.setMaximum(len(positions))
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._log_message("Scan started…")
        self._thread.start()

    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log_message("Abort requested.")
            self._abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self._progress.setMaximum(n)
        self._progress.setValue(i)

    def _on_finished(self, scan_log_path: str) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self._abort_btn.setEnabled(False)
        self._start_btn.setEnabled(True)

        if scan_log_path:
            self._log_message(f"Scan finished. Log: {scan_log_path}")
        else:
            self._log_message("Scan finished with errors.")

    # -------------------------------------------------------------------------
    # Live view
    # -------------------------------------------------------------------------

    def _on_toggle_live_view(self, checked: bool) -> None:
        if checked:
            if self._live_view is None:
                self._live_view = TOverlapLiveView(None)
            self._live_view.reset()
            self._live_view.set_xlim_params(
                self._xlim_checkbox.isChecked(),
                float(self._xlim_center_sb.value()),
                float(self._xlim_half_sb.value()),
            )
            self._live_view._center_on_screen()
            self._live_view.show()
            self._live_view.raise_()
            self._live_view.activateWindow()
        else:
            if self._live_view:
                self._live_view.hide()