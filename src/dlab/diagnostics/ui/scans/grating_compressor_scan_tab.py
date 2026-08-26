from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import List

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QMessageBox,
    QDoubleSpinBox,
    QComboBox,
    QProgressBar,
    QCheckBox,
    QSizePolicy,
    QApplication,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.paths_utils import data_dir
from dlab.diagnostics.ui.scans.scan_utils import (
    confirm_large_scan,
    generate_positions,
    save_png_with_meta,
)


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------


class GCWorker(QObject):
    """Worker for grating compressor scan."""

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    live_point = pyqtSignal(float, float)
    best_ready = pyqtSignal(float, float)

    def __init__(
        self,
        stage_key: str,
        andor_key: str,
        positions: list[float],
        exposure_us: int,
        averages: int,
        settle_s: float,
        comment: str,
        mcp_voltage: str,
        do_background: bool = False,
        existing_scan_log: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.andor_key = andor_key
        self.positions = list(float(p) for p in positions)
        self.exposure_us = int(exposure_us)
        self.averages = int(max(1, averages))
        self.settle_s = float(settle_s)
        self.scan_name = "grating_compressor"
        self.comment = comment
        self.mcp_voltage = str(mcp_voltage)
        self.do_background = bool(do_background)
        self.existing_scan_log = existing_scan_log
        self.abort = False
        self._best_sum = float("-inf")
        self._best_pos = None

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)

    def _open_or_create_scan_log(self, root: Path, now: datetime.datetime) -> Path:
        if self.existing_scan_log:
            scan_log = Path(self.existing_scan_log)
            if not scan_log.exists():
                scan_log.parent.mkdir(parents=True, exist_ok=True)
                with open(scan_log, "w", encoding="utf-8") as f:
                    f.write("ImageFile\tStageKey\tPosition_mm\tExposure_us\tAverages\tMCP_Voltage\n")
                    f.write(f"# {self.comment}\n")
            return scan_log

        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)
        date_str = now.strftime("%Y-%m-%d")
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        scan_log = candidate
        with open(scan_log, "w", encoding="utf-8") as f:
            f.write("ImageFile\tStageKey\tPosition_mm\tExposure_us\tAverages\tMCP_Voltage\n")
            f.write(f"# {self.comment}\n")
        return scan_log

    def run(self) -> None:
        stage = REGISTRY.get(self.stage_key)
        camwin = REGISTRY.get(self.andor_key)

        if stage is None:
            self._emit(f"Stage '{self.stage_key}' not found.")
            self.finished.emit("")
            return
        if camwin is None:
            self._emit(f"Camera '{self.andor_key}' not found.")
            self.finished.emit("")
            return
        if not hasattr(camwin, "grab_frame_for_scan"):
            self._emit("Selected camera does not expose grab_frame_for_scan().")
            self.finished.emit("")
            return

        try:
            if hasattr(camwin, "set_exposure_us"):
                camwin.set_exposure_us(self.exposure_us)
            elif hasattr(camwin, "setExposureUS"):
                camwin.setExposureUS(self.exposure_us)
            elif hasattr(camwin, "set_exposure"):
                camwin.set_exposure(self.exposure_us)
        except Exception as e:
            self._emit(f"Warning: failed to preset exposure on '{self.andor_key}': {e}")

        now = datetime.datetime.now()
        root = data_dir()
        scan_log = self._open_or_create_scan_log(root, now)

        total = len(self.positions)
        done = 0

        for pos in self.positions:
            if self.abort:
                self._emit("Scan aborted.")
                self.finished.emit(scan_log.as_posix())
                return

            # Move stage
            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f} mm.")
            except Exception as e:
                self._emit(f"Move to {pos:.3f} failed: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            time.sleep(self.settle_s)

            # Capture frame
            try:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=False,
                    dead_pixel_cleanup=True,
                    exposure_us=self.exposure_us,
                )
            except TypeError:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=False,
                    dead_pixel_cleanup=True,
                )
            except Exception as e:
                self._emit(f"Capture failed at {pos:.3f}: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            cam_name = str((meta or {}).get("CameraName", "AndorCam")).strip() or "AndorCam"
            exp_meta = int((meta or {}).get("Exposure_us", self.exposure_us))
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = "Image"
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            cam_fn = f"{cam_name}_{tag}_{ts_ms}.png"

            # Live metric: sum of pixels
            try:
                sum_counts = float(np.sum(frame_u16, dtype=np.uint64))
            except Exception:
                sum_counts = float(np.sum(frame_u16))

            # Update best
            if sum_counts > self._best_sum:
                self._best_sum = sum_counts
                self._best_pos = float(pos)

            try:
                save_png_with_meta(
                    cam_day,
                    cam_fn,
                    frame_u16,
                    {"Exposure_us": exp_meta, "Gain": "", "Comment": self.comment},
                )
            except Exception as e:
                self._emit(f"Save failed at {pos:.3f}: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            # Append scan log row
            try:
                with open(scan_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"{cam_fn}\t{self.stage_key}\t{pos:.6f}\t{exp_meta}\t{self.averages}\t{self.mcp_voltage}\n"
                    )
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")

            # Emit live point
            try:
                self.live_point.emit(float(pos), float(sum_counts))
            except Exception:
                pass

            self._emit(f"Saved {cam_fn} @ {pos:.3f} mm (exp {exp_meta} µs, avg {self.averages}).")
            done += 1
            self.progress.emit(done, total)

        # Background capture
        if self.do_background and not self.abort:
            try:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=True,
                    dead_pixel_cleanup=True,
                    exposure_us=self.exposure_us,
                )
            except TypeError:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=True,
                    dead_pixel_cleanup=True,
                )
            except Exception as e:
                self._emit(f"Background capture failed: {e}")
            else:
                cam_name = str((meta or {}).get("CameraName", "AndorCam")).strip() or "AndorCam"
                exp_meta = int((meta or {}).get("Exposure_us", self.exposure_us))
                ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                tag = "Background"
                cam_day = root / f"{now:%Y-%m-%d}" / cam_name
                cam_fn = f"{cam_name}_{tag}_{ts_ms}.png"
                try:
                    save_png_with_meta(
                        cam_day,
                        cam_fn,
                        frame_u16,
                        {"Exposure_us": exp_meta, "Gain": "", "Comment": self.comment},
                    )
                    with open(scan_log, "a", encoding="utf-8") as f:
                        f.write(
                            f"{cam_fn}\t{self.stage_key}\tBG\t{exp_meta}\t{self.averages}\t{self.mcp_voltage}\n"
                        )
                    self._emit(f"Saved background: {cam_fn}")
                except Exception as e:
                    self._emit(f"Background save/log failed: {e}")

        try:
            if self._best_pos is not None and np.isfinite(self._best_sum):
                self.best_ready.emit(self._best_pos, float(self._best_sum))
        except Exception:
            pass

        self.finished.emit(scan_log.as_posix())


# -----------------------------------------------------------------------------
# Live view window
# -----------------------------------------------------------------------------


class GCScanLiveView(QWidget):
    """Live view window showing sum vs position during scan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(None)
        self.setWindowTitle("GC Scan — Live View (Sum vs Position)")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(700, 420)
        self.resize(880, 520)

        self._xs: List[float] = []
        self._ys: List[float] = []

        self._init_ui()
        self._center_on_screen()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(8, 5))
        self._figure.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("Position (mm)")
        self._ax.set_ylabel("Sum of pixels (a.u.)")
        self._ax.grid(True, which="both", linestyle="--", alpha=0.3)
        (self._line,) = self._ax.plot([], [], linestyle="-")
        self._canvas.draw_idle()

    def _center_on_screen(self) -> None:
        try:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            geo = self.frameGeometry()
            geo.moveCenter(screen_geo.center())
            self.move(geo.topLeft())
        except Exception:
            pass

    def reset(self) -> None:
        """Clear all data points."""
        self._xs.clear()
        self._ys.clear()
        self._line.set_data([], [])
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw_idle()

    def add_point(self, x: float, y: float) -> None:
        """Add a data point to the plot."""
        self._xs.append(x)
        self._ys.append(y)
        self._line.set_data(self._xs, self._ys)
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw_idle()


# -----------------------------------------------------------------------------
# GCScanTab
# -----------------------------------------------------------------------------


class GCScanTab(QWidget):
    """Tab for grating compressor scan."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self._log = log_panel
        self._thread: QThread | None = None
        self._worker: GCWorker | None = None
        self._last_scan_log_path: str | None = None
        self._live_view: GCScanLiveView | None = None

        self._init_ui()
        self._refresh_devices()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Devices group
        main.addWidget(self._create_devices_group())

        # Parameters group
        main.addWidget(self._create_parameters_group())

        # Options group
        main.addWidget(self._create_options_group())

        # Controls row
        main.addLayout(self._create_controls_row())

    def _create_devices_group(self) -> QGroupBox:
        group = QGroupBox("Devices")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Stage:"))
        self._stage_label = QLabel("stage:zaber:grating_compressor")
        layout.addWidget(self._stage_label, 1)

        layout.addWidget(QLabel("Andor camera:"))
        self._cam_combo = QComboBox()
        layout.addWidget(self._cam_combo, 1)

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
        self._start_sb.setValue(24.0)
        layout.addWidget(self._start_sb)

        layout.addWidget(QLabel("End (mm)"))
        self._end_sb = QDoubleSpinBox()
        self._end_sb.setDecimals(3)
        self._end_sb.setRange(-1e6, 1e6)
        self._end_sb.setValue(25.0)
        layout.addWidget(self._end_sb)

        layout.addWidget(QLabel("Step (mm)"))
        self._step_sb = QDoubleSpinBox()
        self._step_sb.setDecimals(3)
        self._step_sb.setRange(1e-6, 1e6)
        self._step_sb.setValue(0.005)
        layout.addWidget(self._step_sb)

        layout.addWidget(QLabel("Settle (s)"))
        self._settle_sb = QDoubleSpinBox()
        self._settle_sb.setDecimals(2)
        self._settle_sb.setRange(0.0, 60.0)
        self._settle_sb.setValue(0.30)
        layout.addWidget(self._settle_sb)

        layout.addWidget(QLabel("Exposure (µs)"))
        self._exp_sb = QDoubleSpinBox()
        self._exp_sb.setDecimals(0)
        self._exp_sb.setRange(1, 5_000_000)
        self._exp_sb.setValue(3000000)
        layout.addWidget(self._exp_sb)

        layout.addWidget(QLabel("Averages"))
        self._avg_sb = QDoubleSpinBox()
        self._avg_sb.setDecimals(0)
        self._avg_sb.setRange(1, 1000)
        self._avg_sb.setValue(1)
        layout.addWidget(self._avg_sb)

        layout.addWidget(QLabel("MCP Voltage"))
        self._mcp_edit = QLineEdit("")
        layout.addWidget(self._mcp_edit)

        layout.addWidget(QLabel("Comment"))
        self._comment_edit = QLineEdit("")
        layout.addWidget(self._comment_edit, 1)

        return group

    def _create_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        layout = QHBoxLayout(group)

        self._bg_checkbox = QCheckBox("Record background after scan")
        self._bg_checkbox.setChecked(False)
        layout.addWidget(self._bg_checkbox)

        layout.addStretch(1)
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
            self._log.log(msg, source="GCScan")

    # -------------------------------------------------------------------------
    # Device management
    # -------------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._cam_combo.clear()
        for k in REGISTRY.keys("camera:andor:"):
            if ":index:" in k:
                continue
            self._cam_combo.addItem(k)

    # -------------------------------------------------------------------------
    # Scan time estimate
    # -------------------------------------------------------------------------

    def _on_estimate_time(self) -> None:
        try:
            start = float(self._start_sb.value())
            end = float(self._end_sb.value())
            step = float(self._step_sb.value())
            pos = generate_positions(start, end, step)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        settle = float(self._settle_sb.value())
        exposure_s = float(self._exp_sb.value()) / 1e6
        avg = int(self._avg_sb.value())
        time_per_point = settle + exposure_s * avg
        total = len(pos) * time_per_point
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = int(total % 60)

        msg = (
            f"Positions: {len(pos)}\n\n"
            f"Settle per point: {settle:.2f} s\n"
            f"Exposure x averages: {exposure_s * avg:.2f} s\n"
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
            stage_key = "stage:zaber:grating_compressor"
            if REGISTRY.get(stage_key) is None:
                raise ValueError(
                    "Grating compressor stage not found in registry. "
                    "Open the Grating Compressor window and Activate it first."
                )
            cam_key = self._cam_combo.currentText().strip()
            if not cam_key:
                raise ValueError("Select an Andor camera.")

            start = float(self._start_sb.value())
            end = float(self._end_sb.value())
            step = float(self._step_sb.value())
            pos = generate_positions(start, end, step)

            settle = float(self._settle_sb.value())
            expo = int(self._exp_sb.value())
            avg = int(self._avg_sb.value())
            comment = self._comment_edit.text()
            mcp_voltage = self._mcp_edit.text()

        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        if not confirm_large_scan(self, len(pos), 1):
            return

        # Create thread/worker
        self._thread = QThread(self)
        self._worker = GCWorker(
            stage_key=stage_key,
            andor_key=cam_key,
            positions=pos,
            exposure_us=expo,
            averages=avg,
            settle_s=settle,
            comment=comment,
            mcp_voltage=mcp_voltage,
            do_background=False,
            existing_scan_log=None,
        )

        # Connect live stream if viewer is open
        if self._live_view:
            self._live_view.reset()
            self._worker.live_point.connect(self._live_view.add_point)

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._progress.setMaximum(len(pos))
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._thread.start()
        self._log_message("Grating compressor scan started…")

    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log_message("Abort requested.")
            self._abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self._progress.setMaximum(n)
        self._progress.setValue(i)

    def _on_finished(self, log_path: str) -> None:
        if log_path:
            self._last_scan_log_path = log_path
            self._log_message(f"Scan finished. Log: {log_path}")
        else:
            self._log_message("Scan finished with errors or aborted.")
            self._last_scan_log_path = None

        do_background = bool(self._bg_checkbox.isChecked() and self._last_scan_log_path)
        skip_background = bool(self._worker and self._worker.abort)

        self._abort_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

        # Optional background capture after scan — routed through the same
        # tracked self._thread/self._worker as the main scan so Abort and the
        # scan-window close guard both see it as a running scan.
        if do_background and not skip_background:
            reply = QMessageBox.information(
                self,
                "Background",
                "Block the beam, then click OK to record one background image.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                self._capture_background()

    def _capture_background(self) -> None:
        """Capture background image after scan."""
        try:
            cam_key = self._cam_combo.currentText().strip()
            stage_key = "stage:zaber:grating_compressor"
            self._log_message("Recording background…")

            self._thread = QThread(self)
            self._worker = GCWorker(
                stage_key=stage_key,
                andor_key=cam_key,
                positions=[],
                exposure_us=int(self._exp_sb.value()),
                averages=int(self._avg_sb.value()),
                settle_s=0.0,
                comment=self._comment_edit.text(),
                mcp_voltage=self._mcp_edit.text(),
                do_background=True,
                existing_scan_log=self._last_scan_log_path,
            )
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.log.connect(self._log_message)
            self._worker.finished.connect(self._on_background_finished)
            self._thread.finished.connect(self._thread.deleteLater)

            self._start_btn.setEnabled(False)
            self._abort_btn.setEnabled(True)
            self._thread.start()

        except Exception as e:
            self._log_message(f"Background failed: {e}")

    def _on_background_finished(self, _path: str) -> None:
        self._log_message("Background captured.")
        self._abort_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    # -------------------------------------------------------------------------
    # Live view
    # -------------------------------------------------------------------------

    def _on_toggle_live_view(self, checked: bool) -> None:
        if checked:
            if self._live_view is None:
                self._live_view = GCScanLiveView(None)
            self._live_view.reset()
            self._live_view._center_on_screen()
            self._live_view.show()
            self._live_view.raise_()
            self._live_view.activateWindow()
        else:
            if self._live_view:
                self._live_view.hide()