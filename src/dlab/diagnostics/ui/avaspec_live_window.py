from __future__ import annotations

import sys
import time
import datetime
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from dlab.hardware.wrappers.avaspec_controller import AvaspecController
from dlab.core.device_registry import REGISTRY
from dlab.utils.paths_utils import data_dir
from dlab.utils.log_panel import LogPanel
from dlab.boot import ROOT, get_config


# -----------------------------------------------------------------------------
# Instance presets
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AvaspecPreset:
    """Default values for an Avaspec live-view instance."""

    label: str
    registry_prefix: str
    ui_registry_key: str


_PRESETS: dict[str, AvaspecPreset] = {
    "w": AvaspecPreset(
        label="\u03c9",
        registry_prefix="spectrometer:avaspec:w",
        ui_registry_key="ui:avaspec_live:w",
    ),
    "2w": AvaspecPreset(
        label="\u03c9/2\u03c9",
        registry_prefix="spectrometer:avaspec:2w",
        ui_registry_key="ui:avaspec_live:2w",
    ),
    "3w": AvaspecPreset(
        label="\u03c9/3\u03c9",
        registry_prefix="spectrometer:avaspec:3w",
        ui_registry_key="ui:avaspec_live:3w",
    ),
}


# -----------------------------------------------------------------------------
# Worker Threads
# -----------------------------------------------------------------------------


class _MeasureThread(QThread):
    """Background thread for continuous spectrum acquisition."""

    data_ready = pyqtSignal(float, object, object)
    error = pyqtSignal(str)

    def __init__(self, ctrl: AvaspecController, int_ms: float, avg: int):
        super().__init__()
        self._ctrl = ctrl
        self._int_ms = float(max(1.0, float(int_ms)))
        self._avg = int(max(1, int(avg)))
        self._running = True
        self._need_apply = True
        self._lock = threading.Lock()

    def update_params(self, int_ms: float, avg: int):
        with self._lock:
            self._int_ms = float(max(1.0, float(int_ms)))
            self._avg = int(max(1, int(avg)))
            self._need_apply = True

    def stop(self):
        self._running = False
        self.requestInterruption()

    def run(self):
        while self._running and not self.isInterruptionRequested():
            try:
                with self._lock:
                    need_apply = self._need_apply
                    int_ms = self._int_ms
                    avg = self._avg

                if need_apply:
                    self._ctrl.set_params(int_ms, avg)
                    with self._lock:
                        self._need_apply = False

                ts, wl, counts = self._ctrl.measure_once()
                self.data_ready.emit(ts, wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break


# -----------------------------------------------------------------------------
# AvaspecLiveWindow
# -----------------------------------------------------------------------------


class AvaspecLiveWindow(QWidget):
    """Live view window for Avaspec spectrometer.

    Parameters
    ----------
    instance_id : str
        Preset identifier (``"w"``, ``"2w"`` or ``"3w"``).  Determines the
        window title, registry keys, and log source.
    preset : AvaspecPreset | None
        Override the built-in preset for *instance_id*.
    log_panel : LogPanel | None
        Shared log panel.
    """

    closed = pyqtSignal()

    def __init__(
        self,
        instance_id: str = "w",
        preset: AvaspecPreset | None = None,
        log_panel: LogPanel | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self._preset = preset or _PRESETS[instance_id]
        self._instance_id = instance_id

        self.setWindowTitle(f"Avaspec Live \u2014 {self._preset.label}")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._ctrl: AvaspecController | None = None
        self._capture_thread: _MeasureThread | None = None
        self._handles: list = []
        self._reg_key: str | None = None

        # Plot state
        self._line = None
        self._fft_line = None
        self._fft_peak_vline = None

        # Blitting state (single background for the whole figure)
        self._bg = None
        self._blit_ready = False
        self._last_xlim = None
        self._last_ylim = None
        self._last_fft_xlim = None
        self._last_fft_ylim = None

        # Data state
        self._last_wl: np.ndarray | None = None
        self._last_counts: np.ndarray | None = None
        self._ref_wl: np.ndarray | None = None

        # Drawing throttle
        self._min_draw_dt = 0.05  # 50 ms
        self._last_draw = 0.0

        # FFT state
        self._fft_indices: np.ndarray | None = None
        self._fft_peak_idx = 300  # Default index

        self._mpl_cid_click = None

        self._init_ui()
        self.resize(1050, 620)

        try:
            REGISTRY.register(self._preset.ui_registry_key, self)
        except Exception:
            pass

    def _init_ui(self):
        root = QVBoxLayout(self)

        # Top: spectrometer selection
        top = QHBoxLayout()
        self._cmb = QComboBox()
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._search_specs)
        self._activate_btn = QPushButton("Activate")
        self._activate_btn.clicked.connect(self._activate)
        self._deactivate_btn = QPushButton("Deactivate")
        self._deactivate_btn.clicked.connect(self._deactivate)
        self._deactivate_btn.setEnabled(False)
        top.addWidget(QLabel("Spectrometer:"))
        top.addWidget(self._cmb, 1)
        top.addWidget(btn_search)
        top.addWidget(self._activate_btn)
        top.addWidget(self._deactivate_btn)
        root.addLayout(top)

        # Acquisition params
        row = QHBoxLayout()
        self._int_edit = QLineEdit("50")
        self._avg_edit = QLineEdit("1")
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_params)
        row.addWidget(QLabel("Integration (ms):"))
        row.addWidget(self._int_edit)
        row.addWidget(QLabel("Averages:"))
        row.addWidget(self._avg_edit)
        row.addWidget(btn_apply)

        self._lbl_ftl = QLabel("FTL: \u2014 fs")
        self._lbl_ftl.setMinimumWidth(140)
        self._lbl_ftl.setAlignment(Qt.AlignCenter)
        self._lbl_ftl.setStyleSheet("QLabel { border: 1px solid #888; padding: 4px; }")
        row.addStretch(1)
        row.addWidget(self._lbl_ftl)
        root.addLayout(row)

        # Background / calibration
        row2 = QHBoxLayout()
        btn_bg_set = QPushButton("Set BG")
        btn_bg_set.clicked.connect(self._set_background)
        btn_bg_clear = QPushButton("Clear BG")
        btn_bg_clear.clicked.connect(self._clear_background)
        self._btn_cal_toggle = QPushButton("Calibration: OFF")
        self._btn_cal_toggle.setCheckable(True)
        self._btn_cal_toggle.clicked.connect(self._toggle_calibration)
        row2.addWidget(btn_bg_set)
        row2.addWidget(btn_bg_clear)
        row2.addWidget(self._btn_cal_toggle)
        row2.addStretch(1)
        root.addLayout(row2)

        # Spectrum zoom + left Y limit
        row_vis = QHBoxLayout()
        self._chk_zoom = QCheckBox("Zoom @ \u03bb\u2080")
        self._chk_zoom.setChecked(True)
        self._cwl_edit = QLineEdit("515")
        self._cwl_edit.setPlaceholderText("\u03bb\u2080 (nm)")
        self._cwl_edit.setFixedWidth(70)
        self._zoom_pm_edit = QLineEdit("20")
        self._zoom_pm_edit.setFixedWidth(50)
        row_vis.addWidget(self._chk_zoom)
        row_vis.addWidget(QLabel("\u03bb\u2080:"))
        row_vis.addWidget(self._cwl_edit)
        row_vis.addWidget(QLabel("\u00b1:"))
        row_vis.addWidget(self._zoom_pm_edit)

        row_vis.addSpacing(15)
        row_vis.addWidget(QLabel("Counts y:"))
        self._ax_ymin_edit = QLineEdit("0")
        self._ax_ymin_edit.setFixedWidth(60)
        row_vis.addWidget(self._ax_ymin_edit)
        self._ax_ymax_edit = QLineEdit("60000")
        self._ax_ymax_edit.setFixedWidth(70)
        row_vis.addWidget(self._ax_ymax_edit)

        btn_ax_ylim = QPushButton("Set")
        btn_ax_ylim.clicked.connect(self._on_update_left_ylim)
        row_vis.addWidget(btn_ax_ylim)

        row_vis.addStretch(1)
        root.addLayout(row_vis)

        # FFT-specific controls (compact: direct x/y limits, no phase/centering)
        row_fft = QHBoxLayout()

        row_fft.addWidget(QLabel("FFT x:"))
        self._fft_xmin_edit = QLineEdit("0")
        self._fft_xmin_edit.setFixedWidth(60)
        row_fft.addWidget(self._fft_xmin_edit)
        self._fft_xmax_edit = QLineEdit("500")
        self._fft_xmax_edit.setFixedWidth(60)
        row_fft.addWidget(self._fft_xmax_edit)

        row_fft.addSpacing(15)
        row_fft.addWidget(QLabel("FFT y:"))
        self._ylim_min_edit = QLineEdit("0")
        self._ylim_min_edit.setFixedWidth(60)
        row_fft.addWidget(self._ylim_min_edit)
        self._ylim_max_edit = QLineEdit("2e5")
        self._ylim_max_edit.setFixedWidth(60)
        row_fft.addWidget(self._ylim_max_edit)

        row_fft.addSpacing(15)
        row_fft.addWidget(QLabel("Skip:"))
        self._downsample_edit = QLineEdit("5")
        self._downsample_edit.setFixedWidth(50)
        row_fft.addWidget(self._downsample_edit)

        btn_fft_lim = QPushButton("Set FFT lim")
        btn_fft_lim.clicked.connect(self._on_update_fft_lim)
        row_fft.addWidget(btn_fft_lim)

        row_fft.addStretch(1)
        root.addLayout(row_fft)

        # Comment + FTL threshold
        row4 = QHBoxLayout()
        self._comment_edit = QLineEdit("")
        self._thresh_edit = QLineEdit("0.02")
        row4.addWidget(QLabel("Comment:"))
        row4.addWidget(self._comment_edit, 1)
        row4.addWidget(QLabel("FTL thresh (0..1):"))
        row4.addWidget(self._thresh_edit)
        root.addLayout(row4)

        # Start/stop/save
        row3 = QHBoxLayout()
        self._start_btn = QPushButton("Start Live")
        self._stop_btn = QPushButton("Stop Live")
        self._stop_btn.setEnabled(False)
        self._save_btn = QPushButton("Save Spectrum")
        self._start_btn.clicked.connect(self._start_live)
        self._stop_btn.clicked.connect(self._stop_live)
        self._save_btn.clicked.connect(self._save_spectrum)
        row3.addWidget(self._start_btn)
        row3.addWidget(self._stop_btn)
        row3.addStretch(1)
        row3.addWidget(self._save_btn)
        root.addLayout(row3)

        # Figure with 2 panels
        self._figure, (self._ax, self._ax_fft) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [2, 1]}, figsize=(9, 3.2)
        )

        self._ax.set_xlabel("Wavelength (nm)")
        self._ax.set_ylabel("Counts")
        self._ax.grid(True)
        self._apply_left_ylim()

        self._ax_fft.set_xlabel("Index")
        self._ax_fft.grid(True)

        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            self._canvas.sizePolicy().Expanding, self._canvas.sizePolicy().Expanding
        )
        root.addWidget(self._canvas, 10)

        # A canvas resize invalidates any captured background.
        self._canvas.mpl_connect("resize_event", lambda _e: self._invalidate_blit())

        # Click handler for picking index from FFT
        self._mpl_cid_click = self._canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

        self._search_specs()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, message: str):
        if self._log:
            self._log.log(message, source=f"Avaspec/{self._preset.label}")

    # -------------------------------------------------------------------------
    # Spectrometer control
    # -------------------------------------------------------------------------

    def _search_specs(self):
        self._cmb.clear()
        self._handles = AvaspecController.list_spectrometers()
        if not self._handles:
            self._log_message("No spectrometers found.")
        else:
            self._log_message(f"Found {len(self._handles)} spectrometer(s).")
        for i in range(len(self._handles)):
            self._cmb.addItem(f"Spec {i + 1}")

    def _activate(self):
        if self._ctrl is not None:
            self._log_message("Already activated.")
            return

        idx = self._cmb.currentIndex()
        if idx < 0 or idx >= len(self._handles):
            QMessageBox.critical(self, "Error", "No spectrometer selected.")
            return

        try:
            self._ctrl = AvaspecController(self._handles[idx])
            self._ctrl.activate()
            self._apply_params()

            key = self._preset.registry_prefix
            try:
                for k, v in REGISTRY.items(prefix=self._preset.registry_prefix):
                    if k == key or v is self._ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass

            try:
                REGISTRY.register(key, self._ctrl)
                self._reg_key = key
            except Exception:
                self._reg_key = None

            self._log_message("Spectrometer activated.")
            self._activate_btn.setEnabled(False)
            self._deactivate_btn.setEnabled(True)
        except Exception as e:
            self._ctrl = None
            QMessageBox.critical(self, "Error", f"Activate failed: {e}")
            self._log_message(f"Activate failed: {e}")

    def _deactivate(self):
        if self._capture_thread is not None:
            QMessageBox.warning(self, "Warning", "Stop live first.")
            return

        try:
            if self._ctrl is not None:
                self._ctrl.deactivate()
        finally:
            try:
                if self._reg_key:
                    REGISTRY.unregister(self._reg_key)
            except Exception:
                pass
            self._reg_key = None
            self._ctrl = None
            self._activate_btn.setEnabled(True)
            self._deactivate_btn.setEnabled(False)
            self._log_message("Spectrometer deactivated.")

    def _apply_params(self):
        if self._ctrl is None:
            return
        try:
            it = float(self._int_edit.text())
            av = int(self._avg_edit.text())
            self._ctrl.set_params(it, av)
            if self._capture_thread is not None and self._capture_thread.isRunning():
                self._capture_thread.update_params(it, av)
            self._log_message(f"Params: int={it} ms, avg={av}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid params: {e}")

    def _start_live(self):
        if self._ctrl is None:
            QMessageBox.critical(self, "Error", "Activate first.")
            return

        if self._capture_thread is not None:
            self._stop_live()

        try:
            it = float(self._int_edit.text())
            av = int(self._avg_edit.text())
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid params.")
            return

        self._ref_wl = None
        self._last_draw = 0.0
        self._invalidate_blit()

        self._capture_thread = _MeasureThread(self._ctrl, it, av)
        self._capture_thread.data_ready.connect(self._on_data, Qt.QueuedConnection)
        self._capture_thread.error.connect(self._on_thread_error, Qt.QueuedConnection)
        self._capture_thread.finished.connect(
            self._on_thread_finished, Qt.QueuedConnection
        )
        self._capture_thread.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._deactivate_btn.setEnabled(False)
        self._log_message("Live started.")

    def _stop_live(self):
        th = self._capture_thread
        self._capture_thread = None

        if th is not None:
            try:
                th.stop()
                th.wait(2000)
            finally:
                try:
                    th.finished.disconnect(self._on_thread_finished)
                except Exception:
                    pass
                try:
                    th.error.disconnect(self._on_thread_error)
                except Exception:
                    pass
                try:
                    th.data_ready.disconnect(self._on_data)
                except Exception:
                    pass

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._ctrl is not None:
            self._deactivate_btn.setEnabled(True)
        self._log_message("Live stopped.")

    def _on_thread_finished(self):
        if self._capture_thread is None:
            self._log_message("Thread finished.")

    def _on_thread_error(self, err: str):
        self._log_message(f"Thread error: {err}")
        QMessageBox.critical(self, "Acquisition error", err)
        self._stop_live()

    # -------------------------------------------------------------------------
    # Blitting helpers
    # -------------------------------------------------------------------------

    def _invalidate_blit(self) -> None:
        """Force a full redraw + background recapture on the next frame."""
        self._blit_ready = False

    def _limits_changed(self) -> bool:
        """Return True if any axis limit differs from the last captured state."""
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        fxlim = self._ax_fft.get_xlim()
        fylim = self._ax_fft.get_ylim()
        changed = (
            xlim != self._last_xlim
            or ylim != self._last_ylim
            or fxlim != self._last_fft_xlim
            or fylim != self._last_fft_ylim
        )
        self._last_xlim = xlim
        self._last_ylim = ylim
        self._last_fft_xlim = fxlim
        self._last_fft_ylim = fylim
        return changed

    def _recapture_background(self) -> None:
        """Full draw of the static scene, then grab the pixel buffer.

        Both animated lines are hidden during the draw so they are NOT baked
        into the background; they get blitted on top afterwards.
        """
        line_vis = self._line.get_visible() if self._line is not None else None
        fft_vis = self._fft_line.get_visible() if self._fft_line is not None else None
        if self._line is not None:
            self._line.set_visible(False)
        if self._fft_line is not None:
            self._fft_line.set_visible(False)

        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._figure.bbox)

        if self._line is not None and line_vis is not None:
            self._line.set_visible(line_vis)
        if self._fft_line is not None and fft_vis is not None:
            self._fft_line.set_visible(fft_vis)

        self._blit_ready = True

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def _on_data(self, ts: float, wl, counts):
        now = time.monotonic()
        wl = np.asarray(wl, float).ravel()
        counts = np.asarray(counts, float).ravel()
        y = self._ctrl.process_counts(wl, counts) if self._ctrl else counts

        if self._ref_wl is None:
            self._ref_wl = wl.copy()
            self._ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])

        if wl.shape != self._ref_wl.shape or np.max(np.abs(wl - self._ref_wl)) > 1e-9:
            y = np.interp(self._ref_wl, wl, y)
            wl = self._ref_wl

        self._last_wl = wl
        self._last_counts = counts

        if (now - self._last_draw) < self._min_draw_dt:
            return
        self._last_draw = now

        disp = self._apply_threshold_for_display(y)
        ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp, level=0.5)
        ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
        self._lbl_ftl.setText(
            f"FTL: {ftl_fs:.0f} fs" if np.isfinite(ftl_fs) else "FTL: \u2014 fs"
        )

        # --- Left panel: spectrum line -------------------------------------
        if self._line is None:
            (self._line,) = self._ax.plot(wl, y, lw=1.8, animated=True)
            self._ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
            self._apply_left_ylim()
            self._invalidate_blit()
        else:
            self._line.set_data(wl, y)

        # Fixed Y limit (user-controlled), then apply the wavelength zoom.
        self._apply_zoom_window(wl)

        # --- Right panel: FFT amplitude line -------------------------------
        self._update_fft(wl, y)

        # --- Blit both panels ----------------------------------------------
        # If any limit moved (zoom, user change), the cached background is
        # stale: recapture it. Otherwise just restore + redraw the animated
        # lines. With a fixed left-Y limit, recapture only happens on zoom /
        # limit changes, so steady-state runs pure blit.
        if not self._blit_ready or self._limits_changed():
            self._recapture_background()

        self._canvas.restore_region(self._bg)
        if self._line is not None and self._line.get_visible():
            self._ax.draw_artist(self._line)
        if self._fft_line is not None and self._fft_line.get_visible():
            self._ax_fft.draw_artist(self._fft_line)
            if (
                self._fft_peak_vline is not None
                and self._fft_peak_vline.get_visible()
            ):
                self._ax_fft.draw_artist(self._fft_peak_vline)
        self._canvas.blit(self._figure.bbox)

    def _get_threshold_fraction(self) -> float:
        try:
            v = float(self._thresh_edit.text())
        except Exception:
            v = 0.02
        if not np.isfinite(v):
            v = 0.02
        return float(min(max(v, 0.0), 1.0))

    def _apply_threshold_for_display(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        y_pos = np.clip(y, 0, None)
        thr = self._get_threshold_fraction()
        ymax = float(np.nanmax(y_pos)) if y_pos.size else 0.0
        if not np.isfinite(ymax) or ymax <= 0:
            return np.zeros_like(y_pos)
        cutoff = thr * ymax
        return np.where(y_pos >= cutoff, y_pos, 0.0)

    def _apply_left_ylim(self) -> None:
        """Apply the user-set fixed Y limits to the counts (left) axis."""
        try:
            ymin = float(self._ax_ymin_edit.text())
            ymax = float(self._ax_ymax_edit.text())
            if ymin < ymax:
                self._ax.set_ylim(ymin, ymax)
        except Exception:
            pass

    def _on_update_left_ylim(self) -> None:
        """User-triggered left-Y limit update: invalidate the blit background."""
        self._apply_left_ylim()
        self._invalidate_blit()
        self._canvas.draw_idle()

    def _apply_zoom_window(self, wl: np.ndarray):
        if not self._chk_zoom.isChecked():
            if self._ref_wl is not None:
                self._ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
            return

        try:
            cwl = float(self._cwl_edit.text())
            pm = float(self._zoom_pm_edit.text())
        except Exception:
            if self._ref_wl is not None:
                self._ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
            return

        if pm <= 0:
            if self._ref_wl is not None:
                self._ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
            return

        wl = np.asarray(wl, float).ravel()
        if wl.size < 2:
            return

        wl_min, wl_max = float(wl[0]), float(wl[-1])
        x0 = max(wl_min, cwl - pm)
        x1 = min(wl_max, cwl + pm)
        if x1 <= x0:
            self._ax.set_xlim(wl_min, wl_max)
        else:
            self._ax.set_xlim(x0, x1)

    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------

    def _on_canvas_click(self, event):
        if event.button != 1 or event.inaxes is not self._ax_fft:
            return
        if self._fft_indices is None or self._fft_indices.size == 0:
            return
        if event.xdata is None:
            return

        x = float(event.xdata)
        idx = int(np.argmin(np.abs(self._fft_indices - x)))
        if idx < 0 or idx >= self._fft_indices.size:
            return
        self._set_peak_idx_internal(idx)

    def _set_peak_idx_internal(self, idx: int):
        self._fft_peak_idx = idx

        if self._fft_peak_vline is None:
            self._fft_peak_vline = self._ax_fft.axvline(
                idx, color="r", linestyle="--", linewidth=2.0, alpha=0.8,
                animated=True,
            )
        else:
            self._fft_peak_vline.set_xdata([idx, idx])

        self._log_message(f"Marked index {idx}")

    def _apply_fft_limits(self) -> None:
        """Apply direct x/y limits from the edit fields to the FFT axis."""
        try:
            xmin = float(self._fft_xmin_edit.text())
            xmax = float(self._fft_xmax_edit.text())
            if xmin < xmax:
                self._ax_fft.set_xlim(xmin, xmax)
        except Exception:
            pass
        try:
            ymin = float(self._ylim_min_edit.text())
            ymax = float(self._ylim_max_edit.text())
            if ymin < ymax:
                self._ax_fft.set_ylim(ymin, ymax)
        except Exception:
            pass

    def _on_update_fft_lim(self) -> None:
        """User-triggered FFT limit update: invalidate the blit background."""
        self._apply_fft_limits()
        self._invalidate_blit()
        self._canvas.draw_idle()

    def _update_fft(self, wl: np.ndarray, y: np.ndarray) -> None:
        """Update the FFT amplitude line (data only; drawing is done via blit)."""
        wl = np.asarray(wl, float).ravel()
        y = np.asarray(y, float).ravel()
        if wl.size < 4 or y.size != wl.size:
            return

        mag = np.abs(np.fft.rfft(y))
        indices = np.arange(mag.size)
        self._fft_indices = indices

        try:
            skip = max(1, int(self._downsample_edit.text()))
        except Exception:
            skip = 1

        x = indices[::skip]
        mag_plot = mag[::skip]

        if self._fft_line is None:
            (self._fft_line,) = self._ax_fft.plot(x, mag_plot, lw=1.8, animated=True)
            self._apply_fft_limits()
            self._invalidate_blit()
        else:
            self._fft_line.set_data(x, mag_plot)

    # -------------------------------------------------------------------------
    # FTL computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _fwxm(time_v: np.ndarray, intens: np.ndarray, x: float = 0.5) -> float:
        t = np.asarray(time_v, float).ravel()
        y = np.asarray(intens, float).ravel()
        if t.size != y.size or t.size < 3:
            return float("nan")
        m = float(np.nanmax(y)) if y.size else float("nan")
        if not np.isfinite(m) or m <= 0:
            return float("nan")
        P = y / m
        idx = np.where(P >= x)[0]
        if idx.size == 0:
            return float("nan")
        i0, i1 = int(idx[0]), int(idx[-1])
        if i0 == 0:
            t1 = t[0]
        else:
            m1 = (P[i0] - P[i0 - 1]) / (
                t[i0] - t[i0 - 1] if t[i0] != t[i0 - 1] else 1e-300
            )
            n1 = P[i0] - m1 * t[i0]
            t1 = (x - n1) / m1
        if i1 >= t.size - 1:
            t2 = t[-1]
        else:
            m2 = (P[i1 + 1] - P[i1]) / (
                t[i1 + 1] - t[i1] if t[i1 + 1] != t[i1] else 1e-300
            )
            n2 = P[i1] - m2 * t[i1]
            t2 = (x - n2) / m2
        return float(t2 - t1)

    @staticmethod
    def _compute_ftl_from_spectrum(
        wl_nm: np.ndarray, spec: np.ndarray, level: float = 0.5
    ):
        c = 299_792_458.0
        wl = np.asarray(wl_nm, float).ravel()
        S = np.asarray(spec, float).ravel()
        if wl.size != S.size or wl.size < 3:
            return float("nan"), np.array([]), np.array([])
        freqs = c / (wl * 1e-9)
        N = freqs.size
        freqs_i = np.linspace(freqs[-1], freqs[0], N)
        specs_f = np.abs(np.interp(freqs_i, np.flip(freqs), np.flip(S)))
        if N < 2:
            return float("nan"), np.array([]), np.array([])
        dnu = freqs_i[1] - freqs_i[0]
        if not np.isfinite(dnu) or dnu == 0:
            return float("nan"), np.array([]), np.array([])
        amp_f = np.sqrt(np.clip(specs_f, 0, None))
        E_t = np.fft.fftshift(np.fft.ifft(amp_f))
        I_t = np.abs(E_t) ** 2
        time_v = np.fft.fftshift(np.fft.fftfreq(N, d=dnu))
        ftl_s = AvaspecLiveWindow._fwxm(time_v, I_t, x=level)
        I_t_n = I_t / (np.max(I_t) if np.max(I_t) > 0 else 1.0)
        return ftl_s, time_v, I_t_n

    # -------------------------------------------------------------------------
    # Background / calibration
    # -------------------------------------------------------------------------

    def _set_background(self):
        if self._last_wl is None or self._last_counts is None or self._ctrl is None:
            self._log_message("No spectrum to set as BG.")
            return
        try:
            self._ctrl.set_background(self._last_wl, self._last_counts)
            self._log_message("Background set.")
        except Exception as e:
            self._log_message(f"Set BG failed: {e}")
            QMessageBox.warning(self, "Background", f"Failed: {e}")

    def _clear_background(self):
        if self._ctrl is None:
            return
        try:
            self._ctrl.clear_background()
            self._log_message("Background cleared.")
        except Exception as e:
            self._log_message(f"Clear BG failed: {e}")
            QMessageBox.warning(self, "Background", f"Failed: {e}")

    def _toggle_calibration(self):
        enabled = self._btn_cal_toggle.isChecked()
        self._btn_cal_toggle.setText(
            "Calibration: ON" if enabled else "Calibration: OFF"
        )
        if self._ctrl is None:
            return
        try:
            self._ctrl.enable_calibration(enabled)
            self._log_message(f"Calibration {'enabled' if enabled else 'disabled'}.")
        except Exception as e:
            self._btn_cal_toggle.setChecked(False)
            self._btn_cal_toggle.setText("Calibration: OFF")
            self._log_message(f"Calibration failed: {e}")
            QMessageBox.warning(self, "Calibration", f"Failed: {e}")

    # -------------------------------------------------------------------------
    # Save spectrum
    # -------------------------------------------------------------------------

    def _get_save_directory(self) -> tuple[Path, datetime.datetime]:
        now = datetime.datetime.now()
        dir_path = data_dir() / now.strftime("%Y-%m-%d") / "avaspec"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path, now

    def _write_log_entry(
        self, log_path: Path, filename: str, int_ms: float, averages: int, comment: str
    ):
        header = "File Name\tIntegration_ms\tAverages\tComment\n"
        if not log_path.exists():
            log_path.write_text(header, encoding="utf-8")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{int_ms}\t{averages}\t{comment}\n")

    def _save_spectrum(self):
        if self._last_wl is None or self._last_counts is None or self._ctrl is None:
            QMessageBox.warning(self, "Save", "No spectrum to save.")
            return

        dir_path, now = self._get_save_directory()
        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = f"Avaspec_{self._instance_id}"
        filename = f"{base}_Spectrum_{safe_ts}.txt"
        filepath = dir_path / filename
        comment = self._comment_edit.text() or ""

        try:
            wl = np.asarray(self._last_wl, float)
            raw = np.asarray(self._last_counts, float)
            bgsub = self._ctrl._apply_background(wl, raw)
            use_cal = bool(self._ctrl._cal_enabled)
            divcal = None
            cal_applied = False
            cal_path_str = ""

            if use_cal:
                cal_applied = (
                    self._ctrl._cal_wl is not None and self._ctrl._cal_vals is not None
                )
                if cal_applied:
                    cal_path = (
                        get_config().get("avaspec", {}).get("calibration_file", "")
                    )
                    cal_path_str = str((ROOT / cal_path).resolve()) if cal_path else ""
                    divcal = self._ctrl._apply_calibration(wl, bgsub)

            thr = self._get_threshold_fraction()
            disp_for_ftl = np.where(
                bgsub >= thr * max(1.0, float(np.nanmax(np.clip(bgsub, 0, None)))),
                bgsub,
                0.0,
            )
            ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp_for_ftl, level=0.5)
            ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")

            with open(filepath, "w", encoding="utf-8") as f:
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
                f.write(f"# Instance: {self._instance_id} ({self._preset.label})\n")
                f.write(f"# IntegrationTime_ms: {self._int_edit.text()}\n")
                f.write(f"# Averages: {self._avg_edit.text()}\n")
                f.write(f"# BackgroundApplied: {self._ctrl._bg_counts is not None}\n")
                f.write(f"# CalibrationApplied: {bool(cal_applied)}\n")
                if cal_applied and cal_path_str:
                    f.write(f"# CalibrationFile: {cal_path_str}\n")
                f.write(f"# DisplayThreshold: {thr:.4f}\n")
                if np.isfinite(ftl_fs):
                    f.write(f"# FTL_fs: {ftl_fs:.3f}\n")

                if cal_applied and divcal is not None:
                    f.write(
                        "Wavelength_nm;Counts_raw;Counts_bgsub;Counts_bgsub_divcal\n"
                    )
                    for x, y_raw, y_bg, y_cal in zip(wl, raw, bgsub, divcal):
                        f.write(
                            f"{float(x):.6f};{float(y_raw):.6f};"
                            f"{float(y_bg):.6f};{float(y_cal):.6f}\n"
                        )
                else:
                    f.write("Wavelength_nm;Counts_raw;Counts_bgsub\n")
                    for x, y_raw, y_bg in zip(wl, raw, bgsub):
                        f.write(
                            f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f}\n"
                        )

            # Write log entry
            try:
                int_ms = float(self._int_edit.text())
            except Exception:
                int_ms = float("nan")
            try:
                averages = int(self._avg_edit.text())
            except Exception:
                averages = 1

            log_filename = f"{base}_log_{now.strftime('%Y-%m-%d')}.log"
            log_path = dir_path / log_filename
            self._write_log_entry(log_path, filename, int_ms, averages, comment)

            self._log_message(f"Spectrum saved to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
            self._log_message(f"Save failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _safe_shutdown(self):
        if self._capture_thread is not None:
            try:
                self._log_message("Stopping live thread...")
                th = self._capture_thread
                self._capture_thread = None
                th.stop()
                th.wait(2000)
                try:
                    th.finished.disconnect(self._on_thread_finished)
                except Exception:
                    pass
                try:
                    th.error.disconnect(self._on_thread_error)
                except Exception:
                    pass
                try:
                    th.data_ready.disconnect(self._on_data)
                except Exception:
                    pass
            except Exception as e:
                self._log_message(f"Error stopping thread: {e}")

        try:
            if self._ctrl is not None:
                self._log_message("Deactivating spectrometer...")
                try:
                    self._ctrl.deactivate()
                except Exception as e:
                    self._log_message(f"Deactivate failed: {e}")
        finally:
            try:
                if self._reg_key:
                    REGISTRY.unregister(self._reg_key)
            except Exception:
                pass
            self._reg_key = None
            self._ctrl = None

        try:
            if self._mpl_cid_click is not None:
                self._canvas.mpl_disconnect(self._mpl_cid_click)
        except Exception:
            pass

        try:
            if self._ax is not None:
                self._ax.cla()
            if self._ax_fft is not None:
                self._ax_fft.cla()
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

        try:
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._deactivate_btn.setEnabled(self._ctrl is not None)
        except Exception:
            pass

        self._log_message("AvaspecLive shutdown complete.")

    def closeEvent(self, event):
        try:
            self._safe_shutdown()
        finally:
            try:
                self.closed.emit()
            except Exception:
                pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AvaspecLiveWindow(instance_id="w")
    gui.show()
    sys.exit(app.exec_())