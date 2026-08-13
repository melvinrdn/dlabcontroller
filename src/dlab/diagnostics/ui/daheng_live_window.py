from __future__ import annotations

import sys
import time
import datetime
import threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from PIL import Image, PngImagePlugin

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QMessageBox, QSplitter, QCheckBox, QSpinBox, QGroupBox,
    QShortcut, QComboBox
)
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from dlab.hardware.wrappers.daheng_controller import (
    DahengController, DahengControllerError,
    DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US,
    DEFAULT_GAIN, MIN_GAIN, MAX_GAIN
)
from dlab.core.device_registry import REGISTRY
from dlab.utils.paths_utils import data_dir
from dlab.utils.log_panel import LogPanel
from dlab.utils.yaml_utils import read_yaml, write_yaml
from dlab.utils.colormaps import COLORMAPS, resolve_cmap
from dlab.boot import ROOT

PIXEL_SIZE_M = 3.45e-6
MIN_INTERVAL_US = 500_000
SENSOR_MAX_VALUE = 255  # Daheng cameras are always run in MONO8


def _config_path() -> Path:
    return ROOT / "config" / "config.yaml"


class _LiveCaptureThread(QThread):
    """Background thread for continuous image capture."""

    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam: DahengController, exposure_us: int, gain: int, interval_us: int, cap_lock=None):
        super().__init__()
        self._cam = cam
        self._exposure_us = exposure_us
        self._gain = gain
        self._interval_s = interval_us / 1e6
        self._running = True
        self._lock = threading.Lock()
        self._cap_lock = cap_lock

    def update_parameters(self, exposure_us: int, gain: int, interval_us: int):
        with self._lock:
            self._exposure_us = exposure_us
            self._gain = gain
            self._interval_s = interval_us / 1e6

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                with self._lock:
                    exp = self._exposure_us
                    g = self._gain
                    wait_s = self._interval_s

                lock = self._cap_lock or threading.Lock()
                with lock:
                    frame = self._cam.capture_single(exp, g)
                self.image_signal.emit(frame)
                time.sleep(wait_s)
            except Exception:
                break


class DahengLiveWindow(QWidget):
    """Live view window for Daheng camera."""

    closed = pyqtSignal()
    gui_update_image = pyqtSignal(object)
    gui_log = pyqtSignal(str)

    def __init__(self, camera_name: str = "Daheng", fixed_index: int = 1, log_panel: LogPanel | None = None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._camera_name = camera_name
        self._fixed_index = fixed_index
        self._log = log_panel

        self.setWindowTitle(f"DahengLiveWindow - {camera_name}")

        self._cam: DahengController | None = None
        self._capture_thread: _LiveCaptureThread | None = None
        self._capture_lock = threading.Lock()
        self._live_running = False

        self._last_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Plot state
        self._image_artist = None
        self._cbar = None
        self._fix_cbar = False
        self._fixed_vmax: float | None = None
        self._cmap_key = "cmr.rainforest"
        self._cmap = resolve_cmap(self._cmap_key)

        # ROI state
        self._roi_px: tuple[int, int, int, int] | None = None
        self._roi_artist = None
        self._rect_selector = None

        # Crosshair 1 state
        self._crosshair_visible = False
        self._crosshair_locked = False
        self._crosshair_pos_mm: tuple[float, float] | None = None
        self._ch_h = None
        self._ch_v = None

        # Crosshair 2 state
        self._crosshair2_visible = False
        self._crosshair2_locked = False
        self._crosshair2_pos_mm: tuple[float, float] | None = None
        self._ch2_h = None
        self._ch2_v = None

        self._mpl_cid_click = None
        self._mpl_cid_motion = None

        self._init_ui()
        self.gui_update_image.connect(self._update_image, Qt.QueuedConnection)
        self.gui_log.connect(self._log_message, Qt.QueuedConnection)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left panel - parameters
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Camera index
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        idx_layout.addWidget(QLabel(str(self._fixed_index)))
        param_layout.addLayout(idx_layout)

        # Exposure
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self._exposure_edit = QLineEdit(str(DEFAULT_EXPOSURE_US))
        self._exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self._exposure_edit.textChanged.connect(self._on_params_changed)
        exp_layout.addWidget(self._exposure_edit)
        param_layout.addLayout(exp_layout)

        # Update interval
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (µs):"))
        init_interval = max(DEFAULT_EXPOSURE_US, MIN_INTERVAL_US)
        self._interval_edit = QLineEdit(str(init_interval))
        self._interval_edit.setEnabled(False)
        int_layout.addWidget(self._interval_edit)
        param_layout.addLayout(int_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self._gain_edit = QLineEdit(str(DEFAULT_GAIN))
        self._gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self._gain_edit.textChanged.connect(self._on_params_changed)
        gain_layout.addWidget(self._gain_edit)
        param_layout.addLayout(gain_layout)

        # Comment
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self._comment_edit = QLineEdit()
        comment_layout.addWidget(self._comment_edit)
        param_layout.addLayout(comment_layout)

        # Frames to save
        nsave_layout = QHBoxLayout()
        nsave_layout.addWidget(QLabel("Frames to Save:"))
        self._frames_to_save_edit = QLineEdit("1")
        self._frames_to_save_edit.setValidator(QIntValidator(1, 1000, self))
        nsave_layout.addWidget(self._frames_to_save_edit)
        param_layout.addLayout(nsave_layout)

        # Background checkbox
        self._background_cb = QCheckBox("Background")
        param_layout.addWidget(self._background_cb)

        # Colorbar options
        self._fix_cbar_cb = QCheckBox("Fix Colorbar Max")
        self._fix_cbar_cb.toggled.connect(self._on_fix_cbar)
        param_layout.addWidget(self._fix_cbar_cb)

        self._fix_value_edit = QLineEdit("10000")
        self._fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self._fix_value_edit.setEnabled(False)
        self._fix_value_edit.textChanged.connect(self._on_fix_value_changed)
        self._fix_cbar_cb.toggled.connect(self._fix_value_edit.setEnabled)
        param_layout.addWidget(self._fix_value_edit)

        self._sensor_max_btn = QPushButton("Set Colorbar to Sensor Max")
        self._sensor_max_btn.setToolTip(
            "Pin the colorbar max to the sensor's true saturation level, "
            "the only case where the saturation colormap is meaningful."
        )
        self._sensor_max_btn.clicked.connect(self._on_set_sensor_max)
        param_layout.addWidget(self._sensor_max_btn)

        # Colormap
        cmap_group = QGroupBox("Colormap")
        cmap_layout = QHBoxLayout(cmap_group)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.setCurrentText(self._cmap_key)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_layout.addWidget(QLabel("Map:"))
        cmap_layout.addWidget(self._cmap_combo)
        param_layout.addWidget(cmap_group)

        # ROI controls
        roi_group = QGroupBox("ROI")
        roi_layout = QVBoxLayout(roi_group)

        roi_row1 = QHBoxLayout()
        self._use_roi_cb = QCheckBox("Use ROI")
        self._preview_roi_cb = QCheckBox("Preview crop")
        roi_row1.addWidget(self._use_roi_cb)
        roi_row1.addWidget(self._preview_roi_cb)
        roi_layout.addLayout(roi_row1)

        roi_row2 = QHBoxLayout()
        roi_row2.addWidget(QLabel("W(px):"))
        self._center_w_spin = QSpinBox()
        self._center_w_spin.setRange(4, 4096)
        self._center_w_spin.setSingleStep(10)
        self._center_w_spin.setValue(200)
        roi_row2.addWidget(self._center_w_spin)
        roi_row2.addWidget(QLabel("H(px):"))
        self._center_h_spin = QSpinBox()
        self._center_h_spin.setRange(4, 4096)
        self._center_h_spin.setSingleStep(10)
        self._center_h_spin.setValue(200)
        roi_row2.addWidget(self._center_h_spin)
        roi_layout.addLayout(roi_row2)

        roi_row3 = QHBoxLayout()
        btn_center = QPushButton("Center on max")
        btn_center.clicked.connect(self._center_on_max)
        btn_set_roi = QPushButton("Set ROI")
        btn_set_roi.clicked.connect(self._start_roi_selection)
        btn_clear_roi = QPushButton("Clear ROI")
        btn_clear_roi.clicked.connect(self._clear_roi)
        roi_row3.addWidget(btn_center)
        roi_row3.addWidget(btn_set_roi)
        roi_row3.addWidget(btn_clear_roi)
        roi_layout.addLayout(roi_row3)
        param_layout.addWidget(roi_group)

        # Crosshair controls
        ch_grp = QGroupBox("Crosshairs")
        ch_layout = QVBoxLayout(ch_grp)

        # Crosshair 1
        ch_row1 = QHBoxLayout()
        btn_ch_toggle = QPushButton("X1 Toggle (Shift+C)")
        btn_ch_toggle.clicked.connect(self._toggle_crosshair)
        btn_ch_lock = QPushButton("X1 Lock/Unlock")
        btn_ch_lock.clicked.connect(self._toggle_lock_manual)
        btn_ch_save = QPushButton("X1 Save")
        btn_ch_save.clicked.connect(self._save_crosshair_position)
        btn_ch_goto = QPushButton("X1 Load")
        btn_ch_goto.clicked.connect(self._goto_saved_crosshair)
        ch_row1.addWidget(btn_ch_toggle)
        ch_row1.addWidget(btn_ch_lock)
        ch_row1.addWidget(btn_ch_save)
        ch_row1.addWidget(btn_ch_goto)
        ch_layout.addLayout(ch_row1)

        # Crosshair 2
        ch_row2 = QHBoxLayout()
        btn_ch2_toggle = QPushButton("X2 Toggle (Ctrl+D)")
        btn_ch2_toggle.clicked.connect(self._toggle_crosshair2)
        btn_ch2_lock = QPushButton("X2 Lock/Unlock")
        btn_ch2_lock.clicked.connect(self._toggle_lock_manual2)
        btn_ch2_save = QPushButton("X2 Save")
        btn_ch2_save.clicked.connect(self._save_crosshair2_position)
        btn_ch2_goto = QPushButton("X2 Load")
        btn_ch2_goto.clicked.connect(self._goto_saved_crosshair2)
        ch_row2.addWidget(btn_ch2_toggle)
        ch_row2.addWidget(btn_ch2_lock)
        ch_row2.addWidget(btn_ch2_save)
        ch_row2.addWidget(btn_ch2_goto)
        ch_layout.addLayout(ch_row2)

        param_layout.addWidget(ch_grp)

        # Camera control buttons
        btn_layout = QVBoxLayout()
        self._activate_btn = QPushButton("Activate Camera")
        self._activate_btn.clicked.connect(self._activate_camera)
        btn_layout.addWidget(self._activate_btn)

        self._deactivate_btn = QPushButton("Deactivate Camera")
        self._deactivate_btn.clicked.connect(self._deactivate_camera)
        self._deactivate_btn.setEnabled(False)
        btn_layout.addWidget(self._deactivate_btn)

        self._start_btn = QPushButton("Start Live Capture")
        self._start_btn.clicked.connect(self._start_capture)
        self._start_btn.setEnabled(False)
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop Live Capture")
        self._stop_btn.clicked.connect(self._stop_capture)
        self._stop_btn.setEnabled(False)
        btn_layout.addWidget(self._stop_btn)

        save_btn = QPushButton("Save Frame(s) (Ctrl+S)")
        save_btn.clicked.connect(self._save_frames)
        btn_layout.addWidget(save_btn)
        param_layout.addLayout(btn_layout)

        param_layout.addStretch()
        splitter.addWidget(param_panel)

        # Right panel - plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self._figure, self._ax = plt.subplots()
        self._ax.set_title(f"DahengLiveWindow - {self._camera_name}")
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._figure.subplots_adjust(right=0.85)

        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas)
        splitter.addWidget(plot_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)
        self.resize(1080, 720)

        # Mouse events
        self._mpl_cid_click = self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._mpl_cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # Shortcuts
        QShortcut(QKeySequence("Shift+C"), self).activated.connect(self._toggle_crosshair)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self._toggle_crosshair2)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_frames)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, message: str):
        if self._log:
            self._log.log(message, source="Daheng")

    # -------------------------------------------------------------------------
    # Camera control
    # -------------------------------------------------------------------------

    def _activate_camera(self):
        if self._cam:
            try:
                self._cam.deactivate()
            except Exception:
                pass
            self._cam = None

        try:
            self._cam = DahengController(self._fixed_index)
            self._cam.activate()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self._log_message(f"Activation error: {e}")
            return

        self._log_message(f"Camera {self._fixed_index} activated")

        key_name = f"camera:daheng:{self._camera_name.lower()}"
        key_index = f"camera:daheng:index:{self._fixed_index}"
        for k in (key_name, key_index):
            prev = REGISTRY.get(k)
            if prev and prev is not self._cam:
                self._log_message(f"Registry key '{k}' already in use. Replacing.")
            REGISTRY.register(k, self)

        self._activate_btn.setEnabled(False)
        self._deactivate_btn.setEnabled(True)
        self._start_btn.setEnabled(True)

    def _deactivate_camera(self):
        if self._cam:
            try:
                self._cam.deactivate()
            except Exception:
                pass
            self._log_message(f"Camera {self._fixed_index} deactivated")

        for k in (
            f"camera:daheng:{self._camera_name.lower()}",
            f"camera:daheng:index:{self._fixed_index}",
        ):
            REGISTRY.unregister(k)

        self._cam = None
        self._activate_btn.setEnabled(True)
        self._deactivate_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)

    def _start_capture(self):
        if self._cam is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return

        try:
            exp_us = int(self._exposure_edit.text())
            gain = int(self._gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return

        self._capture_thread = _LiveCaptureThread(
            self._cam, exp_us, gain, interval_us, cap_lock=self._capture_lock
        )
        self._capture_thread.image_signal.connect(self._update_image)
        self._capture_thread.start()
        self._live_running = True

        self._log_message("Live capture started.")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop_capture(self):
        if self._capture_thread:
            self._capture_thread.stop()
            self._capture_thread.wait()
            self._capture_thread = None
            self._live_running = False

        self._log_message("Live capture stopped.")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _update_interval_field(self, exp_us: int) -> int:
        interval_us = exp_us if exp_us >= MIN_INTERVAL_US else MIN_INTERVAL_US
        self._interval_edit.setText(str(interval_us))
        return interval_us

    def _on_params_changed(self):
        try:
            exp_us = int(self._exposure_edit.text())
            gain = int(self._gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
            if self._capture_thread:
                self._capture_thread.update_parameters(exp_us, gain, interval_us)
            elif self._cam:
                self._cam.set_exposure(exp_us)
                self._cam.set_gain(gain)
        except ValueError:
            pass

    # -------------------------------------------------------------------------
    # Image display
    # -------------------------------------------------------------------------

    def _px_per_mm(self) -> float:
        return 1.0 / (PIXEL_SIZE_M * 1e3)

    def _update_image(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            self._log_message("Invalid image received")
            return

        with self._frame_lock:
            self._last_frame = image

        disp = image
        extent_mm = None

        # Apply ROI crop for preview
        if self._preview_roi_cb.isChecked() and self._roi_px is not None and self._use_roi_cb.isChecked():
            x0, y0, x1, y1 = self._roi_px
            h0, w0 = disp.shape
            x0 = max(0, min(w0 - 1, x0))
            x1 = max(1, min(w0, x1))
            y0 = max(0, min(h0 - 1, y0))
            y1 = max(1, min(h0, y1))
            disp = disp[y0:y1, x0:x1]
            mm_per_px = PIXEL_SIZE_M * 1e3
            extent_mm = [x0 * mm_per_px, x1 * mm_per_px, y0 * mm_per_px, y1 * mm_per_px]

        vmin, vmax = float(disp.min()), float(disp.max())
        h, w = disp.shape

        if extent_mm is None:
            x_mm = w * (PIXEL_SIZE_M * 1e3)
            y_mm = h * (PIXEL_SIZE_M * 1e3)
            extent = [0, x_mm, 0, y_mm]
        else:
            extent = extent_mm

        if self._image_artist is None:
            self._ax.clear()
            self._image_artist = self._ax.imshow(
                disp, cmap=self._cmap, vmin=vmin, vmax=vmax,
                extent=extent, origin="lower", aspect="equal"
            )
            self._ax.set_xlabel("X (mm)")
            self._ax.set_ylabel("Y (mm)")
            self._cbar = self._figure.colorbar(
                self._image_artist, ax=self._ax, fraction=0.046, pad=0.04
            )
        else:
            self._image_artist.set_data(disp)
            if self._fix_cbar and self._fixed_vmax is not None:
                self._image_artist.set_clim(vmin, self._fixed_vmax)
            else:
                self._image_artist.set_clim(vmin, vmax)
            self._image_artist.set_extent(extent)
            self._image_artist.set_cmap(self._cmap)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)

        # Draw ROI overlay if not previewing crop
        if not (self._preview_roi_cb.isChecked() and self._use_roi_cb.isChecked()):
            self._draw_roi_overlay()

        # Refresh crosshairs
        if self._crosshair_visible:
            self._refresh_crosshair()
        if self._crosshair2_visible:
            self._refresh_crosshair2()

        self._canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Colorbar controls
    # -------------------------------------------------------------------------

    def _on_fix_cbar(self, checked: bool):
        self._fix_cbar = checked
        if checked:
            try:
                self._fixed_vmax = float(self._fix_value_edit.text())
                self._log_message(f"Colorbar max set to {self._fixed_vmax:.1f}")
            except ValueError:
                self._fixed_vmax = None
                self._fix_cbar_cb.setChecked(False)
                self._log_message("Invalid colorbar max")
        else:
            self._fixed_vmax = None
            self._log_message("Colorbar auto scale")

    def _on_fix_value_changed(self, text: str):
        if not self._fix_cbar_cb.isChecked() or not self._image_artist:
            return
        try:
            self._fixed_vmax = float(text)
            vmin, _ = self._image_artist.get_clim()
            self._image_artist.set_clim(vmin, self._fixed_vmax)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        except ValueError:
            pass

    def _on_set_sensor_max(self):
        self._fix_value_edit.setText(str(SENSOR_MAX_VALUE))
        if not self._fix_cbar_cb.isChecked():
            self._fix_cbar_cb.setChecked(True)

    def _on_cmap_changed(self, key: str):
        self._cmap_key = key
        self._cmap = resolve_cmap(key)
        if self._image_artist is not None:
            self._image_artist.set_cmap(self._cmap)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        self._log_message(f"Colormap set to {key}")

    # -------------------------------------------------------------------------
    # ROI
    # -------------------------------------------------------------------------

    def _start_roi_selection(self):
        with self._frame_lock:
            if self._last_frame is None:
                self._log_message("No image yet — start live or capture one to set ROI.")
                return

        if self._rect_selector is not None:
            try:
                self._rect_selector.disconnect_events()
            except Exception:
                pass
            self._rect_selector = None

        def _on_select(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return

            px_per_mm = self._px_per_mm()
            x0_mm, y0_mm = eclick.xdata, eclick.ydata
            x1_mm, y1_mm = erelease.xdata, erelease.ydata

            x0 = int(np.floor(min(x0_mm, x1_mm) * px_per_mm))
            x1 = int(np.ceil(max(x0_mm, x1_mm) * px_per_mm))
            y0 = int(np.floor(min(y0_mm, y1_mm) * px_per_mm))
            y1 = int(np.ceil(max(y0_mm, y1_mm) * px_per_mm))

            with self._frame_lock:
                h, w = self._last_frame.shape

            x0 = max(0, min(w - 1, x0))
            x1 = max(1, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(1, min(h, y1))

            if x1 <= x0 or y1 <= y0:
                self._log_message("Invalid ROI")
                return

            self._roi_px = (x0, y0, x1, y1)
            self._draw_roi_overlay()
            self._log_message(f"ROI set: x={x0}:{x1}, y={y0}:{y1} (px)")

            if self._rect_selector:
                self._rect_selector.set_visible(False)
                self._canvas.draw_idle()

        self._rect_selector = RectangleSelector(
            self._ax, _on_select, useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords="pixels", interactive=True
        )

    def _draw_roi_overlay(self):
        try:
            if self._roi_artist is not None:
                self._roi_artist.remove()
                self._roi_artist = None
        except Exception:
            pass

        if self._roi_px is None:
            self._canvas.draw_idle()
            return

        x0, y0, x1, y1 = self._roi_px
        mm_per_px = PIXEL_SIZE_M * 1e3
        x0_mm, x1_mm = x0 * mm_per_px, x1 * mm_per_px
        y0_mm, y1_mm = y0 * mm_per_px, y1 * mm_per_px

        self._roi_artist = self._ax.add_patch(
            Rectangle(
                (x0_mm, y0_mm), (x1_mm - x0_mm), (y1_mm - y0_mm),
                fill=False, linewidth=1.5, linestyle="--"
            )
        )
        self._canvas.draw_idle()

    def _clear_roi(self):
        self._roi_px = None
        self._draw_roi_overlay()
        self._log_message("ROI cleared")

    def _center_on_max(self):
        with self._frame_lock:
            if self._last_frame is None or self._last_frame.size == 0:
                self._log_message("No image to center on.")
                return
            frame = self._last_frame
            h, w = frame.shape
            idx = int(np.argmax(frame))

        y_max, x_max = divmod(idx, w)
        rw = int(self._center_w_spin.value())
        rh = int(self._center_h_spin.value())
        rw = max(4, min(w, rw))
        rh = max(4, min(h, rh))

        half_w = rw // 2
        half_h = rh // 2
        x0 = max(0, x_max - half_w)
        y0 = max(0, y_max - half_h)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
        x0 = max(0, x1 - rw)
        y0 = max(0, y1 - rh)

        if x1 <= x0 or y1 <= y0:
            self._log_message("Center ROI failed due to size.")
            return

        self._roi_px = (int(x0), int(y0), int(x1), int(y1))
        self._draw_roi_overlay()
        self._use_roi_cb.setChecked(True)
        self._preview_roi_cb.setChecked(True)
        self._log_message(f"ROI centered on max at ({x_max},{y_max}), size=({rw}x{rh})")

    # -------------------------------------------------------------------------
    # Crosshairs
    # -------------------------------------------------------------------------

    def _ensure_crosshair_artists(self):
        if self._ch_h is None or self._ch_v is None:
            self._ch_h = self._ax.axhline(0, linestyle="--", linewidth=1.2, color="r")
            self._ch_v = self._ax.axvline(0, linestyle="--", linewidth=1.2, color="r")
            self._ch_h.set_visible(self._crosshair_visible)
            self._ch_v.set_visible(self._crosshair_visible)

    def _ensure_crosshair2_artists(self):
        if self._ch2_h is None or self._ch2_v is None:
            self._ch2_h = self._ax.axhline(0, linestyle="-.", linewidth=1.2, color="green")
            self._ch2_v = self._ax.axvline(0, linestyle="-.", linewidth=1.2, color="green")
            self._ch2_h.set_visible(self._crosshair2_visible)
            self._ch2_v.set_visible(self._crosshair2_visible)

    def _toggle_crosshair(self):
        if not self._crosshair_visible and self._crosshair_pos_mm is None:
            with self._frame_lock:
                if self._last_frame is not None:
                    h, w = self._last_frame.shape
                    mm_per_px = PIXEL_SIZE_M * 1e3
                    cx = (w * mm_per_px) / 2.0
                    cy = (h * mm_per_px) / 2.0
                    self._crosshair_pos_mm = (cx, cy)
                else:
                    self._crosshair_pos_mm = (0.0, 0.0)

        self._crosshair_visible = not self._crosshair_visible
        self._ensure_crosshair_artists()
        self._refresh_crosshair()
        self._log_message(f"Crosshair 1 {'shown' if self._crosshair_visible else 'hidden'}")

    def _toggle_crosshair2(self):
        if not self._crosshair2_visible and self._crosshair2_pos_mm is None:
            with self._frame_lock:
                if self._last_frame is not None:
                    h, w = self._last_frame.shape
                    mm_per_px = PIXEL_SIZE_M * 1e3
                    cx = (w * mm_per_px) / 2.0
                    cy = (h * mm_per_px) / 2.0
                    self._crosshair2_pos_mm = (cx, cy)
                else:
                    self._crosshair2_pos_mm = (0.0, 0.0)

        self._crosshair2_visible = not self._crosshair2_visible
        self._ensure_crosshair2_artists()
        self._refresh_crosshair2()
        self._log_message(f"Crosshair 2 {'shown' if self._crosshair2_visible else 'hidden'}")

    def _toggle_lock_manual(self):
        if not self._crosshair_visible:
            return
        self._crosshair_locked = not self._crosshair_locked
        self._refresh_crosshair()

    def _toggle_lock_manual2(self):
        if not self._crosshair2_visible:
            return
        self._crosshair2_locked = not self._crosshair2_locked
        self._refresh_crosshair2()

    def _refresh_crosshair(self):
        self._ensure_crosshair_artists()
        vis = bool(self._crosshair_visible)
        self._ch_h.set_visible(vis)
        self._ch_v.set_visible(vis)
        if vis and self._crosshair_pos_mm is not None:
            x_mm, y_mm = self._crosshair_pos_mm
            self._ch_h.set_ydata([y_mm, y_mm])
            self._ch_v.set_xdata([x_mm, x_mm])
        self._canvas.draw_idle()

    def _refresh_crosshair2(self):
        self._ensure_crosshair2_artists()
        vis = bool(self._crosshair2_visible)
        self._ch2_h.set_visible(vis)
        self._ch2_v.set_visible(vis)
        if vis and self._crosshair2_pos_mm is not None:
            x_mm, y_mm = self._crosshair2_pos_mm
            self._ch2_h.set_ydata([y_mm, y_mm])
            self._ch2_v.set_xdata([x_mm, x_mm])
        self._canvas.draw_idle()

    def _save_crosshair_position(self):
        if not self._crosshair_visible or self._crosshair_pos_mm is None:
            QMessageBox.warning(self, "Crosshair", "Crosshair 1 must be visible to save its position.")
            return

        path = _config_path()
        data = read_yaml(path)
        cam_key = f"daheng_{self._fixed_index}"
        node = data.get("crosshair", {}) if isinstance(data.get("crosshair"), dict) else {}
        node[cam_key] = {
            "x_mm": float(self._crosshair_pos_mm[0]),
            "y_mm": float(self._crosshair_pos_mm[1]),
        }
        data["crosshair"] = node

        try:
            write_yaml(path, data)
            self._log_message(f"Crosshair 1 saved: ({self._crosshair_pos_mm[0]:.3f}, {self._crosshair_pos_mm[1]:.3f}) mm")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair 1: {e}")

    def _save_crosshair2_position(self):
        if not self._crosshair2_visible or self._crosshair2_pos_mm is None:
            QMessageBox.warning(self, "Crosshair 2", "Crosshair 2 must be visible to save its position.")
            return

        path = _config_path()
        data = read_yaml(path)
        cam_key = f"daheng_{self._fixed_index}"
        node = data.get("crosshair2", {}) if isinstance(data.get("crosshair2"), dict) else {}
        node[cam_key] = {
            "x_mm": float(self._crosshair2_pos_mm[0]),
            "y_mm": float(self._crosshair2_pos_mm[1]),
        }
        data["crosshair2"] = node

        try:
            write_yaml(path, data)
            self._log_message(f"Crosshair 2 saved: ({self._crosshair2_pos_mm[0]:.3f}, {self._crosshair2_pos_mm[1]:.3f}) mm")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair 2: {e}")

    def _goto_saved_crosshair(self):
        data = read_yaml(_config_path())
        cam_key = f"daheng_{self._fixed_index}"
        pos = ((data.get("crosshair") or {}).get(cam_key)) or {}

        if "x_mm" not in pos or "y_mm" not in pos:
            QMessageBox.information(self, "Crosshair", "No saved position found for crosshair 1 on this camera.")
            return

        try:
            x_mm = float(pos["x_mm"])
            y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair", "Saved position for crosshair 1 is invalid.")
            return

        self._crosshair_pos_mm = (x_mm, y_mm)
        if not self._crosshair_visible:
            self._crosshair_visible = True
        self._refresh_crosshair()
        self._log_message(f"Crosshair 1 loaded: ({x_mm:.3f}, {y_mm:.3f}) mm")

    def _goto_saved_crosshair2(self):
        data = read_yaml(_config_path())
        cam_key = f"daheng_{self._fixed_index}"
        pos = ((data.get("crosshair2") or {}).get(cam_key)) or {}

        if "x_mm" not in pos or "y_mm" not in pos:
            QMessageBox.information(self, "Crosshair 2", "No saved position found for crosshair 2 on this camera.")
            return

        try:
            x_mm = float(pos["x_mm"])
            y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair 2", "Saved position for crosshair 2 is invalid.")
            return

        self._crosshair2_pos_mm = (x_mm, y_mm)
        if not self._crosshair2_visible:
            self._crosshair2_visible = True
        self._refresh_crosshair2()
        self._log_message(f"Crosshair 2 loaded: ({x_mm:.3f}, {y_mm:.3f}) mm")

    # -------------------------------------------------------------------------
    # Mouse events
    # -------------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self._crosshair_visible and not self._crosshair_locked:
            self._crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
        if self._crosshair2_visible and not self._crosshair2_locked:
            self._crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()

    def _on_mouse_press(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Right-click toggles crosshair 1 lock
        if event.button == 3 and self._crosshair_visible:
            self._crosshair_locked = not self._crosshair_locked
            if self._crosshair_locked:
                self._crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()

        # Middle-click toggles crosshair 2 lock
        elif event.button == 2 and self._crosshair2_visible:
            self._crosshair2_locked = not self._crosshair2_locked
            if self._crosshair2_locked:
                self._crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()

    # -------------------------------------------------------------------------
    # Save frames
    # -------------------------------------------------------------------------

    def _get_save_directory(self) -> tuple[Path, datetime.datetime]:
        now = datetime.datetime.now()
        dir_path = data_dir() / now.strftime("%Y-%m-%d") / self._camera_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path, now

    def _generate_filename(self, timestamp: datetime.datetime, index: int, is_bg: bool) -> str:
        stem = f"{self._camera_name}_Background" if is_bg else f"{self._camera_name}_Image"
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{stem}_{ts_str}_{index}.png"

    def _save_single_frame(self, frame: np.ndarray, filepath: Path, exp_us: int, gain: int, comment: str):
        f8 = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
        img = Image.fromarray(f8, mode="L")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Exposure_us", str(exp_us))
        metadata.add_text("Gain", str(gain))
        metadata.add_text("Comment", comment)
        img.save(filepath, format="PNG", pnginfo=metadata)

    def _write_log_entry(self, log_path: Path, filename: str, exp_us: int, gain: int, comment: str):
        header = "ImageFile\tExposure_us\tGain\tComment\n"
        if not log_path.exists():
            log_path.write_text(header, encoding="utf-8")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{exp_us}\t{gain}\t{comment}\n")

    def _save_frames(self):
        with self._frame_lock:
            if self._cam is None and self._last_frame is None:
                QMessageBox.warning(self, "Warning", "No frame available.")
                return

        try:
            exp_us = int(self._exposure_edit.text())
            gain = int(self._gain_edit.text())
            n_frames = int(self._frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters.")
            return

        dir_path, base_ts = self._get_save_directory()
        comment = self._comment_edit.text()
        is_bg = self._background_cb.isChecked()

        saved = []
        for i in range(1, n_frames + 1):
            try:
                ts = datetime.datetime.now()
                with self._frame_lock:
                    frame = self._cam.capture_single(exp_us, gain) if self._cam else self._last_frame

                if frame is None:
                    raise RuntimeError("No frame captured.")

                # Apply ROI if enabled
                if self._use_roi_cb.isChecked() and self._roi_px is not None:
                    x0, y0, x1, y1 = self._roi_px
                    h0, w0 = frame.shape
                    x0 = max(0, min(w0 - 1, x0))
                    x1 = max(1, min(w0, x1))
                    y0 = max(0, min(h0 - 1, y0))
                    y1 = max(1, min(h0, y1))
                    frame = frame[y0:y1, x0:x1]

                filename = self._generate_filename(ts, i, is_bg)
                self._save_single_frame(frame, dir_path / filename, exp_us, gain, comment)
                saved.append(filename)
                self._log_message(f"Saved {filename}")
            except Exception as e:
                self._log_message(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                break

        if saved:
            log_filename = f"{self._camera_name}_log_{base_ts.strftime('%Y-%m-%d')}.log"
            log_path = dir_path / log_filename
            try:
                for filename in saved:
                    self._write_log_entry(log_path, filename, exp_us, gain, comment)
                self._log_message(f"Logged {len(saved)} file(s).")
            except Exception as e:
                self._log_message(f"Error writing log: {e}")

    # -------------------------------------------------------------------------
    # External API for scans
    # -------------------------------------------------------------------------

    def grab_frame_for_scan(
        self,
        averages: int = 1,
        adaptive=None,
        background: bool = False,
    ):
        """Grab frame(s) for use in scanning routines."""
        if not self._cam:
            raise DahengControllerError("Camera not activated.")
        was_live = bool(self._live_running)
        if was_live:
            try:
                self._stop_capture()
            except Exception:
                pass
        try:
            exp_us = int(self._exposure_edit.text())
        except ValueError:
            exp_us = DEFAULT_EXPOSURE_US
        try:
            device_gain = int(self._gain_edit.text())
        except ValueError:
            device_gain = DEFAULT_GAIN

        def _cap_once(cur_exp_us):
            self._cam.set_exposure(cur_exp_us)
            self._cam.set_gain(device_gain)
            return self._cam.capture_single(cur_exp_us, device_gain)

        n = max(1, int(averages))
        acc = None
        for _ in range(n):
            f = np.asarray(_cap_once(exp_us), dtype=np.float32)
            if self._use_roi_cb.isChecked() and self._roi_px is not None:
                x0, y0, x1, y1 = self._roi_px
                h0, w0 = f.shape
                x0 = max(0, min(w0 - 1, x0))
                x1 = max(1, min(w0, x1))
                y0 = max(0, min(h0 - 1, y0))
                y1 = max(1, min(h0, y1))
                f = f[y0:y1, x0:x1]
            acc = f if acc is None else (acc + f)
        avg = acc / n
        frame_u8 = np.clip(avg, 0, 255).astype(np.uint8)
        self.gui_update_image.emit(frame_u8)
        meta = {
            "CameraName": f"DahengCam_{self._fixed_index}",
            "CameraIndex": self._fixed_index,
            "Exposure_us": exp_us,
            "Gain": device_gain,
            "Background": "1" if background else "0",
            "ROI_px": (
                ""
                if self._roi_px is None
                else f"{self._roi_px[0]},{self._roi_px[1]},{self._roi_px[2]},{self._roi_px[3]}"
            ),
            "ROI_Used": (
                "1"
                if (self._use_roi_cb.isChecked() and self._roi_px is not None)
                else "0"
            ),
        }
        return frame_u8, meta

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._capture_thread:
            self._stop_capture()
        if self._cam:
            try:
                self._cam.deactivate()
            except Exception:
                pass

        for k in (
            f"camera:daheng:{self._camera_name.lower()}",
            f"camera:daheng:index:{self._fixed_index}",
        ):
            REGISTRY.unregister(k)

        try:
            if self._mpl_cid_click is not None:
                self._canvas.mpl_disconnect(self._mpl_cid_click)
            if self._mpl_cid_motion is not None:
                self._canvas.mpl_disconnect(self._mpl_cid_motion)
        except Exception:
            pass

        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLiveWindow(camera_name="Camera 1", fixed_index=1)
    gui.show()
    sys.exit(app.exec_())