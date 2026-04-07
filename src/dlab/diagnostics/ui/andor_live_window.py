from __future__ import annotations

import sys
import time
import datetime
import threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import rotate as sp_rotate
from PIL import Image, PngImagePlugin
import cmasher as cmr

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QDoubleSpinBox,
    QShortcut,
    QComboBox,
)
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from dlab.hardware.wrappers.andor_controller import (
    AndorController,
    AndorControllerError,
    DEFAULT_EXPOSURE_US,
    MIN_EXPOSURE_US,
    MAX_EXPOSURE_US,
)
from dlab.core.device_registry import REGISTRY
from dlab.utils.config_utils import cfg_get
from dlab.utils.paths_utils import data_dir
from dlab.utils.log_panel import LogPanel
from dlab.utils.yaml_utils import read_yaml, write_yaml
from dlab.boot import ROOT

REGISTRY_KEY = "camera:andor:andorcam_1"
SAVE_NAME = "AndorCam_1"

DEFAULT_PREPROCESS = {
    "enabled": False,
    "angle": -95.0,
    "x0": 165,
    "x1": 490,
    "y0": 210,
    "y1": 340,
}

COLORMAPS = [
    "cmr.rainforest",
    "cmr.neutral",
    "cmr.sunburst",
    "cmr.freeze",
    "turbo",
    "viridis",
    "plasma",
]


def _config_path() -> Path:
    return ROOT / "config" / "config.yaml"


def _resolve_cmap(key: str):
    if key.startswith("cmr."):
        name = key.split(".", 1)[1]
        return getattr(cmr, name)
    return plt.get_cmap(key)


class _LiveCaptureThread(QThread):
    """Background thread for continuous image capture."""

    image_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)

    def __init__(self, controller: AndorController, exposure: int, interval_ms: int):
        super().__init__()
        self._controller = controller
        self._exposure = exposure
        self._interval_s = interval_ms / 1000.0
        self._running = True
        self._lock = threading.Lock()
        self._frame_times: list[float] = []

    def update_parameters(self, exposure: int, interval_ms: int):
        with self._lock:
            self._exposure = exposure
            self._interval_s = interval_ms / 1000.0

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                with self._lock:
                    exp = self._exposure
                    interval = self._interval_s

                image = self._controller.capture_single(exp)
                self.image_signal.emit(image)
                self._update_fps()
                time.sleep(interval)
            except Exception:
                break

    def _update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]

        if len(self._frame_times) >= 2:
            fps = len(self._frame_times) / (
                self._frame_times[-1] - self._frame_times[0]
            )
        else:
            fps = 0.0
        self.fps_signal.emit(fps)


class AndorLiveWindow(QWidget):
    """Live view window for Andor camera."""

    closed = pyqtSignal()
    external_image_signal = pyqtSignal(np.ndarray)

    def __init__(self, log_panel: LogPanel | None = None):
        super().__init__()
        self.setWindowTitle("AndorLiveWindow")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._cam: AndorController | None = None
        self._capture_thread: _LiveCaptureThread | None = None
        self._last_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Plot state
        self._image_artist = None
        self._cbar = None
        self._fixed_cbar_max: float | None = None
        self._cmap_key = "cmr.rainforest"
        self._cmap = _resolve_cmap(self._cmap_key)

        # Crosshair state
        self._crosshair_visible = False
        self._crosshair_locked = False
        self._crosshair_pos: tuple[float, float] | None = None
        self._ch_h = None
        self._ch_v = None

        # Line drawing state
        self._line_mode_active = False
        self._line_start: tuple[float, float] | None = None
        self._line_artists: list = []

        self._init_ui()
        self._load_preprocess_from_config()
        self.external_image_signal.connect(self._update_image)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left panel - parameters
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Exposure
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self._exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self._exposure_edit.setValidator(
            QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self)
        )
        self._exposure_edit.textChanged.connect(self._on_params_changed)
        exp_layout.addWidget(self._exposure_edit)
        param_layout.addLayout(exp_layout)

        # Update interval
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self._interval_edit = QLineEdit("100")
        self._interval_edit.setValidator(QIntValidator(100, 10000, self))
        self._interval_edit.textChanged.connect(self._on_params_changed)
        int_layout.addWidget(self._interval_edit)
        param_layout.addLayout(int_layout)

        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self._fps_label = QLabel("0.0")
        fps_layout.addWidget(self._fps_label)
        fps_layout.addStretch()
        param_layout.addLayout(fps_layout)

        # MCP Voltage
        mcp_layout = QHBoxLayout()
        mcp_layout.addWidget(QLabel("MCP Voltage:"))
        self._mcp_voltage_edit = QLineEdit("Not specified")
        mcp_layout.addWidget(self._mcp_voltage_edit)
        param_layout.addLayout(mcp_layout)

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

        # Colorbar options
        self._autofix_cbar_cb = QCheckBox("Autofix Colorbar Max")
        self._autofix_cbar_cb.toggled.connect(self._on_autofix_cbar)
        param_layout.addWidget(self._autofix_cbar_cb)

        self._fix_cbar_cb = QCheckBox("Fix Colorbar Max")
        param_layout.addWidget(self._fix_cbar_cb)
        self._fix_value_edit = QLineEdit("10000")
        self._fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self._fix_value_edit.setEnabled(False)
        param_layout.addWidget(self._fix_value_edit)
        self._fix_cbar_cb.toggled.connect(self._fix_value_edit.setEnabled)
        self._fix_cbar_cb.toggled.connect(self._on_fix_cbar)
        self._fix_value_edit.textChanged.connect(self._on_fix_value_changed)

        # Background checkbox
        self._background_cb = QCheckBox("Background")
        param_layout.addWidget(self._background_cb)

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

        # Crosshair controls
        ch_grp = QGroupBox("Crosshair")
        ch_layout = QVBoxLayout(ch_grp)
        ch_row1 = QHBoxLayout()
        btn_ch_toggle = QPushButton("Toggle (Shift+C)")
        btn_ch_toggle.clicked.connect(self._toggle_crosshair)
        btn_ch_lock = QPushButton("Lock/Unlock")
        btn_ch_lock.clicked.connect(self._toggle_lock_manual)
        ch_row1.addWidget(btn_ch_toggle)
        ch_row1.addWidget(btn_ch_lock)
        ch_layout.addLayout(ch_row1)
        ch_row2 = QHBoxLayout()
        btn_ch_save = QPushButton("Save Position")
        btn_ch_save.clicked.connect(self._save_crosshair_to_config)
        btn_ch_goto = QPushButton("Load Position")
        btn_ch_goto.clicked.connect(self._goto_saved_crosshair)
        ch_row2.addWidget(btn_ch_save)
        ch_row2.addWidget(btn_ch_goto)
        ch_layout.addLayout(ch_row2)
        param_layout.addWidget(ch_grp)

        # Line controls
        ln_grp = QGroupBox("Lines")
        ln_layout = QHBoxLayout(ln_grp)
        btn_line_start = QPushButton("Start Line")
        btn_line_start.clicked.connect(self._start_line_mode)
        btn_line_clear = QPushButton("Clear Lines")
        btn_line_clear.clicked.connect(self._clear_lines)
        ln_layout.addWidget(btn_line_start)
        ln_layout.addWidget(btn_line_clear)
        param_layout.addWidget(ln_grp)

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

        # Preprocess controls
        pre_grp = QGroupBox("Display Preprocess (rotate + crop)")
        pre_layout = QVBoxLayout(pre_grp)

        self._pre_enable_cb = QCheckBox("Enable")
        self._pre_enable_cb.setChecked(DEFAULT_PREPROCESS["enabled"])
        pre_layout.addWidget(self._pre_enable_cb)

        row_angle = QHBoxLayout()
        row_angle.addWidget(QLabel("Angle (°):"))
        self._pre_angle_sb = QDoubleSpinBox()
        self._pre_angle_sb.setRange(-360.0, 360.0)
        self._pre_angle_sb.setDecimals(2)
        self._pre_angle_sb.setSingleStep(1.0)
        self._pre_angle_sb.setValue(DEFAULT_PREPROCESS["angle"])
        row_angle.addWidget(self._pre_angle_sb)
        pre_layout.addLayout(row_angle)

        row_x = QHBoxLayout()
        self._pre_x0_sb = QSpinBox()
        self._pre_x1_sb = QSpinBox()
        self._pre_x0_sb.setRange(0, 99999)
        self._pre_x1_sb.setRange(1, 99999)
        self._pre_x0_sb.setValue(DEFAULT_PREPROCESS["x0"])
        self._pre_x1_sb.setValue(DEFAULT_PREPROCESS["x1"])
        row_x.addWidget(QLabel("Crop x0:"))
        row_x.addWidget(self._pre_x0_sb)
        row_x.addWidget(QLabel("x1:"))
        row_x.addWidget(self._pre_x1_sb)
        pre_layout.addLayout(row_x)

        row_y = QHBoxLayout()
        self._pre_y0_sb = QSpinBox()
        self._pre_y1_sb = QSpinBox()
        self._pre_y0_sb.setRange(0, 99999)
        self._pre_y1_sb.setRange(1, 99999)
        self._pre_y0_sb.setValue(DEFAULT_PREPROCESS["y0"])
        self._pre_y1_sb.setValue(DEFAULT_PREPROCESS["y1"])
        row_y.addWidget(QLabel("Crop y0:"))
        row_y.addWidget(self._pre_y0_sb)
        row_y.addWidget(QLabel("y1:"))
        row_y.addWidget(self._pre_y1_sb)
        pre_layout.addLayout(row_y)

        pre_btn_row = QHBoxLayout()
        btn_pre_save = QPushButton("Save Settings")
        btn_pre_save.clicked.connect(self._save_preprocess_to_config)
        btn_pre_load = QPushButton("Load Settings")
        btn_pre_load.clicked.connect(self._load_preprocess_from_config)
        btn_pre_reset = QPushButton("Reset to Default")
        btn_pre_reset.clicked.connect(self._reset_preprocess_to_default)
        pre_btn_row.addWidget(btn_pre_save)
        pre_btn_row.addWidget(btn_pre_load)
        pre_btn_row.addWidget(btn_pre_reset)
        pre_layout.addLayout(pre_btn_row)
        param_layout.addWidget(pre_grp)

        param_layout.addStretch()
        splitter.addWidget(param_panel)

        # Right panel - plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self._figure, (self._ax_img, self._ax_profile) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        self._ax_img.set_title("Andor Camera Image")
        self._ax_img.set_xlabel("X (px)")
        self._ax_img.set_ylabel("Y (px)")
        self._ax_profile.set_title("Integrated Profile")
        self._ax_profile.grid(True, alpha=0.3)
        self._figure.subplots_adjust(right=0.85, hspace=0.15)

        self._canvas = FigureCanvas(self._figure)
        plot_layout.addWidget(self._canvas)
        splitter.addWidget(plot_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)
        self.resize(1200, 800)

        # Mouse events
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)

        # Shortcuts
        QShortcut(QKeySequence("Shift+C"), self).activated.connect(
            self._toggle_crosshair
        )
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_frames)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, message: str):
        if self._log:
            self._log.log(message, source="Andor")

    # -------------------------------------------------------------------------
    # Camera control
    # -------------------------------------------------------------------------

    def _activate_camera(self):
        try:
            self._cam = AndorController(device_index=0)
            self._cam.activate()
            REGISTRY.register(REGISTRY_KEY, self)
            self._log_message("Camera activated.")

            self._activate_btn.setEnabled(False)
            self._deactivate_btn.setEnabled(True)
            self._start_btn.setEnabled(True)
        except AndorControllerError as e:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {e}")
            self._log_message(f"Activation error: {e}")

    def _deactivate_camera(self):
        try:
            REGISTRY.unregister(REGISTRY_KEY)
            if self._cam:
                self._cam.deactivate()
                self._log_message("Camera deactivated.")
            self._cam = None

            self._activate_btn.setEnabled(True)
            self._deactivate_btn.setEnabled(False)
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
        except AndorControllerError as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate: {e}")

    def _start_capture(self):
        if self._cam is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return

        try:
            exposure = int(self._exposure_edit.text())
            interval = int(self._interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return

        self._capture_thread = _LiveCaptureThread(self._cam, exposure, interval)
        self._capture_thread.image_signal.connect(self._update_image)
        self._capture_thread.fps_signal.connect(self._update_fps)
        self._capture_thread.start()

        self._log_message("Live capture started.")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop_capture(self):
        if self._capture_thread:
            self._capture_thread.stop()
            self._capture_thread.wait()
            self._capture_thread = None

        self._log_message("Live capture stopped.")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._fps_label.setText("0.0")

    def _on_params_changed(self):
        try:
            exposure = int(self._exposure_edit.text())
            interval = int(self._interval_edit.text())
            if self._capture_thread:
                self._capture_thread.update_parameters(exposure, interval)
            elif self._cam:
                self._cam.set_exposure(exposure)
        except ValueError:
            pass

    def _update_fps(self, fps: float):
        self._fps_label.setText(f"{fps:.1f}")

    # -------------------------------------------------------------------------
    # Image display
    # -------------------------------------------------------------------------

    def _update_image(self, image: np.ndarray):
        with self._frame_lock:
            self._last_frame = image

        disp = self._display_preprocess(image)
        max_val = float(np.max(disp))
        min_val = float(np.min(disp))
        sum_val = float(np.sum(disp))
        mean_val = float(np.mean(disp))
        title = f"Sum: {sum_val:.0f} | Max: {max_val:.0f} | Mean: {mean_val:.1f}"

        # Check if dimensions changed
        recreate = False
        if self._image_artist is not None:
            if self._image_artist.get_array().shape != disp.shape:
                recreate = True

        if self._image_artist is None or recreate:
            self._ax_img.clear()
            if self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass

            self._image_artist = self._ax_img.imshow(
                disp, cmap=self._cmap, interpolation="None"
            )
            self._ax_img.set_title(title)
            self._ax_img.set_xlabel("X (px)")
            self._ax_img.set_ylabel("Y (px)")

            h, w = disp.shape
            fraction = min(0.04, max(0.015, 0.03 * (h / w)))
            self._cbar = self._figure.colorbar(
                self._image_artist,
                ax=self._ax_img,
                fraction=fraction,
                pad=0.02,
                aspect=30,
            )
            self._cbar.ax.set_ylabel("Intensity", rotation=270, labelpad=15)
        else:
            xlim = self._ax_img.get_xlim()
            ylim = self._ax_img.get_ylim()
            self._image_artist.set_data(disp)
            self._ax_img.set_xlim(xlim)
            self._ax_img.set_ylim(ylim)
            self._ax_img.set_title(title)

        # Colorbar scaling
        if self._autofix_cbar_cb.isChecked():
            self._fixed_cbar_max = max_val
            self._fix_value_edit.setText(str(int(max_val)))
            self._image_artist.set_clim(min_val, max_val)
        elif self._fix_cbar_cb.isChecked() and self._fixed_cbar_max is not None:
            self._image_artist.set_clim(min_val, self._fixed_cbar_max)
        else:
            self._fixed_cbar_max = None
            self._image_artist.set_clim(min_val, max_val)

        self._image_artist.set_cmap(self._cmap)
        self._cbar.update_normal(self._image_artist)

        # Profile
        axis = self._profile_axis_for_display()
        profile = np.sum(disp, axis=axis)
        self._ax_profile.clear()
        self._ax_profile.grid(True, alpha=0.3)

        if axis == 1:
            x = np.arange(disp.shape[0])
            self._ax_profile.plot(x, profile, linewidth=1.5)
            self._ax_profile.fill_between(x, profile, alpha=0.3)
            self._ax_profile.set_xlabel("Row (px)")
            self._ax_profile.set_xlim(0, disp.shape[0] - 1)
        else:
            x = np.arange(disp.shape[1])
            self._ax_profile.plot(x, profile, linewidth=1.5)
            self._ax_profile.fill_between(x, profile, alpha=0.3)
            self._ax_profile.set_xlabel("Column (px)")
            self._ax_profile.set_xlim(0, disp.shape[1] - 1)

        self._ax_profile.set_ylabel("Integrated Intensity")
        self._ax_profile.set_ylim(bottom=0)

        self._refresh_crosshair()
        self._canvas.draw_idle()

    def _display_preprocess(self, image: np.ndarray) -> np.ndarray:
        out = image
        if self._pre_enable_cb.isChecked():
            angle = float(self._pre_angle_sb.value())
            if abs(angle) > 1e-9:
                out = sp_rotate(
                    out, angle, reshape=True, order=3, mode="nearest"
                ).astype(image.dtype)

            y0, y1 = int(self._pre_y0_sb.value()), int(self._pre_y1_sb.value())
            x0, x1 = int(self._pre_x0_sb.value()), int(self._pre_x1_sb.value())
            h, w = out.shape[:2]
            y0, y1 = max(0, min(y0, h - 1)), max(y0 + 1, min(y1, h))
            x0, x1 = max(0, min(x0, w - 1)), max(x0 + 1, min(x1, w))
            out = out[y0:y1, x0:x1]
        return out

    def _profile_axis_for_display(self) -> int:
        if not self._pre_enable_cb.isChecked():
            return 1
        ang = float(self._pre_angle_sb.value()) % 180.0
        ang = min(ang, 180.0 - ang)
        return 0 if ang > 45.0 else 1

    # -------------------------------------------------------------------------
    # Colorbar controls
    # -------------------------------------------------------------------------

    def _on_autofix_cbar(self, checked: bool):
        if checked and self._fix_cbar_cb.isChecked():
            self._fix_cbar_cb.setChecked(False)
        self._log_message("Autofix colorbar " + ("enabled" if checked else "disabled"))

    def _on_fix_cbar(self, checked: bool):
        if checked:
            if self._autofix_cbar_cb.isChecked():
                self._autofix_cbar_cb.setChecked(False)
            try:
                self._fixed_cbar_max = float(self._fix_value_edit.text())
                self._log_message(f"Colorbar max fixed to {self._fixed_cbar_max:.1f}")
            except ValueError:
                self._fix_cbar_cb.setChecked(False)
        else:
            self._fixed_cbar_max = None
            self._log_message("Colorbar auto scale")

    def _on_fix_value_changed(self, text: str):
        if not self._fix_cbar_cb.isChecked() or not self._image_artist:
            return
        try:
            self._fixed_cbar_max = float(text)
            vmin, _ = self._image_artist.get_clim()
            self._image_artist.set_clim(vmin, self._fixed_cbar_max)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        except ValueError:
            pass

    def _on_cmap_changed(self, key: str):
        self._cmap_key = key
        self._cmap = _resolve_cmap(key)
        if self._image_artist is not None:
            self._image_artist.set_cmap(self._cmap)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        self._log_message(f"Colormap set to {key}")

    # -------------------------------------------------------------------------
    # Crosshair
    # -------------------------------------------------------------------------

    def _toggle_crosshair(self):
        self._crosshair_visible = not self._crosshair_visible
        if not self._crosshair_visible:
            self._crosshair_locked = False
        self._refresh_crosshair()
        self._canvas.draw_idle()

    def _toggle_lock_manual(self):
        if not self._crosshair_visible:
            return
        self._crosshair_locked = not self._crosshair_locked
        self._refresh_crosshair()
        self._canvas.draw_idle()

    def _refresh_crosshair(self):
        for artist in (self._ch_h, self._ch_v):
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._ch_h = self._ch_v = None

        if not self._crosshair_visible:
            return

        if self._crosshair_pos is None:
            y0, y1 = self._ax_img.get_ylim()
            x0, x1 = self._ax_img.get_xlim()
            self._crosshair_pos = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

        x, y = self._crosshair_pos
        color = "red" if self._crosshair_locked else "yellow"
        self._ch_h = self._ax_img.axhline(
            y, color=color, linestyle="--", linewidth=1.5, alpha=0.8
        )
        self._ch_v = self._ax_img.axvline(
            x, color=color, linestyle="--", linewidth=1.5, alpha=0.8
        )

    def _save_crosshair_to_config(self):
        if not self._crosshair_visible or self._crosshair_pos is None:
            QMessageBox.warning(self, "Crosshair", "Crosshair must be visible to save.")
            return

        path = _config_path()
        data = read_yaml(path)
        andor = data.get("andor", {}) if isinstance(data.get("andor"), dict) else {}
        andor["crosshair"] = {
            "x": float(self._crosshair_pos[0]),
            "y": float(self._crosshair_pos[1]),
        }
        data["andor"] = andor

        try:
            write_yaml(path, data)
            self._log_message(
                f"Crosshair saved: ({self._crosshair_pos[0]:.1f}, {self._crosshair_pos[1]:.1f})"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair: {e}")

    def _goto_saved_crosshair(self):
        data = read_yaml(_config_path())
        ch = ((data.get("andor") or {}).get("crosshair")) or {}

        if "x" not in ch or "y" not in ch:
            QMessageBox.information(self, "Crosshair", "No saved crosshair position.")
            return

        self._crosshair_visible = True
        self._crosshair_locked = True
        self._crosshair_pos = (float(ch["x"]), float(ch["y"]))
        self._refresh_crosshair()
        self._canvas.draw_idle()
        self._log_message(
            f"Loaded crosshair: ({self._crosshair_pos[0]:.1f}, {self._crosshair_pos[1]:.1f})"
        )

    # -------------------------------------------------------------------------
    # Lines
    # -------------------------------------------------------------------------

    def _start_line_mode(self):
        self._line_mode_active = True
        self._line_start = None
        self._log_message("Line: click to set start point.")

    def _clear_lines(self):
        for ln in self._line_artists:
            try:
                ln.remove()
            except Exception:
                pass
        self._line_artists.clear()
        self._line_start = None
        self._line_mode_active = False
        self._canvas.draw_idle()
        self._log_message("Lines cleared.")

    # -------------------------------------------------------------------------
    # Mouse events
    # -------------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if self._crosshair_visible and not self._crosshair_locked:
            if (
                event.xdata is None
                or event.ydata is None
                or event.inaxes != self._ax_img
            ):
                return
            self._crosshair_pos = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
            self._canvas.draw_idle()

    def _on_mouse_press(self, event):
        if event.inaxes != self._ax_img:
            return

        if self._line_mode_active and event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            if self._line_start is None:
                self._line_start = (float(event.xdata), float(event.ydata))
                self._log_message(
                    f"Line start: ({self._line_start[0]:.1f}, {self._line_start[1]:.1f})"
                )
            else:
                x0, y0 = self._line_start
                x1, y1 = float(event.xdata), float(event.ydata)
                (ln,) = self._ax_img.plot(
                    [x0, x1], [y0, y1], linewidth=2.0, color="red", alpha=0.8
                )
                self._line_artists.append(ln)
                self._line_start = None
                self._line_mode_active = False
                self._canvas.draw_idle()
                self._log_message(
                    f"Line added: ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f})"
                )
            return

        if event.button == 3 and self._crosshair_visible:
            self._crosshair_locked = not self._crosshair_locked
            if (
                self._crosshair_locked
                and event.xdata is not None
                and event.ydata is not None
            ):
                self._crosshair_pos = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
            self._canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Preprocess settings
    # -------------------------------------------------------------------------

    def _validate_preprocess(self) -> bool:
        x0, x1 = int(self._pre_x0_sb.value()), int(self._pre_x1_sb.value())
        y0, y1 = int(self._pre_y0_sb.value()), int(self._pre_y1_sb.value())
        if x0 >= x1:
            QMessageBox.warning(
                self, "Invalid Settings", f"x0 ({x0}) must be < x1 ({x1})"
            )
            return False
        if y0 >= y1:
            QMessageBox.warning(
                self, "Invalid Settings", f"y0 ({y0}) must be < y1 ({y1})"
            )
            return False
        return True

    def _save_preprocess_to_config(self):
        if not self._validate_preprocess():
            return

        path = _config_path()
        data = read_yaml(path)
        andor = data.get("andor", {}) if isinstance(data.get("andor"), dict) else {}
        andor["preprocess"] = {
            "enabled": bool(self._pre_enable_cb.isChecked()),
            "angle": float(self._pre_angle_sb.value()),
            "x0": int(self._pre_x0_sb.value()),
            "x1": int(self._pre_x1_sb.value()),
            "y0": int(self._pre_y0_sb.value()),
            "y1": int(self._pre_y1_sb.value()),
        }
        data["andor"] = andor

        try:
            write_yaml(path, data)
            self._log_message("Preprocess settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _load_preprocess_from_config(self):
        data = read_yaml(_config_path())
        preprocess = ((data.get("andor") or {}).get("preprocess")) or {}

        if not preprocess:
            self._log_message("No saved preprocess settings.")
            return

        self._pre_enable_cb.setChecked(
            bool(preprocess.get("enabled", DEFAULT_PREPROCESS["enabled"]))
        )
        self._pre_angle_sb.setValue(
            float(preprocess.get("angle", DEFAULT_PREPROCESS["angle"]))
        )
        self._pre_x0_sb.setValue(int(preprocess.get("x0", DEFAULT_PREPROCESS["x0"])))
        self._pre_x1_sb.setValue(int(preprocess.get("x1", DEFAULT_PREPROCESS["x1"])))
        self._pre_y0_sb.setValue(int(preprocess.get("y0", DEFAULT_PREPROCESS["y0"])))
        self._pre_y1_sb.setValue(int(preprocess.get("y1", DEFAULT_PREPROCESS["y1"])))
        self._log_message("Preprocess settings loaded.")

    def _reset_preprocess_to_default(self):
        self._pre_enable_cb.setChecked(DEFAULT_PREPROCESS["enabled"])
        self._pre_angle_sb.setValue(DEFAULT_PREPROCESS["angle"])
        self._pre_x0_sb.setValue(DEFAULT_PREPROCESS["x0"])
        self._pre_x1_sb.setValue(DEFAULT_PREPROCESS["x1"])
        self._pre_y0_sb.setValue(DEFAULT_PREPROCESS["y0"])
        self._pre_y1_sb.setValue(DEFAULT_PREPROCESS["y1"])
        self._log_message("Preprocess reset to defaults.")

    # -------------------------------------------------------------------------
    # Save frames
    # -------------------------------------------------------------------------

    def _get_save_directory(self) -> tuple[Path, datetime.datetime]:
        now = datetime.datetime.now()
        dir_path = data_dir() / now.strftime("%Y-%m-%d") / SAVE_NAME
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path, now

    def _generate_filename(
        self, timestamp: datetime.datetime, index: int, is_bg: bool
    ) -> str:
        stem = f"{SAVE_NAME}_Background" if is_bg else f"{SAVE_NAME}_Image"
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{stem}_{ts_str}_{index}.png"

    def _save_single_frame(
        self, frame: np.ndarray, filepath: Path, exposure: int, mcp: str, comment: str
    ):
        frame_uint16 = np.clip(frame, 0, 65535).astype(np.uint16)
        img = Image.fromarray(frame_uint16, mode="I;16")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Exposure", str(exposure))
        metadata.add_text("MCP Voltage", mcp)
        metadata.add_text("Comment", comment)
        img.save(filepath, format="PNG", pnginfo=metadata)

    def _write_log_entry(
        self, log_path: Path, filename: str, exposure: int, mcp: str, comment: str
    ):
        header = "File Name\tExposure (µs)\tMCP Voltage\tComment\n"
        if not log_path.exists():
            log_path.write_text(header, encoding="utf-8")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{exposure}\t{mcp}\t{comment}\n")

    def _save_frames(self):
        with self._frame_lock:
            if self._cam is None and self._last_frame is None:
                QMessageBox.warning(self, "Warning", "No frame available.")
                return

        try:
            exposure = int(self._exposure_edit.text())
            n_frames = int(self._frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters.")
            return

        dir_path, base_ts = self._get_save_directory()
        mcp = self._mcp_voltage_edit.text()
        comment = self._comment_edit.text()
        is_bg = self._background_cb.isChecked()

        saved = []
        for i in range(1, n_frames + 1):
            try:
                ts = datetime.datetime.now()
                with self._frame_lock:
                    frame = (
                        self._cam.capture_single(exposure)
                        if self._cam
                        else self._last_frame
                    )

                if frame is None:
                    raise RuntimeError("No frame captured.")

                filename = self._generate_filename(ts, i, is_bg)
                self._save_single_frame(
                    frame, dir_path / filename, exposure, mcp, comment
                )
                saved.append(filename)
                self._log_message(f"Saved {filename}")
            except Exception as e:
                self._log_message(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                break

        if saved:
            log_filename = f"{SAVE_NAME}_log_{base_ts.strftime('%Y-%m-%d')}.log"
            log_path = dir_path / log_filename
            try:
                for filename in saved:
                    self._write_log_entry(log_path, filename, exposure, mcp, comment)
                self._log_message(f"Logged {len(saved)} file(s).")
            except Exception as e:
                self._log_message(f"Error writing log: {e}")

    def grab_frame_for_scan(
        self,
        averages: int = 1,
        background: bool = False,
        dead_pixel_cleanup: bool = False,
        exposure_us: int | None = None,
        force_roi: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Grab frame(s) for use in scanning routines."""

        if not self._cam:
            raise AndorControllerError("Camera not activated.")

        exp = exposure_us or int(self._exposure_edit.text())
        n = max(1, averages)
        acc = None

        for _ in range(n):
            f = self._cam.capture_single(exp).astype(np.float64)
            acc = f if acc is None else (acc + f)
        avg = acc / n

        if dead_pixel_cleanup:
            avg[avg >= 65535.0] = 0.0

        frame = np.clip(avg, 0, 65535).astype(np.uint16)

        self.external_image_signal.emit(frame.astype(np.float64))

        meta = {
            "DeviceName": SAVE_NAME,
            "Exposure_us": exp,
            "Background": "1" if background else "0",
        }

        return frame, meta

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._capture_thread:
            self._stop_capture()
        if self._cam:
            self._deactivate_camera()
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AndorLiveWindow()
    gui.show()
    sys.exit(app.exec_())
