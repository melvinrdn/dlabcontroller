from __future__ import annotations

import time
import datetime
import threading
from pathlib import Path
from typing import List

import numpy as np
import pyvisa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QMessageBox, QSplitter, QCheckBox, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.powermeter_controller import PowermeterController
from dlab.utils.paths_utils import data_dir
from dlab.utils.log_panel import LogPanel

REGISTRY_KEY_PREFIX = "powermeter:thorlabs"


class _LivePowerThread(QThread):
    """Background thread for continuous power reading."""

    power_signal = pyqtSignal(float, float)
    error_signal = pyqtSignal(str)

    def __init__(self, ctrl: PowermeterController, period_s: float, io_lock=None):
        super().__init__()
        self._ctrl = ctrl
        self._period = float(max(0.02, period_s))
        self._running = True
        self._lock = threading.Lock()
        self._io_lock = io_lock

    def update_period(self, period_s: float) -> None:
        with self._lock:
            self._period = float(max(0.02, period_s))

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            try:
                with self._lock:
                    period = self._period

                io_lock = self._io_lock or threading.Lock()
                with io_lock:
                    val = float(self._ctrl.read_power())
                ts = time.time()
                self.power_signal.emit(ts, val)
                time.sleep(period)
            except Exception as e:
                self.error_signal.emit(str(e))
                break


class PowermeterLiveWindow(QWidget):
    """Live view window for Thorlabs powermeter."""

    closed = pyqtSignal()

    def __init__(self, log_panel: LogPanel | None = None):
        super().__init__()
        self.setWindowTitle("PowermeterLiveWindow")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._ctrl: PowermeterController | None = None
        self._capture_thread: _LivePowerThread | None = None
        self._io_lock = threading.Lock()
        self._registry_key: str | None = None

        # Data buffers
        self._t: List[float] = []
        self._y: List[float] = []

        self._init_ui()

        try:
            REGISTRY.register("ui:powermeter_live", self)
        except Exception:
            pass

    def _init_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)
        splitter = QSplitter()

        # Left panel - controls
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(6, 6, 6, 6)
        left_l.setSpacing(6)
        left.setMaximumWidth(230)

        # Powermeter selection
        self._pm_combo = QComboBox()
        btn_search = QPushButton("Search Powermeters")
        btn_search.clicked.connect(self._search_powermeters)
        left_l.addWidget(QLabel("Select Powermeter:"))
        left_l.addWidget(self._pm_combo)
        left_l.addWidget(btn_search)

        # Activate/Deactivate
        self._activate_btn = QPushButton("Activate")
        self._deactivate_btn = QPushButton("Deactivate")
        self._deactivate_btn.setEnabled(False)
        self._activate_btn.clicked.connect(self._activate_hardware)
        self._deactivate_btn.clicked.connect(self._deactivate_hardware)
        left_l.addWidget(self._activate_btn)
        left_l.addWidget(self._deactivate_btn)

        # Settings
        self._wl_combo = QComboBox()
        self._wl_combo.addItems(["1030", "515", "343"])
        self._avg_edit = QLineEdit("1")
        self._period_edit = QLineEdit("0.1")
        self._win_edit = QLineEdit("10")

        for w in (self._wl_combo, self._avg_edit, self._period_edit, self._win_edit, self._pm_combo):
            w.setMaximumWidth(110)

        left_l.addWidget(QLabel("Wavelength (nm):"))
        left_l.addWidget(self._wl_combo)
        left_l.addWidget(QLabel("Averaging count:"))
        left_l.addWidget(self._avg_edit)
        left_l.addWidget(QLabel("Sampling period (s):"))
        left_l.addWidget(self._period_edit)
        left_l.addWidget(QLabel("Time window (s):"))
        left_l.addWidget(self._win_edit)

        # Y range group
        range_grp = QGroupBox("Y range")
        rg_l = QVBoxLayout(range_grp)
        rg_l.setContentsMargins(6, 6, 6, 6)
        rg_l.setSpacing(4)

        self._ylim_cb = QCheckBox("Choose range")
        self._ylim_cb.toggled.connect(self._toggle_ylim_inputs)

        row_min = QHBoxLayout()
        row_max = QHBoxLayout()
        self._ymin_edit = QLineEdit("")
        self._ymin_edit.setMaximumWidth(90)
        self._ymax_edit = QLineEdit("")
        self._ymax_edit.setMaximumWidth(90)
        self._ymin_edit.setEnabled(False)
        self._ymax_edit.setEnabled(False)

        row_min.addWidget(QLabel("Power min:"))
        row_min.addWidget(self._ymin_edit, 1)
        row_max.addWidget(QLabel("Power max:"))
        row_max.addWidget(self._ymax_edit, 1)

        rg_l.addWidget(self._ylim_cb)
        rg_l.addLayout(row_min)
        rg_l.addLayout(row_max)
        left_l.addWidget(range_grp)

        # Comment
        left_l.addWidget(QLabel("Comment:"))
        self._comment_edit = QLineEdit("")
        self._comment_edit.setMaximumWidth(200)
        left_l.addWidget(self._comment_edit)

        # Apply button
        btn_apply = QPushButton("Apply Settings")
        btn_apply.clicked.connect(self._apply_settings)
        left_l.addWidget(btn_apply)

        # Start/Stop buttons
        self._start_btn = QPushButton("Start Live")
        self._start_btn.clicked.connect(self._start_live)
        self._stop_btn = QPushButton("Stop Live")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_live)
        left_l.addWidget(self._start_btn)
        left_l.addWidget(self._stop_btn)

        left_l.addStretch()
        splitter.addWidget(left)

        # Right panel - plot
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6, 6, 6, 6)
        right_l.setSpacing(6)

        # Header with power display
        header = QWidget()
        header_l = QHBoxLayout(header)
        header_l.setContentsMargins(0, 0, 0, 0)
        header_l.setSpacing(12)

        self._lbl_big_now = QLabel("—")
        self._lbl_big_max = QLabel("—")
        self._lbl_big_mean = QLabel("—")

        self._lbl_big_now.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._lbl_big_max.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._lbl_big_mean.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._lbl_big_now.setStyleSheet("font-size: 36pt; font-weight: 800;")
        self._lbl_big_max.setStyleSheet("font-size: 24pt; font-weight: 700;")
        self._lbl_big_mean.setStyleSheet("font-size: 24pt; font-weight: 700;")

        lab_now = QLabel("Power:")
        lab_now.setStyleSheet("font-size: 12pt;")
        lab_max = QLabel("Max:")
        lab_max.setStyleSheet("font-size: 12pt;")
        lab_mean = QLabel("Mean:")
        lab_mean.setStyleSheet("font-size: 12pt;")

        self._save_window_btn = QPushButton("Save Window")
        self._save_window_btn.clicked.connect(self._save_current_window)

        header_l.addWidget(lab_now)
        header_l.addWidget(self._lbl_big_now, 1)
        header_l.addSpacing(8)
        header_l.addWidget(lab_max)
        header_l.addWidget(self._lbl_big_max)
        header_l.addSpacing(8)
        header_l.addWidget(lab_mean)
        header_l.addWidget(self._lbl_big_mean)
        header_l.addStretch(1)
        header_l.addWidget(self._save_window_btn)
        right_l.addWidget(header)

        # Plot
        self._figure, self._ax = plt.subplots()
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Power (W)")
        self._ax.grid(True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        right_l.addWidget(self._toolbar)
        right_l.addWidget(self._canvas)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 780])

        main.addWidget(splitter)
        self.resize(1100, 680)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str):
        if self._log:
            self._log.log(msg, source="Powermeter")

    # -------------------------------------------------------------------------
    # UI helpers
    # -------------------------------------------------------------------------

    def _toggle_ylim_inputs(self, checked: bool):
        self._ymin_edit.setEnabled(checked)
        self._ymax_edit.setEnabled(checked)

    def _plot_color(self) -> str:
        w = self._current_wavelength_nm()
        if w == 343:
            return "#9400D3"
        if w == 515:
            return "#00A000"
        return "#CC0000"

    def _fmt_power(self, v: float) -> str:
        a = abs(v)
        if a < 1e-6:
            return f"{v * 1e9:.3g} nW"
        if a < 1e-3:
            return f"{v * 1e6:.3g} µW"
        if a < 1:
            return f"{v * 1e3:.3g} mW"
        return f"{v:.3g} W"

    def _current_wavelength_nm(self) -> int:
        try:
            return int(self._wl_combo.currentText())
        except Exception:
            return 1030

    # -------------------------------------------------------------------------
    # Hardware control
    # -------------------------------------------------------------------------

    def _search_powermeters(self):
        self._pm_combo.clear()
        try:
            rm = pyvisa.ResourceManager()
            res = list(rm.list_resources())
        except Exception:
            res = []

        if not res:
            QMessageBox.critical(self, "Error", "No VISA resources found.")
            self._log_message("No VISA resources found.")
            return

        self._pm_combo.addItems(res)
        self._log_message(f"Found {len(res)} resource(s).")

    def _activate_hardware(self):
        idx = self._pm_combo.currentIndex()
        if idx < 0:
            QMessageBox.critical(self, "Error", "No powermeter selected.")
            return

        res = self._pm_combo.currentText().strip()
        if not res:
            QMessageBox.critical(self, "Error", "Invalid resource.")
            return

        try:
            self._ctrl = PowermeterController(res)
            self._ctrl.activate()
            self._apply_settings()

            key = f"{REGISTRY_KEY_PREFIX}:pm_{idx + 1}"
            try:
                for k, v in REGISTRY.items(prefix=f"{REGISTRY_KEY_PREFIX}:"):
                    if k == key or v is self._ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass

            REGISTRY.register(key, self._ctrl)
            self._registry_key = key
            self._log_message(f"Activated and registered '{key}'.")
            self._activate_btn.setEnabled(False)
            self._deactivate_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self._log_message(f"Activate failed: {e}")

    def _deactivate_hardware(self):
        if self._capture_thread is not None:
            QMessageBox.information(
                self, "Live running", "Stop Live before deactivating the device."
            )
            return

        try:
            if self._ctrl:
                self._ctrl.deactivate()
                self._log_message("Powermeter deactivated.")
        finally:
            try:
                if self._registry_key:
                    REGISTRY.unregister(self._registry_key)
                    self._log_message(f"Unregistered '{self._registry_key}'.")
            except Exception:
                pass
            self._registry_key = None
            self._ctrl = None
            self._activate_btn.setEnabled(True)
            self._deactivate_btn.setEnabled(False)
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def _apply_settings(self):
        if not self._ctrl:
            QMessageBox.warning(self, "Warning", "Powermeter not activated.")
            return

        try:
            wl = float(self._current_wavelength_nm())
            av = int(float(self._avg_edit.text()))
            with self._io_lock:
                self._ctrl.set_wavelength(wl)
                self._ctrl.set_avg(av)

                try:
                    self._ctrl.set_auto_range(True)
                except Exception:
                    pass
                try:
                    self._ctrl.set_bandwidth("high")
                except Exception:
                    pass

            self._log_message("Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")
            self._log_message(f"Apply failed: {e}")

    # -------------------------------------------------------------------------
    # Live capture
    # -------------------------------------------------------------------------

    def _start_live(self):
        if not self._ctrl:
            QMessageBox.critical(self, "Error", "Powermeter not activated.")
            return

        try:
            period = float(self._period_edit.text())
        except Exception:
            period = 0.1

        self._t.clear()
        self._y.clear()

        self._capture_thread = _LivePowerThread(self._ctrl, period, io_lock=self._io_lock)
        self._capture_thread.power_signal.connect(self._update_power)
        self._capture_thread.error_signal.connect(self._on_capture_thread_error, Qt.QueuedConnection)
        self._capture_thread.start()

        self._log_message("Live started.")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._deactivate_btn.setEnabled(False)

    def _stop_live(self):
        if self._capture_thread:
            self._capture_thread.stop()
            if not self._capture_thread.wait(2000):
                self._log_message("Capture thread did not stop cleanly.")
            self._capture_thread = None
            self._log_message("Live stopped.")

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._deactivate_btn.setEnabled(True)

    def _on_capture_thread_error(self, err: str):
        self._log_message(f"Live capture thread error: {err}")
        QMessageBox.critical(self, "Acquisition error", f"Live capture stopped: {err}")
        self._stop_live()

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def _window_arrays(self):
        if not self._t:
            return np.array([]), np.array([])

        try:
            win = float(self._win_edit.text())
        except Exception:
            win = 10.0
        if win <= 0:
            win = 10.0

        t_now = self._t[-1]
        t0_keep = t_now - win

        i0 = 0
        for i in range(len(self._t)):
            if self._t[i] >= t0_keep:
                i0 = i
                break

        x = np.asarray(self._t[i0:]) - float(self._t[i0] if len(self._t[i0:]) else t_now)
        y = np.asarray(self._y[i0:])
        return x, y

    def _update_power(self, ts: float, val: float):
        self._t.append(ts)
        self._y.append(val)

        x, y = self._window_arrays()

        self._ax.cla()
        color = self._plot_color()
        self._ax.plot(x, y, color=color, linewidth=2.0)
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Power (W)")
        self._ax.grid(True)
        self._ax.set_title(f"Powermeter Live — {self._current_wavelength_nm()} nm")

        if self._ylim_cb.isChecked():
            try:
                ymin = float(self._ymin_edit.text())
                ymax = float(self._ymax_edit.text())
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    self._ax.set_ylim(ymin, ymax)
            except Exception:
                pass

        self._canvas.draw_idle()

        # Update labels
        p_now = self._fmt_power(val)
        p_max = self._fmt_power(np.max(y) if y.size else val)
        p_mean = self._fmt_power(float(np.mean(y)) if y.size else val)

        self._lbl_big_now.setText(p_now)
        self._lbl_big_max.setText(p_max)
        self._lbl_big_mean.setText(p_mean)

        self._lbl_big_now.setStyleSheet(f"font-size: 36pt; font-weight: 800; color: {color};")
        self._lbl_big_max.setStyleSheet(f"font-size: 24pt; font-weight: 700; color: {color};")
        self._lbl_big_mean.setStyleSheet(f"font-size: 24pt; font-weight: 700; color: {color};")

    @pyqtSlot(float, float)
    def set_power_from_scan(self, ts: float, val: float) -> None:
        """External slot for receiving power values from scans."""
        self._update_power(ts, val)

    @pyqtSlot(object)
    def refresh_from_device(self, dev) -> None:
        """External slot for refreshing from device."""
        try:
            val = float(dev.fetch_power())
            self._update_power(time.time(), val)
        except Exception as e:
            self._log_message(f"Live refresh error: {e}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def _get_save_directory(self) -> tuple[Path, datetime.datetime]:
        now = datetime.datetime.now()
        dir_path = data_dir() / now.strftime("%Y-%m-%d") / "powermeter"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path, now

    def _save_current_window(self):
        x, y = self._window_arrays()
        if y.size == 0:
            QMessageBox.warning(self, "Save Window", "No data to save.")
            return

        dir_path, now = self._get_save_directory()
        filename = f"powermeter_log_{now.strftime('%Y-%m-%d_%H_%M_%S')}.txt"
        filepath = dir_path / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                comment = self._comment_edit.text().strip()
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Wavelength: {self._current_wavelength_nm()} nm\n")
                f.write("# t_s\tpower_W\n")
                for xi, yi in zip(x, y):
                    f.write(f"{xi:.6f}\t{yi:.9g}\n")
            self._log_message(f"Saved window to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Window", f"Failed to save: {e}")
            self._log_message(f"Save failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        try:
            self._stop_live()
        finally:
            try:
                if self._ctrl:
                    self._ctrl.deactivate()
            finally:
                try:
                    if self._registry_key:
                        REGISTRY.unregister(self._registry_key)
                except Exception:
                    pass
                try:
                    REGISTRY.unregister("ui:powermeter_live")
                except Exception:
                    pass
                self._registry_key = None
                self._ctrl = None

        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = PowermeterLiveWindow()
    w.show()
    sys.exit(app.exec_())