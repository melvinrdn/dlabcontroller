from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QGroupBox,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.hardware.wrappers.piezojena_controller import NV40


# -----------------------------------------------------------------------------
# Instance presets
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseLockPreset:
    """Default values for a phase-lock instance."""

    label: str
    registry_key: str
    spec_key: str
    stage_key: str


_PRESETS: dict[str, PhaseLockPreset] = {
    "2w": PhaseLockPreset(
        label="ω/2ω",
        registry_key="phaselock:avaspec:2w",
        spec_key="spectrometer:avaspec:2w",
        stage_key="stage:piezojena:nv40_2w",
    ),
    "3w": PhaseLockPreset(
        label="ω/3ω",
        registry_key="phaselock:avaspec:3w",
        spec_key="spectrometer:avaspec:3w",
        stage_key="stage:piezojena:nv40_3w",
    ),
}

PHASE_MODE_SINGLE = "Single bin"
PHASE_MODE_VECTOR = "Vector avg"


# -----------------------------------------------------------------------------
# Signal processing
# -----------------------------------------------------------------------------


def fft_deskew(y: np.ndarray) -> np.ndarray:
    """FFT of *y* with the analytic half-sample linear phase ramp removed."""
    n = y.size
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y)))
    j = np.arange(n)
    omega = 2.0 * np.pi * (j - n / 2.0) / (n - 1)
    return np.fft.ifftshift(spectrum * np.exp(-1j * (n - 1) / n * omega * 0.5))


def fit_ramp_slope(
    spectrum: np.ndarray,
    center: int,
    half_window: int = 12,
) -> float:
    """Weighted linear fit of the unwrapped phase around *center*; returns the slope in rad/bin, or 0.0 when it cannot be fitted."""
    n = spectrum.size
    if not (0 <= center < n) or half_window < 2:
        return 0.0

    i0 = max(0, center - half_window)
    i1 = min(n, center + half_window + 1)
    if i1 - i0 < 3:
        return 0.0

    bins = np.arange(i0, i1)
    unwrapped = np.unwrap(np.angle(spectrum[i0:i1]))
    weights = np.abs(spectrum[i0:i1])
    if not np.any(weights):
        return 0.0

    return float(np.polyfit(bins - center, unwrapped, 1, w=weights)[0])


def ramp_phasor(n: int, center: int, slope: float) -> np.ndarray:
    """Correction exp(-i*slope*(j-center)), equal to 1 at *center* so the measured phase there is unchanged."""
    return np.exp(-1j * slope * (np.arange(n) - center))


def extract_phase(
    spectrum_fft: np.ndarray,
    center: int,
    half_window: int,
    mode: str,
) -> float:
    """Phase at *center*: single bin angle, or angle of the coherent sum over center +/- half_window."""
    n = spectrum_fft.size
    if not (0 <= center < n):
        return float("nan")

    if mode == PHASE_MODE_VECTOR and half_window > 0:
        i0 = max(0, center - half_window)
        i1 = min(n, center + half_window + 1)
        acc = np.sum(spectrum_fft[i0:i1])
        if acc == 0:
            return float("nan")
        return float(np.angle(acc))

    return float(np.angle(spectrum_fft[center]))


# -----------------------------------------------------------------------------
# Worker Threads
# -----------------------------------------------------------------------------


class ControlThread(QThread):
    """PID control thread for phase locking via piezo stage."""

    update_status = pyqtSignal(float, float)

    def __init__(self, stage: NV40) -> None:
        super().__init__()
        self.stage = stage
        self.kp = 0.1
        self.ki = 0.0
        self.kd = 0.0
        self.gain = -1.0
        self.max_step = 0.05
        self.target = 0.0
        self.unwrap = True
        self.period = 0.0  # min seconds between corrections; 0 = every frame
        self.integral = 0.0
        self.last_t: float | None = None
        self.last_err = 0.0
        self.q: deque[float] = deque(maxlen=1)  # loop must act on the latest sample
        self.vmin = 0.0
        self.vmax = 140.0
        self._phi_prev: float | None = None
        self._running = True
        try:
            self.current_v = float(self.stage.get_position())
        except Exception:
            self.current_v = 80.0
        self.enabled = False

    def reset(self) -> None:
        """Clear all integrator, derivative and unwrap state."""
        self.integral = 0.0
        self.last_t = None
        self.last_err = 0.0
        self._phi_prev = None

    def seed_unwrap(self, phi: float) -> None:
        """Align the loop's unwrap chain with an externally unwrapped value."""
        self._phi_prev = float(phi)

    def run(self) -> None:
        while self._running and not self.isInterruptionRequested():
            if not self.enabled:
                # Stale timebase would blow up the derivative on re-enable.
                self.q.clear()
                self.last_t = None
                time.sleep(0.005)
                continue

            if not self.q:
                time.sleep(0.001)
                continue

            phi = self.q.popleft()

            # Unwrap on every sample so 2π jumps are never missed, even when
            # the correction itself is rate-limited below.
            if self.unwrap:
                if self._phi_prev is None:
                    self._phi_prev = phi
                phi_use = float(np.unwrap([self._phi_prev, phi])[-1])
                self._phi_prev = phi_use
                err = self.target - phi_use
            else:
                phi_use = phi
                err = (self.target - phi + np.pi) % (2 * np.pi) - np.pi

            now = time.monotonic()
            if self.last_t is None:
                # Seed with the actual error: no derivative kick on the next cycle.
                self.last_t = now
                self.last_err = err
                continue

            if now - self.last_t < self.period:
                continue  # sample consumed for unwrap, correction skipped

            dt = max(1e-3, now - self.last_t)
            self.last_t = now

            d_err = (err - self.last_err) / dt
            self.last_err = err

            if self.ki != 0.0:
                self.integral += err * dt
                # Clamp so |gain * ki * integral| <= max_step.
                denom = abs(self.ki) * (abs(self.gain) if self.gain != 0.0 else 1.0)
                max_int = self.max_step / denom
                self.integral = float(np.clip(self.integral, -max_int, max_int))
            else:
                self.integral = 0.0

            u = self.gain * (self.kp * err + self.ki * self.integral + self.kd * d_err)

            if abs(u) > self.max_step:
                u = float(np.sign(u) * self.max_step)

            new_v = self.current_v + u
            new_v = max(self.vmin, min(self.vmax, new_v))
            new_v = round(new_v / 0.01) * 0.01

            try:
                self.stage.set_position(new_v)
                self.current_v = new_v
            except Exception:
                pass

            self.update_status.emit(phi_use, self.current_v)

    def stop(self) -> None:
        self._running = False
        self.enabled = False
        self.requestInterruption()


class AvaspecThread(QThread):
    """Spectrometer acquisition thread."""

    data_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.running = True

    def run(self) -> None:
        while self.running and not self.isInterruptionRequested():
            try:
                ts, wl, counts = self.ctrl.measure_once()
                self.data_ready.emit(wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break

    def stop(self) -> None:
        self.running = False
        self.requestInterruption()


# -----------------------------------------------------------------------------
# AvaspecPhaseLockWindow
# -----------------------------------------------------------------------------


class AvaspecPhaseLockWindow(QWidget):
    """Phase lock window: fringe spectrum -> FFT peak phase -> PID -> piezo."""

    closed = pyqtSignal()

    def __init__(
        self,
        instance_id: str = "2w",
        preset: PhaseLockPreset | None = None,
        log_panel: LogPanel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._preset = preset or _PRESETS[instance_id]
        self._instance_id = instance_id

        self.setWindowTitle(f"Phase Lock — {self._preset.label}")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._registry_key = self._preset.registry_key
        self._spec_ctrl = None
        self._acq_thread: AvaspecThread | None = None
        self._ctrl_thread: ControlThread | None = None

        # Plot state
        self._max_points = 200
        self._hist_phi_raw: deque[float] = deque(maxlen=self._max_points)
        self._hist_phi_unwrapped: deque[float] = deque(maxlen=self._max_points)
        self._hist_t: deque[float] = deque(maxlen=self._max_points)
        self._last_draw = 0.0
        self._min_draw_dt = 0.05
        self._fft_scatter = None
        self._fft_marker = None
        self._phase_scatter = None
        self._phase_sp_line = None
        self._bg_fft = None
        self._bg_phase = None
        self._blitting_initialized = False

        self._last_mag: np.ndarray | None = None  # for the Find Peak button

        # Ramp-removal cache: the slope is fitted once and reused until the peak
        # selection changes, the checkbox is toggled, or a refit is requested.
        self._ramp_slope: float = 0.0
        self._ramp_key: tuple[int, int] | None = None  # (center, half_window)
        self._ramp_phasor: np.ndarray | None = None

        # Setpoint modulation state
        self._mod_active = False
        self._mod_t0 = 0.0
        self._mod_amp = 0.0
        self._mod_period = 1.0
        self._mod_center = 0.0

        self._init_ui()
        REGISTRY.register(self._registry_key, self)

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)

        self._fig, (self._ax_fft, self._ax_phase) = plt.subplots(1, 2, figsize=(14, 4.5))
        self._fig.tight_layout(pad=2.0)
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, 3)

        panel = QVBoxLayout()
        panel.addWidget(self._create_connection_group())
        panel.addWidget(self._create_fft_group())
        panel.addWidget(self._create_pid_group())
        panel.addWidget(self._create_voltage_group())
        panel.addWidget(self._create_modulation_group())
        panel.addLayout(self._create_control_row())
        root.addLayout(panel, 2)

        self._mod_timer = QTimer(self)
        self._mod_timer.timeout.connect(self._on_mod_step)

    @staticmethod
    def _add_field(
        layout: QGridLayout,
        row: int,
        col: int,
        label: str,
        widget: QWidget,
        width: int = 70,
    ) -> int:
        """Place a (label, editor) pair and return the next free column."""
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(lbl, row, col)
        widget.setFixedWidth(width)
        layout.addWidget(widget, row, col + 1)
        return col + 2

    def _create_connection_group(self) -> QGroupBox:
        group = QGroupBox("Devices")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Spectrometer:"), 0, 0)
        self._spec_key_edit = QLineEdit(self._preset.spec_key)
        self._spec_key_edit.setFixedWidth(220)
        layout.addWidget(self._spec_key_edit, 0, 1)
        btn_spec = QPushButton("Connect")
        btn_spec.setFixedWidth(90)
        btn_spec.clicked.connect(self._on_connect_spec)
        layout.addWidget(btn_spec, 0, 2)

        layout.addWidget(QLabel("NV40 Stage:"), 1, 0)
        self._stage_key_edit = QLineEdit(self._preset.stage_key)
        self._stage_key_edit.setFixedWidth(220)
        layout.addWidget(self._stage_key_edit, 1, 1)
        btn_stage = QPushButton("Connect")
        btn_stage.setFixedWidth(90)
        btn_stage.clicked.connect(self._on_connect_stage)
        layout.addWidget(btn_stage, 1, 2)

        layout.setColumnStretch(3, 1)
        return group

    def _create_fft_group(self) -> QGroupBox:
        group = QGroupBox("FFT Analysis")
        layout = QGridLayout(group)
        layout.setSpacing(6)

        int_val = QIntValidator(0, 100000)

        # --- Row 0: peak selection -------------------------------------------
        col = 0
        self._center_idx_edit = QLineEdit("300")
        self._center_idx_edit.setValidator(int_val)
        col = self._add_field(layout, 0, col, "Center:", self._center_idx_edit, 70)

        self._window_edit = QLineEdit("5")
        self._window_edit.setValidator(int_val)
        col = self._add_field(layout, 0, col, "Half-win:", self._window_edit, 70)

        lbl_mode = QLabel("Mode:")
        lbl_mode.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(lbl_mode, 0, col)
        self._phase_mode_combo = QComboBox()
        self._phase_mode_combo.addItems([PHASE_MODE_SINGLE, PHASE_MODE_VECTOR])
        self._phase_mode_combo.setFixedWidth(110)
        layout.addWidget(self._phase_mode_combo, 0, col + 1)
        col += 2

        self._remove_ramp_checkbox = QCheckBox("Remove Ramp")
        self._remove_ramp_checkbox.setChecked(False)
        self._remove_ramp_checkbox.setToolTip(
            "Flatten the phase across the peak (analytic + fitted linear ramp). "
            "Correction is 1 at Center: the measured phase there is unchanged. "
            "The slope is fitted once and held; it is refitted only when Center "
            "or Half-win changes, on toggle, or via Refit Ramp."
        )
        self._remove_ramp_checkbox.stateChanged.connect(self._on_toggle_remove_ramp)
        layout.addWidget(self._remove_ramp_checkbox, 0, col, 1, 2)
        col += 2

        btn_refit = QPushButton("Refit Ramp")
        btn_refit.setFixedWidth(100)
        btn_refit.setToolTip("Re-fit the held ramp slope on the next frame.")
        btn_refit.clicked.connect(self._on_refit_ramp)
        layout.addWidget(btn_refit, 0, col, 1, 2)
        col += 2

        # --- Row 1: peak search ----------------------------------------------
        col = 0
        self._search_min_edit = QLineEdit("0")
        self._search_min_edit.setValidator(int_val)
        col = self._add_field(layout, 1, col, "Search min:", self._search_min_edit, 70)

        self._search_max_edit = QLineEdit("700")
        self._search_max_edit.setValidator(int_val)
        col = self._add_field(layout, 1, col, "Search max:", self._search_max_edit, 70)

        btn_find = QPushButton("Find Peak")
        btn_find.setFixedWidth(110)
        btn_find.setToolTip("Set Center to argmax(|FFT|) within the search range.")
        btn_find.clicked.connect(self._on_find_peak)
        layout.addWidget(btn_find, 1, col + 1)

        self._ramp_label = QLabel("Ramp slope = —")
        layout.addWidget(self._ramp_label, 1, col + 2, 1, 4)

        # --- Row 2: FFT axis limits ------------------------------------------
        col = 0
        self._fft_xmin_edit = QLineEdit("0")
        col = self._add_field(layout, 2, col, "FFT X min:", self._fft_xmin_edit, 70)
        self._fft_xmax_edit = QLineEdit("1000")
        col = self._add_field(layout, 2, col, "max:", self._fft_xmax_edit, 70)
        self._fft_ymin_edit = QLineEdit("0")
        col = self._add_field(layout, 2, col, "FFT Y min:", self._fft_ymin_edit, 70)
        self._fft_ymax_edit = QLineEdit("200000")
        col = self._add_field(layout, 2, col, "max:", self._fft_ymax_edit, 70)

        # --- Row 3: phase axis limits and drawing -----------------------------
        col = 0
        self._phase_ymin_edit = QLineEdit("-6")
        col = self._add_field(layout, 3, col, "Phase Y min:", self._phase_ymin_edit, 70)
        self._phase_ymax_edit = QLineEdit("6")
        col = self._add_field(layout, 3, col, "max:", self._phase_ymax_edit, 70)
        self._max_points_edit = QLineEdit("200")
        self._max_points_edit.setValidator(int_val)
        col = self._add_field(layout, 3, col, "Max pts:", self._max_points_edit, 70)
        self._plot_skip_edit = QLineEdit("5")
        self._plot_skip_edit.setValidator(int_val)
        col = self._add_field(layout, 3, col, "Plot every:", self._plot_skip_edit, 70)

        btn_update = QPushButton("Update Limits")
        btn_update.setFixedWidth(110)
        btn_update.clicked.connect(self._on_update_limits)
        layout.addWidget(btn_update, 3, col + 1)

        layout.setColumnStretch(14, 1)
        return group

    def _create_pid_group(self) -> QGroupBox:
        group = QGroupBox("PID Control")
        layout = QGridLayout(group)
        layout.setSpacing(6)

        dbl = QDoubleValidator()

        col = 0
        self._setpoint_edit = QLineEdit("0.0")
        self._setpoint_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "φ₀ [rad]:", self._setpoint_edit, 70)

        self._kp_edit = QLineEdit("0.1")
        self._kp_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "Kp:", self._kp_edit, 70)

        self._ki_edit = QLineEdit("0.0")
        self._ki_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "Ki:", self._ki_edit, 70)

        self._kd_edit = QLineEdit("0.0")
        self._kd_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "Kd:", self._kd_edit, 70)

        self._gain_edit = QLineEdit("-1.0")
        self._gain_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "Gain:", self._gain_edit, 70)

        self._max_step_edit = QLineEdit("0.05")
        self._max_step_edit.setValidator(dbl)
        col = self._add_field(layout, 0, col, "Max step:", self._max_step_edit, 70)

        # 0 = correct on every acquired frame; otherwise the loop still consumes
        # every frame (for unwrapping) but only actuates at this period.
        self._rate_edit = QLineEdit("20")
        self._rate_edit.setValidator(QIntValidator(0, 100000))
        col = self._add_field(layout, 0, col, "Rate [ms]:", self._rate_edit, 70)

        self._unwrap_checkbox = QCheckBox("Unwrap")
        self._unwrap_checkbox.setChecked(True)
        self._unwrap_checkbox.setToolTip(
            "Unwrap the phase in time between successive frames."
        )
        layout.addWidget(self._unwrap_checkbox, 1, 0, 1, 2)

        self._auto_reset_checkbox = QCheckBox("Auto Reset")
        self._auto_reset_checkbox.setChecked(False)
        layout.addWidget(self._auto_reset_checkbox, 1, 2, 1, 2)

        self._auto_lock_checkbox = QCheckBox("Auto Lock")
        self._auto_lock_checkbox.setChecked(False)
        self._auto_lock_checkbox.setToolTip(
            "On each new setpoint, fully drop the lock and re-engage it at the "
            "new setpoint (same as unticking then reticking LOCK by hand)."
        )
        layout.addWidget(self._auto_lock_checkbox, 1, 4, 1, 2)

        self._lock_checkbox = QCheckBox("LOCK")
        self._lock_checkbox.setStyleSheet("QCheckBox { font-weight: bold; color: blue; }")
        self._lock_checkbox.stateChanged.connect(self._on_toggle_lock)
        layout.addWidget(self._lock_checkbox, 1, 6, 1, 2)

        btn_update_sp = QPushButton("Update SP")
        btn_update_sp.setFixedWidth(100)
        btn_update_sp.clicked.connect(self._on_update_setpoint)
        layout.addWidget(btn_update_sp, 1, 8, 1, 2)

        btn_update_pid = QPushButton("Update PID")
        btn_update_pid.setFixedWidth(100)
        btn_update_pid.clicked.connect(self._on_update_pid)
        layout.addWidget(btn_update_pid, 1, 10, 1, 2)

        btn_reset_loop = QPushButton("Reset Loop")
        btn_reset_loop.setFixedWidth(100)
        btn_reset_loop.setToolTip(
            "Clear integrator/derivative/unwrap state without dropping the lock."
        )
        btn_reset_loop.clicked.connect(self._on_reset_loop)
        layout.addWidget(btn_reset_loop, 1, 12, 1, 2)

        layout.setColumnStretch(16, 1)
        return group

    def _create_voltage_group(self) -> QGroupBox:
        group = QGroupBox("Voltage Limits")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Min:"))
        self._vmin_edit = QLineEdit("75.0")
        self._vmin_edit.setFixedWidth(70)
        self._vmin_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._vmin_edit)

        layout.addWidget(QLabel("Max:"))
        self._vmax_edit = QLineEdit("85.0")
        self._vmax_edit.setFixedWidth(70)
        self._vmax_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._vmax_edit)

        layout.addWidget(QLabel("Start:"))
        self._vstart_edit = QLineEdit("80.0")
        self._vstart_edit.setFixedWidth(70)
        self._vstart_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._vstart_edit)

        btn_update = QPushButton("Update Limits")
        btn_update.setFixedWidth(110)
        btn_update.clicked.connect(self._on_update_voltage_limits)
        layout.addWidget(btn_update)

        layout.addStretch()
        return group

    def _create_modulation_group(self) -> QGroupBox:
        group = QGroupBox("Setpoint Modulation (sinus)")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Amp [rad]:"))
        self._mod_amp_edit = QLineEdit("1.0")
        self._mod_amp_edit.setFixedWidth(70)
        self._mod_amp_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._mod_amp_edit)

        layout.addWidget(QLabel("Period [s]:"))
        self._mod_period_edit = QLineEdit("10.0")
        self._mod_period_edit.setFixedWidth(70)
        self._mod_period_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._mod_period_edit)

        layout.addWidget(QLabel("Center [rad]:"))
        self._mod_center_edit = QLineEdit("0.0")
        self._mod_center_edit.setFixedWidth(70)
        self._mod_center_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._mod_center_edit)

        layout.addWidget(QLabel("Update [ms]:"))
        self._mod_update_edit = QLineEdit("50")
        self._mod_update_edit.setFixedWidth(60)
        self._mod_update_edit.setValidator(QIntValidator(10, 10000))
        layout.addWidget(self._mod_update_edit)

        self._mod_btn = QPushButton("Start Sinus")
        self._mod_btn.setFixedWidth(100)
        self._mod_btn.clicked.connect(self._on_toggle_modulation)
        layout.addWidget(self._mod_btn)

        layout.addStretch()
        return group

    def _create_control_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        btn_start = QPushButton("Start")
        btn_start.setFixedWidth(90)
        btn_start.clicked.connect(self._on_start)
        layout.addWidget(btn_start)

        btn_stop = QPushButton("Stop")
        btn_stop.setFixedWidth(90)
        btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(btn_stop)

        layout.addStretch(1)

        self._phase_label = QLabel("φ = — rad")
        self._phase_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        layout.addWidget(self._phase_label)

        layout.addWidget(QLabel(" | "))
        self._error_label = QLabel("Error = — rad")
        layout.addWidget(self._error_label)

        layout.addWidget(QLabel(" | "))
        self._voltage_label = QLabel("V = — ")
        layout.addWidget(self._voltage_label)

        layout.addWidget(QLabel(" | "))
        self._peak_label = QLabel("Peak = —")
        layout.addWidget(self._peak_label)

        return layout

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source=f"PhaseLock/{self._preset.label}")

    @staticmethod
    def _read_int(edit: QLineEdit, default: int) -> int:
        try:
            return int(edit.text())
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _read_float(edit: QLineEdit, default: float) -> float:
        try:
            return float(edit.text())
        except (ValueError, TypeError):
            return default

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_current_phase(self) -> float:
        """Most recent phase value, unwrapped in time if unwrapping is enabled."""
        if self._hist_phi_unwrapped:
            return float(self._hist_phi_unwrapped[-1])
        return float("nan")

    def get_phase_average(self, duration_s: float) -> tuple[float, float]:
        """Mean and std of the phase over the last *duration_s* seconds (timestamp based)."""
        if not self._hist_phi_unwrapped:
            return float("nan"), float("nan")
        t_cut = time.monotonic() - duration_s
        samples = [p for t, p in zip(self._hist_t, self._hist_phi_unwrapped) if t >= t_cut]
        if not samples:
            samples = [self._hist_phi_unwrapped[-1]]
        avg = float(np.mean(samples))
        std = float(np.std(samples)) if len(samples) > 1 else 0.0
        return avg, std

    def get_current_voltage(self) -> float:
        """Current piezo voltage."""
        if self._ctrl_thread is not None:
            return float(self._ctrl_thread.current_v)
        return float("nan")

    def get_ramp_slope(self) -> float:
        """Currently held ramp slope in rad/bin (0.0 when ramp removal is off)."""
        if not self._remove_ramp_checkbox.isChecked():
            return 0.0
        return float(self._ramp_slope)

    def refit_ramp(self) -> None:
        """Force a ramp refit on the next acquired frame."""
        self._invalidate_ramp()

    def set_target(self, target_rad: float) -> None:
            """Set the phase lock setpoint, optionally re-arming or relocking the loop."""
            if self._ctrl_thread is None:
                return
            target_rad = float(target_rad)
            if self._auto_lock_checkbox.isChecked():
                self._relock_at_new_setpoint(target_rad)
                return
            self._setpoint_edit.setText(f"{target_rad:.6f}")
            if self._auto_reset_checkbox.isChecked() and self._ctrl_thread.enabled:
                self._ctrl_thread.enabled = False
                self._ctrl_thread.target = target_rad
                self._ctrl_thread.reset()
                self._seed_unwrap_from_display()
                self._ctrl_thread.enabled = True
            else:
                self._ctrl_thread.target = target_rad

    def is_locked(self) -> bool:
        """True if the PID loop is actively driving the stage."""
        return self._ctrl_thread is not None and self._ctrl_thread.enabled

    def _seed_unwrap_from_display(self) -> None:
        """Keep the loop's unwrap chain consistent with the displayed phase."""
        if self._ctrl_thread is not None and self._hist_phi_unwrapped:
            self._ctrl_thread.seed_unwrap(self._hist_phi_unwrapped[-1])

    def _relock_at_new_setpoint(self, target_rad: float) -> None:
        """Fully drop the lock and re-engage it at target_rad, exactly as if the
        user unticked LOCK, changed the setpoint, and reticked it by hand."""
        self._setpoint_edit.setText(f"{target_rad:.6f}")
        if self._lock_checkbox.isChecked():
            self._lock_checkbox.setChecked(False)   # -> _on_toggle_lock: lock OFF
        if self._ctrl_thread is not None:
            self._ctrl_thread.target = float(target_rad)
        self._lock_checkbox.setChecked(True)        # -> _on_toggle_lock: lock ON, fresh

    # -------------------------------------------------------------------------
    # Device connection
    # -------------------------------------------------------------------------

    def _on_connect_spec(self) -> None:
        key = self._spec_key_edit.text().strip()
        ctrl = REGISTRY.get(key)
        if ctrl is None:
            QMessageBox.critical(self, "Spectrometer", f"Not found: {key}")
            return
        self._spec_ctrl = ctrl
        self._log_message(f"Connected spectrometer: {key}")

    def _on_connect_stage(self) -> None:
        key = self._stage_key_edit.text().strip()
        stage = REGISTRY.get(key)
        if stage is None:
            QMessageBox.critical(self, "NV40", f"Not found: {key}")
            return

        if self._ctrl_thread is not None:
            self._ctrl_thread.stop()
            self._ctrl_thread.wait(2000)

        thread = ControlThread(stage)
        thread.vmin = self._read_float(self._vmin_edit, 0.0)
        thread.vmax = self._read_float(self._vmax_edit, 140.0)

        v_start = self._read_float(self._vstart_edit, thread.current_v)
        v_start = max(thread.vmin, min(thread.vmax, v_start))
        try:
            stage.set_position(v_start)
            thread.current_v = v_start
            self._log_message(f"Stage set to {v_start:.2f} V")
        except Exception as exc:
            self._log_message(f"Could not preset stage: {exc}")

        thread.update_status.connect(self._on_control_update)
        thread.start()
        self._ctrl_thread = thread
        self._log_message(
            f"Connected NV40 ({thread.vmin:.1f}-{thread.vmax:.1f} V, "
            f"start {thread.current_v:.1f} V)"
        )

    # -------------------------------------------------------------------------
    # Parameter updates
    # -------------------------------------------------------------------------

    def _on_update_setpoint(self) -> None:
        if self._ctrl_thread is None:
            return
        new_sp = self._read_float(self._setpoint_edit, self._ctrl_thread.target)
        if self._auto_lock_checkbox.isChecked():
            self._relock_at_new_setpoint(new_sp)
            self._log_message(f"Setpoint {new_sp:.3f} rad (relocked)")
            return
        if self._auto_reset_checkbox.isChecked() and self._ctrl_thread.enabled:
            self._ctrl_thread.enabled = False
            self._ctrl_thread.target = new_sp
            self._ctrl_thread.reset()
            self._seed_unwrap_from_display()
            self._ctrl_thread.enabled = True
            self._log_message(f"Setpoint {new_sp:.3f} rad (loop re-armed)")
        else:
            self._ctrl_thread.target = new_sp
            self._log_message(f"Setpoint {new_sp:.3f} rad")

    def _on_update_pid(self) -> None:
        if self._ctrl_thread is None:
            self._log_message("No stage connected")
            return
        t = self._ctrl_thread
        t.kp = self._read_float(self._kp_edit, t.kp)
        t.ki = self._read_float(self._ki_edit, t.ki)
        t.kd = self._read_float(self._kd_edit, t.kd)
        t.gain = self._read_float(self._gain_edit, t.gain)
        t.max_step = abs(self._read_float(self._max_step_edit, t.max_step))
        t.target = self._read_float(self._setpoint_edit, t.target)
        t.unwrap = self._unwrap_checkbox.isChecked()
        t.period = max(0, self._read_int(self._rate_edit, 0)) / 1000.0
        t.vmin = self._read_float(self._vmin_edit, t.vmin)
        t.vmax = self._read_float(self._vmax_edit, t.vmax)
        self._log_message(
            f"PID updated: Kp={t.kp:g} Ki={t.ki:g} Kd={t.kd:g} "
            f"gain={t.gain:g} step={t.max_step:g} rate={t.period * 1000:.0f} ms"
        )

    def _on_reset_loop(self) -> None:
        """Clear integrator/derivative/unwrap state without dropping the lock."""
        if self._ctrl_thread is None:
            self._log_message("No stage connected")
            return
        self._ctrl_thread.reset()
        self._seed_unwrap_from_display()
        self._log_message("Loop state reset")

    def _on_update_limits(self) -> None:
        xmin = self._read_float(self._fft_xmin_edit, float("nan"))
        xmax = self._read_float(self._fft_xmax_edit, float("nan"))
        if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
            self._ax_fft.set_xlim(xmin, xmax)

        ymin = self._read_float(self._fft_ymin_edit, float("nan"))
        ymax = self._read_float(self._fft_ymax_edit, float("nan"))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            self._ax_fft.set_ylim(ymin, ymax)

        pmin = self._read_float(self._phase_ymin_edit, float("nan"))
        pmax = self._read_float(self._phase_ymax_edit, float("nan"))
        if np.isfinite(pmin) and np.isfinite(pmax) and pmax > pmin:
            self._ax_phase.set_ylim(pmin, pmax)

        new_max = max(2, self._read_int(self._max_points_edit, self._max_points))
        if new_max != self._max_points:
            self._max_points = new_max
            self._hist_phi_raw = deque(self._hist_phi_raw, maxlen=new_max)
            self._hist_phi_unwrapped = deque(self._hist_phi_unwrapped, maxlen=new_max)
            self._hist_t = deque(self._hist_t, maxlen=new_max)
            self._ax_phase.set_xlim(-1, new_max)

        self._blitting_initialized = False
        self._canvas.draw()
        self._log_message("Plot limits updated")

    def _on_update_voltage_limits(self) -> None:
        if self._ctrl_thread is None:
            self._log_message("No stage connected")
            return
        vmin = self._read_float(self._vmin_edit, self._ctrl_thread.vmin)
        vmax = self._read_float(self._vmax_edit, self._ctrl_thread.vmax)
        if vmax <= vmin:
            QMessageBox.critical(self, "Voltage", "Max must exceed min")
            return
        self._ctrl_thread.vmin = vmin
        self._ctrl_thread.vmax = vmax
        self._log_message(f"Voltage limits: {vmin:.1f}-{vmax:.1f} V")

    # -------------------------------------------------------------------------
    # Ramp removal
    # -------------------------------------------------------------------------

    def _invalidate_ramp(self) -> None:
        """Drop the cached slope so the next frame refits it."""
        self._ramp_key = None
        self._ramp_phasor = None

    def _apply_ramp(self, spectrum_fft: np.ndarray, center: int, half_window: int) -> np.ndarray:
        """Apply the held ramp correction, refitting the slope only when the peak selection changed or the cache was invalidated."""
        key = (center, half_window)
        stale = (
            self._ramp_key != key
            or self._ramp_phasor is None
            or self._ramp_phasor.size != spectrum_fft.size
        )
        if stale:
            self._ramp_slope = fit_ramp_slope(spectrum_fft, center, half_window)
            self._ramp_phasor = ramp_phasor(spectrum_fft.size, center, self._ramp_slope)
            self._ramp_key = key
            self._ramp_label.setText(f"Ramp slope = {self._ramp_slope:+.5f} rad/bin")
            self._log_message(
                f"Ramp slope fitted at bin {center} (half-win {half_window}): "
                f"{self._ramp_slope:+.5f} rad/bin"
            )
        return spectrum_fft * self._ramp_phasor

    def _on_toggle_remove_ramp(self) -> None:
        self._invalidate_ramp()
        if self._remove_ramp_checkbox.isChecked():
            self._log_message("Remove Ramp ON (slope will be fitted on next frame)")
        else:
            self._ramp_label.setText("Ramp slope = —")
            self._log_message("Remove Ramp OFF")

    def _on_refit_ramp(self) -> None:
        """Force a refit of the held slope on the next acquired frame."""
        if not self._remove_ramp_checkbox.isChecked():
            QMessageBox.information(self, "Refit Ramp", "Remove Ramp is off")
            return
        self._invalidate_ramp()
        self._log_message("Ramp refit requested")

    def _on_find_peak(self) -> None:
        """Set Center to the strongest FFT bin in the search range."""
        if self._last_mag is None:
            QMessageBox.information(self, "Find Peak", "No spectrum acquired yet")
            return

        mag = self._last_mag
        n = mag.size

        lo = max(3, self._read_int(self._search_min_edit, 0))  # keep clear of DC
        hi = min(n, self._read_int(self._search_max_edit, n // 2))
        if hi <= lo:
            QMessageBox.critical(self, "Find Peak", "Search max must exceed search min")
            return

        peak = int(lo + np.argmax(mag[lo:hi]))
        self._center_idx_edit.setText(str(peak))
        self._log_message(f"Peak found at bin {peak} (|F| = {mag[peak]:.3g})")

    # -------------------------------------------------------------------------
    # Acquisition control
    # -------------------------------------------------------------------------

    def _on_start(self) -> None:
        if self._spec_ctrl is None:
            QMessageBox.critical(self, "Run", "No spectrometer connected")
            return
        if self._acq_thread is not None:
            self._log_message("Acquisition already running")
            return

        self._max_points = max(2, self._read_int(self._max_points_edit, 100))
        self._hist_phi_raw = deque(maxlen=self._max_points)
        self._hist_phi_unwrapped = deque(maxlen=self._max_points)
        self._hist_t = deque(maxlen=self._max_points)

        self._invalidate_ramp()

        self._ax_fft.clear()
        self._ax_phase.clear()
        self._fft_scatter = None
        self._fft_marker = None
        self._phase_scatter = None
        self._phase_sp_line = None
        self._blitting_initialized = False

        self._acq_thread = AvaspecThread(self._spec_ctrl)
        self._acq_thread.data_ready.connect(self._on_data)
        self._acq_thread.error.connect(self._on_acq_error)
        self._acq_thread.start()
        self._log_message("Acquisition started")

    def _on_stop(self) -> None:
        if self._acq_thread is not None:
            self._acq_thread.stop()
            if not self._acq_thread.wait(2000):
                self._log_message("Acquisition thread did not stop cleanly")
            self._acq_thread = None
            self._log_message("Acquisition stopped")

        self._stop_modulation()
        if self._lock_checkbox.isChecked():
            self._lock_checkbox.setChecked(False)

    def _on_toggle_lock(self) -> None:
        if self._ctrl_thread is None:
            if self._lock_checkbox.isChecked():
                self._lock_checkbox.setChecked(False)
                QMessageBox.warning(self, "Lock", "NV40 not connected")
            return

        if self._lock_checkbox.isChecked():
            self._on_update_pid()
            self._ctrl_thread.reset()
            # Align the loop's unwrap chain with the displayed phase, so a
            # setpoint typed from the plot means the same thing to the loop.
            self._seed_unwrap_from_display()
            self._ctrl_thread.enabled = True
            self._log_message("Lock ON")
        else:
            self._ctrl_thread.enabled = False
            self._stop_modulation()
            self._log_message("Lock OFF")
            if self._phase_sp_line is not None:
                self._phase_sp_line.set_visible(False)
                self._canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Setpoint modulation (sinus)
    # -------------------------------------------------------------------------

    def _on_toggle_modulation(self) -> None:
        if self._mod_active:
            self._stop_modulation()
            return

        if self._ctrl_thread is None:
            QMessageBox.warning(self, "Modulation", "NV40 not connected")
            return
        if not self._ctrl_thread.enabled:
            QMessageBox.warning(self, "Modulation", "Enable LOCK first")
            return

        amp = self._read_float(self._mod_amp_edit, float("nan"))
        period = self._read_float(self._mod_period_edit, float("nan"))
        center = self._read_float(self._mod_center_edit, float("nan"))
        update_ms = self._read_int(self._mod_update_edit, 50)

        if not all(np.isfinite([amp, period, center])) or period <= 0:
            QMessageBox.critical(self, "Modulation", "Invalid parameters")
            return

        self._mod_amp = amp
        self._mod_period = period
        self._mod_center = center
        self._mod_t0 = time.monotonic()
        self._mod_active = True
        self._mod_timer.start(max(10, update_ms))
        self._mod_btn.setText("Stop Sinus")
        self._log_message(
            f"Sinus modulation: {center:.3f} + {amp:.3f}·sin(2πt/{period:g}) rad, "
            f"update {update_ms} ms"
        )

    def _stop_modulation(self) -> None:
        if not self._mod_active:
            return
        self._mod_active = False
        self._mod_timer.stop()
        self._mod_btn.setText("Start Sinus")
        self._log_message("Sinus modulation stopped")

    def _on_mod_step(self) -> None:
        if not self._mod_active or self._ctrl_thread is None:
            return
        t = time.monotonic() - self._mod_t0
        sp = self._mod_center + self._mod_amp * np.sin(2.0 * np.pi * t / self._mod_period)
        # Write the target directly: set_target() with Auto Reset on would
        # reset the integrator every tick and kill the tracking.
        self._ctrl_thread.target = float(sp)
        self._setpoint_edit.setText(f"{sp:.4f}")

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def _on_acq_error(self, err: str) -> None:
        self._log_message(f"Acquisition error: {err}")
        self._on_stop()

    def _on_control_update(self, phi: float, v: float) -> None:
        self._voltage_label.setText(f"V = {v:.3f}")

    def _on_data(self, wl, y) -> None:
        # wl unused: the phase is read from a fixed bin on the pixel axis.
        if self._acq_thread is None:
            return

        y = np.asarray(y, float)
        if y.size < 32 or not np.all(np.isfinite(y)):
            return

        center = self._read_int(self._center_idx_edit, -1)
        half_window = max(0, self._read_int(self._window_edit, 5))
        mode = self._phase_mode_combo.currentText()

        # Mean removed so DC does not dominate the magnitude plot.
        spectrum_fft = fft_deskew(y - y.mean())

        # The fitted slope is held: it is recomputed only when the peak
        # selection changes or the cache is explicitly invalidated.
        if self._remove_ramp_checkbox.isChecked():
            spectrum_fft = self._apply_ramp(spectrum_fft, center, half_window)

        mag = np.abs(spectrum_fft)
        phase = np.angle(spectrum_fft)
        self._last_mag = mag
        n = mag.size

        phi = extract_phase(spectrum_fft, center, half_window, mode)
        if not np.isfinite(phi):
            return

        # Unwrap in time, across frames.
        if self._unwrap_checkbox.isChecked() and self._hist_phi_unwrapped:
            phi_u = float(np.unwrap([self._hist_phi_unwrapped[-1], phi])[-1])
        else:
            phi_u = phi

        self._hist_phi_raw.append(phi)
        self._hist_phi_unwrapped.append(phi_u)
        self._hist_t.append(time.monotonic())

        # Every frame feeds the loop; the loop rate-limits its own corrections.
        if self._ctrl_thread is not None and self._ctrl_thread.enabled:
            self._ctrl_thread.q.append(phi)

        self._phase_label.setText(f"φ = {phi_u:+.3f} rad")
        self._peak_label.setText(f"Peak = {center} (|F| = {mag[center]:.3g})"
                                 if 0 <= center < n else "Peak = out of range")

        if self._ctrl_thread is not None and self._ctrl_thread.enabled:
            sp = self._read_float(self._setpoint_edit, float("nan"))
            if np.isfinite(sp):
                self._error_label.setText(f"Error = {abs(phi_u - sp):.4f} rad")
        else:
            self._error_label.setText("Error = — rad")

        now = time.monotonic()
        if now - self._last_draw < self._min_draw_dt:
            return
        self._last_draw = now

        self._redraw(mag, phase, center, half_window, mode, n)

    def _redraw(
        self,
        mag: np.ndarray,
        phase: np.ndarray,
        center: int,
        half_window: int,
        mode: str,
        n: int,
    ) -> None:
        """Blitted update of the FFT and phase-history axes."""
        i0 = max(0, min(self._read_int(self._fft_xmin_edit, 0), n - 1))
        i1 = max(i0 + 1, min(self._read_int(self._fft_xmax_edit, 1000), n))
        skip = max(1, self._read_int(self._plot_skip_edit, 5))

        x = np.arange(i0, i1, skip)
        mag_dec = mag[i0:i1:skip]
        phase_dec = phase[i0:i1:skip]

        if self._fft_scatter is None:
            self._fft_scatter = self._ax_fft.scatter(
                x,
                mag_dec,
                c=phase_dec,
                cmap="hsv",
                s=20,
                vmin=-np.pi,
                vmax=np.pi,
                animated=True,
            )
            self._ax_fft.set_xlabel("FFT bin")
            self._ax_fft.set_ylabel("Magnitude")
            self._ax_fft.set_xlim(i0, i1)
            ymin = self._read_float(self._fft_ymin_edit, 0.0)
            ymax = self._read_float(self._fft_ymax_edit, 200000.0)
            if ymax > ymin:
                self._ax_fft.set_ylim(ymin, ymax)
        else:
            self._fft_scatter.set_offsets(np.column_stack((x, mag_dec)))
            self._fft_scatter.set_array(phase_dec)

        if self._fft_marker is not None:
            self._fft_marker.remove()
        if mode == PHASE_MODE_VECTOR and half_window > 0:
            self._fft_marker = self._ax_fft.axvspan(
                center - half_window,
                center + half_window,
                alpha=0.25,
                color="orange",
                animated=True,
            )
        else:
            self._fft_marker = self._ax_fft.axvline(
                center, color="red", linestyle="--", animated=True
            )

        use_unwrapped = self._unwrap_checkbox.isChecked()
        hist = self._hist_phi_unwrapped if use_unwrapped else self._hist_phi_raw
        pp = np.asarray(hist, float)
        idx = np.arange(pp.size)
        colours = (pp + np.pi) % (2 * np.pi) - np.pi  # wrapped colour, matches FFT panel

        if self._phase_scatter is None:
            self._phase_scatter = self._ax_phase.scatter(
                idx,
                pp,
                c=colours,
                cmap="hsv",
                s=14,
                vmin=-np.pi,
                vmax=np.pi,
                animated=True,
            )
            self._ax_phase.set_xlabel("Sample")
            self._ax_phase.set_ylabel("Phase [rad]")
            self._ax_phase.set_xlim(-1, self._max_points)
            pmin = self._read_float(self._phase_ymin_edit, -6.0)
            pmax = self._read_float(self._phase_ymax_edit, 6.0)
            if pmax > pmin:
                self._ax_phase.set_ylim(pmin, pmax)
        else:
            self._phase_scatter.set_offsets(np.column_stack((idx, pp)))
            self._phase_scatter.set_array(colours)

        locked = self._ctrl_thread is not None and self._ctrl_thread.enabled
        if locked:
            sp = self._read_float(self._setpoint_edit, float("nan"))
            if np.isfinite(sp):
                if self._phase_sp_line is None:
                    self._phase_sp_line = self._ax_phase.axhline(
                        sp, color="black", linestyle="--", lw=1.2, animated=True
                    )
                else:
                    self._phase_sp_line.set_ydata([sp, sp])
                    self._phase_sp_line.set_visible(True)
        elif self._phase_sp_line is not None:
            self._phase_sp_line.set_visible(False)

        if not self._blitting_initialized:
            self._canvas.draw()
            self._bg_fft = self._canvas.copy_from_bbox(self._ax_fft.bbox)
            self._bg_phase = self._canvas.copy_from_bbox(self._ax_phase.bbox)
            self._blitting_initialized = True

        self._canvas.restore_region(self._bg_fft)
        self._ax_fft.draw_artist(self._fft_scatter)
        self._ax_fft.draw_artist(self._fft_marker)
        self._canvas.blit(self._ax_fft.bbox)

        self._canvas.restore_region(self._bg_phase)
        self._ax_phase.draw_artist(self._phase_scatter)
        if self._phase_sp_line is not None and self._phase_sp_line.get_visible():
            self._ax_phase.draw_artist(self._phase_sp_line)
        self._canvas.blit(self._ax_phase.bbox)

    # -------------------------------------------------------------------------
    # Qt events
    # -------------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        # Cached backgrounds go stale on resize and corrupt the blit.
        self._blitting_initialized = False
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        self._stop_modulation()
        self._on_stop()
        if self._ctrl_thread is not None:
            self._ctrl_thread.stop()
            self._ctrl_thread.wait(2000)
            self._ctrl_thread = None
        try:
            REGISTRY.unregister(self._registry_key)
        except Exception:
            pass
        plt.close(self._fig)
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AvaspecPhaseLockWindow(instance_id="2w")
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec_())