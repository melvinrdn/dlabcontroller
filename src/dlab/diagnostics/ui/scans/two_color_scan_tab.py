from __future__ import annotations

import datetime
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit, QCheckBox,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import logging

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.phase_settings import PhaseSettings
from dlab.utils.paths_utils import data_dir, cfg_get
from dlab.diagnostics.ui.scans.scan_utils import (
    confirm_large_scan,
    generate_positions,
    save_png_with_meta,
)

logger = logging.getLogger("dlab.scans.two_color_scan_tab")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Fixed SLM class driving the IR two-foci shaping. A is the useful peak that
# overlaps with the 2omega beam; beta_a dumps intensity out of A.
# Base case: alpha = 0 -> single focus A with dump A', I_A = I_max * (1 - beta_a).
# Two-foci mode: alpha > 0 -> second focus B on the table, controlled by beta_b.
SLM_CLASS_NAME = "TwoFociStochastic"
SLM_FIELD_BETA_A = "le_beta_a"
SLM_FIELD_BETA_B = "le_beta_b"
SLM_FIELD_ALPHA = "le_alpha"
DEFAULT_ALPHA = 0.0
SPECTRUM_MEASUREMENT_DELAY_S = 0.01

# Numerical margin when checking clip conditions (W/cm2 relative tolerance).
CLIP_TOL = 1e-9


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
# Helper functions - Waveplate calibration (mirrors grid_scan_tab)
# -----------------------------------------------------------------------------


def power_to_angle(power_fraction: float, phase_deg: float) -> float:
    """Convert power fraction (0-1) to waveplate angle using calibration phase."""
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0


def _wp_index_from_stage_key(stage_key: str) -> int | None:
    """Extract waveplate index from stage key like 'stage:3'."""
    num_waveplates = int(cfg_get("waveplates.num_waveplates", 7))
    try:
        if not stage_key.startswith("stage:"):
            return None
        n = int(stage_key.split(":")[1])
        if 1 <= n <= num_waveplates:
            return n
    except (ValueError, IndexError):
        pass
    return None


def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"


def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"


def _reg_key_maxvalue(wp_index: int) -> str:
    return f"waveplate:max_value:{wp_index}"


# -----------------------------------------------------------------------------
# Helper functions - Peak intensity
# -----------------------------------------------------------------------------


def calculate_max_intensity(
    max_power_W: float, waist_um: float, pulse_duration_fs: float, rep_rate_kHz: float
) -> float:
    """Peak intensity [W/cm2] for a Gaussian beam at the given average power."""
    waist_cm = waist_um * 1e-4
    pulse_duration_s = pulse_duration_fs * 1e-15
    rep_rate_Hz = rep_rate_kHz * 1e3

    P_peak = max_power_W / (rep_rate_Hz * pulse_duration_s)
    area_cm2 = np.pi * waist_cm**2
    return 2.0 * P_peak / area_cm2


def intensity_to_power(
    I_peak_W_cm2: float, waist_um: float, pulse_duration_fs: float, rep_rate_kHz: float
) -> float:
    """Average power [W] required to reach a target peak intensity."""
    waist_cm = waist_um * 1e-4
    pulse_duration_s = pulse_duration_fs * 1e-15
    rep_rate_Hz = rep_rate_kHz * 1e3

    area_cm2 = np.pi * waist_cm**2
    P_peak = I_peak_W_cm2 * area_cm2 / 2.0
    return P_peak * rep_rate_Hz * pulse_duration_s


def beta_a_for_intensity(I_omega_target: float, I_max_omega: float, alpha: float) -> float:
    """Invert I_A = I_max * (1 - alpha) * (1 - beta_a) for beta_a, clipped to [0, 1]."""
    denom = I_max_omega * (1.0 - alpha)
    if denom <= 0:
        return 1.0
    return float(np.clip(1.0 - I_omega_target / denom, 0.0, 1.0))


def max_conservable_itot(
    R_start: float, R_end: float, I_max_omega: float, I_max_2omega: float
) -> Tuple[float, str]:
    """Highest I_tot that keeps the whole [R_start, R_end] range conservable.

    With I_tot conserved, I_A = (1 - R) * I_tot and I_2omega = R * I_tot. Neither
    arm may exceed its hardware max anywhere in the range:

        IR    : I_A = (1 - R) * I_tot <= I_max_omega   -> binds at R_start
        green : I_2omega = R * I_tot  <= I_max_2omega  -> binds at R_end

    So I_tot_max = min(I_max_omega / (1 - R_start), I_max_2omega / R_end).
    HHG generation is the user's call at setup and is not enforced here.

    Returns (I_tot_max, binding_arm). R_start/R_end are ordered internally.
    """
    R_lo, R_hi = (R_start, R_end) if R_start <= R_end else (R_end, R_start)
    itot_ir = I_max_omega / (1.0 - R_lo) if R_lo < 1.0 else float("inf")
    itot_green = I_max_2omega / R_hi if R_hi > 0.0 else float("inf")
    if itot_ir <= itot_green:
        return itot_ir, "IR arm"
    return itot_green, "green arm"


# -----------------------------------------------------------------------------
# Monitor window (R vs phi acquisition-time map)
# -----------------------------------------------------------------------------


class MonitorWindow(QWidget):
    """Live R vs phi map coloured by how long each point took to acquire."""

    def __init__(self, ratio_values, setpoints, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Monitor - Acquisition Time per Point")
        self.resize(700, 500)

        self.ratio_values = np.array(ratio_values)
        self.setpoints = np.array(setpoints)
        self.n_ratios = len(ratio_values)
        self.n_phases = len(setpoints)

        self.time_data = np.full((self.n_ratios, self.n_phases), np.nan)

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        extent = [
            self.setpoints[0], self.setpoints[-1],
            self.ratio_values[0], self.ratio_values[-1],
        ]
        self.im = self.ax.imshow(
            self.time_data, aspect="auto", origin="lower",
            extent=extent, cmap="viridis", interpolation="nearest",
        )
        self.ax.set_xlabel("Phase setpoint [rad]")
        self.ax.set_ylabel("Ratio R")
        self.ax.set_title("Acquisition time per point [s]")
        self.cbar = self.figure.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("seconds")

        self.figure.tight_layout()
        layout.addWidget(self.canvas)

    def update_data(self, ratio_idx: int, phase_idx: int, elapsed_s: float):
        """Record the time one point took and refresh the map.

        phase_idx always indexes self.setpoints in its original order, so a
        serpentine sweep colours the same column as a forward sweep would.
        """
        self.time_data[ratio_idx, phase_idx] = elapsed_s
        self.im.set_data(self.time_data)

        valid = self.time_data[~np.isnan(self.time_data)]
        if len(valid) > 0:
            vmax = np.percentile(valid, 95) if len(valid) > 1 else float(valid[0])
            self.im.set_clim(vmin=0, vmax=max(vmax, 1e-3))

        self.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------


class TwoColorScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    monitor_update = pyqtSignal(int, int, float)  # ratio_idx, phase_idx, elapsed_s

    def __init__(
        self,
        phase_ctrl_key: str,
        setpoints: List[float],
        detector_params: Dict[str, tuple],
        max_phase_error_rad: float,
        max_phase_std_rad: float,
        stability_samples: int,
        scan_name: str,
        comment: str,
        # Ratio scan
        ratio_values: List[float],
        total_intensity_W_cm2: float,
        # SLM IR control
        slm_screen: int,
        alpha: float,
        beta_b_fixed: float,
        # IR (omega) laser parameters
        omega_max_power_W: float,
        omega_waist_um: float,
        omega_pulse_duration_fs: float,
        omega_rep_rate_kHz: float,
        omega_beam_split: bool,
        omega_beam_split_ratio: float,  # fraction in B beam (A gets 1 - ratio)
        # 2-omega waveplate + laser parameters
        wp_2omega_key: str,
        omega2_max_power_W: float,
        omega2_waist_um: float,
        omega2_pulse_duration_fs: float,
        omega2_rep_rate_kHz: float,
        # Phase sweep direction
        serpentine: bool = False,
        # Background
        background: bool = False,
        existing_scan_log: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.phase_ctrl_key = phase_ctrl_key
        self.setpoints = setpoints
        self.serpentine = bool(serpentine)
        self.detector_params = detector_params
        self.max_phase_error_rad = float(max_phase_error_rad)
        self.max_phase_std_rad = float(max_phase_std_rad)
        self.stability_samples = max(2, int(stability_samples))
        self.scan_name = scan_name
        self.comment = comment
        self.abort = False
        # Set by the GUI to give up waiting on the current point and shoot now.
        self.skip_point = False

        self.ratio_values = ratio_values
        self.total_intensity_W_cm2 = float(total_intensity_W_cm2)

        self.slm_screen = int(slm_screen)
        self.alpha = float(alpha)
        self.beta_b_fixed = float(beta_b_fixed)

        self.omega_max_power_W = omega_max_power_W
        self.omega_waist_um = omega_waist_um
        self.omega_pulse_duration_fs = omega_pulse_duration_fs
        self.omega_rep_rate_kHz = omega_rep_rate_kHz
        self.omega_beam_split = bool(omega_beam_split)
        self.omega_beam_split_ratio = float(omega_beam_split_ratio)

        self.wp_2omega_key = wp_2omega_key
        self.omega2_max_power_W = omega2_max_power_W
        self.omega2_waist_um = omega2_waist_um
        self.omega2_pulse_duration_fs = omega2_pulse_duration_fs
        self.omega2_rep_rate_kHz = omega2_rep_rate_kHz

        self.background = bool(background)
        self.existing_scan_log = existing_scan_log

        self.data_root = data_dir()
        self.timestamp = datetime.datetime.now()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    # -------------------------------------------------------------------------
    # Intensity model
    # -------------------------------------------------------------------------

    def _I_max_omega(self) -> float:
        """Max peak intensity available for IR peak A (W/cm2), after beam split."""
        I_max = calculate_max_intensity(
            self.omega_max_power_W, self.omega_waist_um,
            self.omega_pulse_duration_fs, self.omega_rep_rate_kHz,
        )
        if self.omega_beam_split:
            # beam_split_ratio is fraction in B, so A gets (1 - ratio)
            I_max *= (1.0 - self.omega_beam_split_ratio)
        return I_max

    def _beta_a_for_intensity(self, I_omega_target: float, I_max_omega: float) -> float:
        return beta_a_for_intensity(I_omega_target, I_max_omega, self.alpha)

    # -------------------------------------------------------------------------
    # Phase sweep ordering
    # -------------------------------------------------------------------------

    def _phase_loop(self, ratio_idx: int) -> List[Tuple[int, float]]:
        """(phase_idx, setpoint) pairs for this ratio row.

        In serpentine mode odd rows are swept backwards, so the sweep resumes
        where the previous row ended instead of jumping the full span back to
        the start of the ramp. phase_idx always refers to the position in
        self.setpoints, so logs and the monitor stay in the original order.
        """
        pairs = list(enumerate(self.setpoints))
        if self.serpentine and ratio_idx % 2 == 1:
            pairs.reverse()
        return pairs

    # -------------------------------------------------------------------------
    # SLM control (mirrors grid_scan_tab SLM path)
    # -------------------------------------------------------------------------

    def _publish_slm(self, beta_a: float) -> None:
        """Push alpha, beta_a, beta_b into the TwoFociStochastic widget and publish."""
        _publish_slm(self.alpha, beta_a, self.beta_b_fixed, self.slm_screen)

    # -------------------------------------------------------------------------
    # 2-omega waveplate control
    # -------------------------------------------------------------------------

    def _set_waveplate_power(self, stage_key: str, power_W: float, max_power_W: float) -> None:
        """Set the 2omega waveplate to achieve a target average power."""
        angle, frac = _set_waveplate_power(stage_key, power_W, max_power_W)
        self._emit(
            f"  {stage_key} -> {power_W:.6f} W / {max_power_W:.6f} W "
            f"({100 * frac:.1f}%, angle: {angle:.3f} deg)"
        )

    # -------------------------------------------------------------------------
    # Phase stability
    # -------------------------------------------------------------------------

    def _wait_for_stability(self, phase_ctrl, setpoint: float) -> Tuple[float, float, float, bool, bool]:
        """Block until the phase criterion holds, or until a skip is requested.

        Each acquisition is one get_current_phase() reading (a single value,
        no time averaging). The criterion requires BOTH:
          - |setpoint - last_sample| < max_phase_error_rad   (current sample only)
          - std(last X samples)      < max_phase_std_rad      (window, wrap-aware)
        Deviations are wrapped to [-pi, pi] so the +/- pi boundary is handled.

        There is NO timeout. The point is taken only when the criterion passes,
        unless the user presses Skip Point (self.skip_point), which gives up
        waiting and returns the current values as-is so the image is taken
        anyway. Abort exits without taking anything.

        Returns (measured_phase, std, error, skipped, aborted).
        """
        start_time = time.time()
        check_count = 0
        recent: deque = deque(maxlen=self.stability_samples)

        def _wrap(x: float) -> float:
            return float(np.angle(np.exp(1j * x)))

        def _window_std() -> float:
            dev = np.angle(np.exp(1j * (np.asarray(recent) - setpoint)))
            return float(np.std(dev))

        while True:
            if self.abort:
                return 0.0, 0.0, 0.0, False, True

            sample = float(phase_ctrl.get_current_phase())
            recent.append(sample)
            check_count += 1
            elapsed = time.time() - start_time

            # Error is the current sample's deviation from setpoint.
            phase_error = abs(_wrap(sample - setpoint))
            std_phase = _window_std() if len(recent) == self.stability_samples else float("nan")

            # User gave up on this point: take it now with whatever we have.
            if self.skip_point:
                self.skip_point = False
                if not np.isfinite(std_phase):
                    std_phase = _window_std() if len(recent) > 1 else 0.0
                self._emit(
                    f"  SKIP: taking point after {elapsed:.2f}s / {check_count} acquisitions "
                    f"without meeting criterion (error={phase_error:.4f}, std={std_phase:.4f})"
                )
                return sample, std_phase, phase_error, True, False

            if len(recent) == self.stability_samples:
                if phase_error < self.max_phase_error_rad and std_phase < self.max_phase_std_rad:
                    self._emit(
                        f"  Phase stable after {elapsed:.2f}s / {check_count} acquisitions "
                        f"(error={phase_error:.4f} rad on last sample, "
                        f"std={std_phase:.4f} rad over last {self.stability_samples}); "
                        f"measured phase={sample:.4f} rad"
                    )
                    # measured_phase = last raw reading, not a window mean.
                    return sample, std_phase, phase_error, False, False

                if check_count % 3 == 0:
                    self._emit(
                        f"  Waiting for stability... ({elapsed:.1f}s) "
                        f"error={phase_error:.4f}, std={std_phase:.4f}"
                    )
            time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Scan log
    # -------------------------------------------------------------------------

    def _create_scan_log(self, scan_dir: Path) -> Path:
        if self.existing_scan_log:
            return Path(self.existing_scan_log)

        date_str = f"{self.timestamp:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        scan_log = candidate

        I_max_omega = self._I_max_omega()
        I_max_2omega = calculate_max_intensity(
            self.omega2_max_power_W, self.omega2_waist_um,
            self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz,
        )

        header = [
            "Ratio_R", "beta_a", "Power_2omega_W",
            "I_A_peak_W_cm2", "I_2omega_peak_W_cm2",
            "Setpoint_rad", "MeasuredPhase_rad", "PhaseStd_rad", "PhaseError_rad",
            "DetectorKey", "DataFile", "Exposure_us", "Averages",
        ]
        with open(scan_log, "w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            f.write(f"# {self.comment}\n")
            f.write("# Ratio scan: R = I_2omega / (I_A + I_2omega), A = useful IR peak\n")
            f.write(f"# Total peak intensity: {self.total_intensity_W_cm2:.6e} W/cm2\n")
            f.write(f"# Phase stability: max_error={self.max_phase_error_rad} rad, "
                    f"max_std={self.max_phase_std_rad} rad\n")
            f.write(f"#   samples={self.stability_samples} (single value per acquisition)\n")
            f.write(f"# Serpentine phase sweep: {self.serpentine} "
                    f"(odd ratio rows swept in reverse; rows are written in "
                    f"acquisition order)\n")
            f.write(f"# SLM IR: {SLM_CLASS_NAME} | alpha={self.alpha:.4f} | "
                    f"beta_b(fixed)={self.beta_b_fixed:.4f} | screen={self.slm_screen}\n")
            f.write(f"#   omega: P={self.omega_max_power_W} W, waist={self.omega_waist_um} um, "
                    f"tau={self.omega_pulse_duration_fs} fs, rep={self.omega_rep_rate_kHz} kHz\n")
            if self.omega_beam_split:
                f.write(f"#   beam split: B fraction={self.omega_beam_split_ratio:.3f}, "
                        f"A gets {1.0 - self.omega_beam_split_ratio:.3f}\n")
            f.write(f"#   I_max_A: {I_max_omega:.6e} W/cm2\n")
            f.write(f"# Waveplate 2omega: {self.wp_2omega_key} | P={self.omega2_max_power_W} W, "
                    f"waist={self.omega2_waist_um} um, tau={self.omega2_pulse_duration_fs} fs, "
                    f"rep={self.omega2_rep_rate_kHz} kHz\n")
            f.write(f"#   I_max_2omega: {I_max_2omega:.6e} W/cm2\n")
        return scan_log

    # -------------------------------------------------------------------------
    # Detector capture (Andor MCP camera only)
    # -------------------------------------------------------------------------

    def _capture_camera(
        self, det_key: str, dev, params: tuple, meta_extra: dict
    ) -> Tuple[str, str]:
        exposure_us = int(params[0]) if len(params) >= 1 else 0
        averages = int(params[1]) if len(params) >= 2 else 1

        try:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
                exposure_us=int(exposure_us),
            )
        except TypeError:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
            )

        exp_meta = int((meta or {}).get("Exposure_us", exposure_us))
        det_name = _detector_display_name(det_key, dev, meta)
        det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / det_name
        ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = "Background" if self.background else "Image"
        fn = f"{det_name}_{tag}_{ts_ms}.png"

        file_meta = dict(meta) if meta else {}
        file_meta.update(meta_extra)
        file_meta.update({"Exposure_us": exp_meta, "Comment": self.comment})

        frame_out = np.clip(frame, 0, 65535).astype(np.uint16, copy=False)
        save_png_with_meta(det_day, fn, frame_out, file_meta)
        return fn, f"exp {exp_meta} us"

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        # Phase controller (skip for background)
        phase_ctrl = None
        if not self.background:
            phase_ctrl = REGISTRY.get(self.phase_ctrl_key)
            if phase_ctrl is None:
                self._emit(f"Phase controller '{self.phase_ctrl_key}' not found.")
                self.finished.emit("")
                return
            if not hasattr(phase_ctrl, "set_target"):
                self._emit("Phase controller doesn't expose required API.")
                self.finished.emit("")
                return
            if not phase_ctrl.is_locked():
                self._emit("Phase controller is not locked. Enable lock first.")
                self.finished.emit("")
                return

        # Detectors (Andor MCP camera only)
        detectors = {}
        for det_key, params in self.detector_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                self._emit(f"Detector '{det_key}' not found.")
                self.finished.emit("")
                return
            if not hasattr(dev, "grab_frame_for_scan"):
                self._emit(f"Detector '{det_key}' is not a scan camera.")
                self.finished.emit("")
                return
            try:
                exposure = int(params[0])
                if hasattr(dev, "set_exposure_us"):
                    dev.set_exposure_us(exposure)
                elif hasattr(dev, "setExposureUS"):
                    dev.setExposureUS(exposure)
                elif hasattr(dev, "set_exposure"):
                    dev.set_exposure(exposure)
            except Exception as e:
                self._emit(f"Warning: failed to preset on '{det_key}': {e}")
            detectors[det_key] = dev

        scan_dir = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "TwoColorScans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)
        scan_log = self._create_scan_log(scan_dir)

        I_max_omega = self._I_max_omega()
        I_max_2omega = calculate_max_intensity(
            self.omega2_max_power_W, self.omega2_waist_um,
            self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz,
        )

        if not self.background and self.serpentine:
            self._emit("Serpentine phase sweep enabled: odd R rows are swept in reverse.")

        n_ratios = len(self.ratio_values)
        n_phases = len(self.setpoints) if not self.background else 1
        total_points = n_ratios * n_phases * len(detectors)
        done = 0

        try:
            for ratio_idx, ratio_R in enumerate(self.ratio_values):
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return

                # Target intensities for this R, total intensity held constant.
                I_2omega = ratio_R * self.total_intensity_W_cm2
                I_A = (1.0 - ratio_R) * self.total_intensity_W_cm2

                # Hard guard: if either arm would clip, I_tot is NOT conserved at
                # this R. Abort instead of silently writing a corrupted point.
                # The UI is supposed to prevent this upstream; this is the net.
                if I_A > I_max_omega * (1.0 + CLIP_TOL) or I_2omega > I_max_2omega * (1.0 + CLIP_TOL):
                    self._emit(
                        f"ABORT: R={ratio_R:.4f} not conservable at "
                        f"I_tot={self.total_intensity_W_cm2:.3e}.\n"
                        f"  needs I_A={I_A:.3e} (max {I_max_omega:.3e}), "
                        f"I_2omega={I_2omega:.3e} (max {I_max_2omega:.3e}).\n"
                        f"  Lower I_tot or narrow the R range so both arms stay within limits."
                    )
                    self.finished.emit("")
                    return

                beta_a = self._beta_a_for_intensity(I_A, I_max_omega)
                power_2omega = intensity_to_power(
                    I_2omega, self.omega2_waist_um,
                    self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz,
                )

                self._emit(f"\n=== R = {ratio_R:.4f} ===")
                self._emit(f"  I_A:      {I_A:.6e} W/cm2 ({100 * (1 - ratio_R):.1f}%)")
                self._emit(f"  I_2omega: {I_2omega:.6e} W/cm2 ({100 * ratio_R:.1f}%)")
                self._emit(f"  beta_a:   {beta_a:.6f}")
                self._emit(f"  P_2omega: {power_2omega:.6f} W")

                try:
                    self._publish_slm(beta_a)
                    self._set_waveplate_power(
                        self.wp_2omega_key, power_2omega, self.omega2_max_power_W
                    )
                except Exception as e:
                    self._emit(f"Failed to set IR/2omega for R={ratio_R:.3f}: {e}")
                    self.finished.emit("")
                    return

                meta_ratio = {
                    "Ratio_R": ratio_R,
                    "beta_a": beta_a,
                    "alpha": self.alpha,
                    "beta_b": self.beta_b_fixed,
                    "I_A_peak_W_cm2": I_A,
                    "I_2omega_peak_W_cm2": I_2omega,
                    "Power_2omega_W": power_2omega,
                }

                # Serpentine: reverse the sweep on odd rows so the next point
                # starts where the previous one ended instead of jumping the
                # whole span back to the start of the ramp. phase_idx keeps
                # indexing self.setpoints in its original order.
                if self.background:
                    phase_loop: List[Tuple[int, float | None]] = [(0, None)]
                else:
                    phase_loop = self._phase_loop(ratio_idx)
                    direction = "reverse" if (self.serpentine and ratio_idx % 2 == 1) else "forward"
                    self._emit(
                        f"  Phase sweep direction: {direction} "
                        f"({phase_loop[0][1]:.4f} -> {phase_loop[-1][1]:.4f} rad)"
                    )

                for phase_idx, sp in phase_loop:
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    point_t0 = time.time()
                    measured_phase = std_phase = phase_error = None
                    if not self.background:
                        # Fresh skip flag for each point.
                        self.skip_point = False
                        phase_ctrl.set_target(float(sp))
                        self._emit(f"R={ratio_R:.3f}, phase setpoint: {sp:.4f} rad")
                        measured_phase, std_phase, phase_error, skipped, aborted = \
                            self._wait_for_stability(phase_ctrl, sp)
                        if aborted or self.abort:
                            self._emit("Scan aborted.")
                            self.finished.emit("")
                            return
                        elapsed = time.time() - point_t0
                        self.monitor_update.emit(ratio_idx, phase_idx, elapsed)
                    else:
                        self._emit(f"Capturing background @ R={ratio_R:.3f}")

                    meta_point = dict(meta_ratio)
                    if not self.background:
                        meta_point.update({
                            "Setpoint_rad": sp,
                            "MeasuredPhase_rad": measured_phase,
                            "PhaseStd_rad": std_phase,
                        })

                    for det_key, dev in detectors.items():
                        if self.abort:
                            self._emit("Scan aborted.")
                            self.finished.emit("")
                            return

                        params = self.detector_params.get(det_key, (0, 1))
                        try:
                            data_fn, saved_label = self._capture_camera(
                                det_key, dev, params, meta_point
                            )

                            row = [
                                f"{float(ratio_R):.9f}",
                                f"{float(beta_a):.9f}",
                                f"{float(power_2omega):.9f}",
                                f"{float(I_A):.9e}",
                                f"{float(I_2omega):.9e}",
                            ]
                            if not self.background:
                                row += [
                                    f"{float(sp):.9f}",
                                    f"{float(measured_phase):.9f}",
                                    f"{float(std_phase):.9f}",
                                    f"{float(phase_error):.9f}",
                                ]
                            else:
                                row += ["", "", "", ""]
                            row += [
                                det_key,
                                data_fn,
                                str(params[0] if len(params) >= 1 else ""),
                                str(params[1] if len(params) >= 2 else ""),
                            ]
                            with open(scan_log, "a", encoding="utf-8") as f:
                                f.write("\t".join(row) + "\n")

                            if self.background:
                                self._emit(f"Saved {data_fn} @ R={ratio_R:.3f} BACKGROUND ({saved_label})")
                            else:
                                self._emit(
                                    f"Saved {data_fn} @ R={ratio_R:.3f}, SP={sp:.4f} rad, "
                                    f"Phase={measured_phase:.4f}+/-{std_phase:.4f} rad ({saved_label})"
                                )
                        except Exception as e:
                            self._emit(f"Capture failed @ R={ratio_R:.3f} on {det_key}: {e}")

                        done += 1
                        self.progress.emit(done, total_points)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self._emit(traceback.format_exc())
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


# -----------------------------------------------------------------------------
# Shared hardware-driving helpers (used by the worker AND the live Set Rmin/Rmax
# buttons, which act completely outside any scan).
# -----------------------------------------------------------------------------


def _publish_slm(alpha: float, beta_a: float, beta_b: float, slm_screen: int) -> None:
    """Push alpha, beta_a, beta_b into the TwoFociStochastic widget and publish."""
    active_classes = REGISTRY.get("slm:red:active_classes") or []
    if SLM_CLASS_NAME not in active_classes:
        raise RuntimeError(
            f"SLM class '{SLM_CLASS_NAME}' is not active.\nActive: {active_classes}"
        )

    widgets = REGISTRY.get("slm:red:widgets") or []
    phase_widget = None
    for w in widgets:
        if getattr(w, "name_", lambda: "")() == SLM_CLASS_NAME:
            phase_widget = w
            break
    if phase_widget is None:
        raise RuntimeError(f"SLM widget for '{SLM_CLASS_NAME}' not found.")

    for field in (SLM_FIELD_ALPHA, SLM_FIELD_BETA_A, SLM_FIELD_BETA_B):
        if not hasattr(phase_widget, field):
            raise RuntimeError(f"Field '{field}' not found in '{SLM_CLASS_NAME}'.")

    getattr(phase_widget, SLM_FIELD_ALPHA).setText(f"{alpha:.6f}")
    getattr(phase_widget, SLM_FIELD_BETA_A).setText(f"{beta_a:.6f}")
    getattr(phase_widget, SLM_FIELD_BETA_B).setText(f"{beta_b:.6f}")

    slm_window = REGISTRY.get("slm:red:window")
    if slm_window is None:
        raise RuntimeError("SLM window not registered.")
    levels = slm_window.compose_levels()

    slm_red = REGISTRY.get("slm:red:controller")
    if slm_red is None:
        raise RuntimeError("Red SLM controller not found.")
    slm_red.publish(levels, screen_num=int(slm_screen))


def _set_waveplate_power(stage_key: str, power_W: float, max_power_W: float) -> Tuple[float, float]:
    """Move the 2omega waveplate to reach a target average power.

    Returns (angle_deg, power_fraction) so the caller can log it.
    """
    wp_index = _wp_index_from_stage_key(stage_key)
    if wp_index is None:
        raise ValueError(f"Invalid waveplate key: {stage_key}")

    amp_off = REGISTRY.get(_reg_key_calib(wp_index)) or (None, None)
    if amp_off[1] is None:
        raise ValueError(f"{stage_key}: No calibration phase found.")
    phase_deg = float(amp_off[1])

    REGISTRY.register(_reg_key_maxvalue(wp_index), float(max_power_W))

    power_fraction = float(np.clip(power_W / float(max_power_W), 0.0, 1.0))
    angle = power_to_angle(power_fraction, phase_deg)

    stage = REGISTRY.get(stage_key)
    if stage is None:
        raise ValueError(f"Stage not found: {stage_key}")
    stage.move_to(float(angle), blocking=True)
    return angle, power_fraction


# -----------------------------------------------------------------------------
# TwoColorScanTab
# -----------------------------------------------------------------------------


class TwoColorScanTab(QWidget):
    def __init__(self, log_panel=None, parent=None):
        super().__init__(parent)
        self._log_panel = log_panel
        self._thread = None
        self._worker = None
        self._monitor_window = None
        self._doing_background = False
        self._cached_params = None
        self._last_scan_log_path = None
        self._scan_ok = True
        # Guard: True while we programmatically write I_tot, so its textChanged
        # handler doesn't fight the user's own edits.
        self._updating_itot = False
        self._build_ui()
        self._refresh_devices()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Phase controller
        ctrl_box = QGroupBox("Phase Lock Controller")
        ctrl_l = QHBoxLayout(ctrl_box)
        self.phase_ctrl_picker = QComboBox()
        ctrl_l.addWidget(QLabel("Controller:"))
        ctrl_l.addWidget(self.phase_ctrl_picker, 1)
        main.addWidget(ctrl_box)

        # Ratio scan model
        ratio_box = QGroupBox("Ratio Scan: R = I_2omega / (I_A + I_2omega)")
        ratio_l = QVBoxLayout(ratio_box)

        # --- Beam maxima readout (computed from IR + green params) ---
        maxima_row = QHBoxLayout()
        self.intensity_max_label = QLabel("I_max_IR: --   I_max_green: --")
        self.intensity_max_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        maxima_row.addWidget(self.intensity_max_label)
        maxima_row.addStretch()
        ratio_l.addLayout(maxima_row)

        # Omega (IR) beam parameters
        omega_box = QGroupBox("Omega (IR) beam parameters")
        omega_l = QVBoxLayout(omega_box)

        omega_row1 = QHBoxLayout()
        self.omega_max_power_le = QLineEdit("")
        self.omega_max_power_le.textChanged.connect(self._on_beam_or_range_changed)
        self.omega_waist_le = QLineEdit("22")
        self.omega_waist_le.textChanged.connect(self._on_beam_or_range_changed)
        omega_row1.addWidget(QLabel("Max power (W):"))
        omega_row1.addWidget(self.omega_max_power_le)
        omega_row1.addWidget(QLabel("Waist at focus (um):"))
        omega_row1.addWidget(self.omega_waist_le)
        omega_l.addLayout(omega_row1)

        omega_row2 = QHBoxLayout()
        self.omega_pulse_duration_le = QLineEdit("220")
        self.omega_pulse_duration_le.textChanged.connect(self._on_beam_or_range_changed)
        self.omega_rep_rate_le = QLineEdit("4")
        self.omega_rep_rate_le.textChanged.connect(self._on_beam_or_range_changed)
        omega_row2.addWidget(QLabel("Pulse duration (fs):"))
        omega_row2.addWidget(self.omega_pulse_duration_le)
        omega_row2.addWidget(QLabel("Rep rate (kHz):"))
        omega_row2.addWidget(self.omega_rep_rate_le)
        omega_l.addLayout(omega_row2)

        omega_row3 = QHBoxLayout()
        self.omega_beam_split_cb = QCheckBox("Beam split (A/B configuration)")
        self.omega_beam_split_cb.toggled.connect(self._on_beam_or_range_changed)
        self.omega_beam_split_ratio_le = QLineEdit("0.5")
        self.omega_beam_split_ratio_le.setEnabled(False)
        self.omega_beam_split_ratio_le.textChanged.connect(self._on_beam_or_range_changed)
        self.omega_beam_split_cb.toggled.connect(self.omega_beam_split_ratio_le.setEnabled)
        omega_row3.addWidget(self.omega_beam_split_cb)
        omega_row3.addWidget(QLabel("Fraction in B beam:"))
        omega_row3.addWidget(self.omega_beam_split_ratio_le)
        omega_row3.addWidget(QLabel("(A gets 1 - fraction)"))
        omega_row3.addStretch()
        omega_l.addLayout(omega_row3)
        ratio_l.addWidget(omega_box)

        # 2-Omega waveplate + beam parameters
        omega2_box = QGroupBox("2-Omega (green) waveplate + beam parameters")
        omega2_l = QVBoxLayout(omega2_box)

        wp2_row = QHBoxLayout()
        self.wp_2omega_picker = QComboBox()
        wp2_row.addWidget(QLabel("Waveplate 2omega:"))
        wp2_row.addWidget(self.wp_2omega_picker, 1)
        omega2_l.addLayout(wp2_row)

        omega2_row1 = QHBoxLayout()
        self.omega2_max_power_le = QLineEdit("")
        self.omega2_max_power_le.textChanged.connect(self._on_beam_or_range_changed)
        self.omega2_waist_le = QLineEdit("20")
        self.omega2_waist_le.textChanged.connect(self._on_beam_or_range_changed)
        omega2_row1.addWidget(QLabel("Max power (W):"))
        omega2_row1.addWidget(self.omega2_max_power_le)
        omega2_row1.addWidget(QLabel("Waist at focus (um):"))
        omega2_row1.addWidget(self.omega2_waist_le)
        omega2_l.addLayout(omega2_row1)

        omega2_row2 = QHBoxLayout()
        self.omega2_pulse_duration_le = QLineEdit("200")
        self.omega2_pulse_duration_le.textChanged.connect(self._on_beam_or_range_changed)
        self.omega2_rep_rate_le = QLineEdit("4")
        self.omega2_rep_rate_le.textChanged.connect(self._on_beam_or_range_changed)
        omega2_row2.addWidget(QLabel("Pulse duration (fs):"))
        omega2_row2.addWidget(self.omega2_pulse_duration_le)
        omega2_row2.addWidget(QLabel("Rep rate (kHz):"))
        omega2_row2.addWidget(self.omega2_rep_rate_le)
        omega2_l.addLayout(omega2_row2)
        ratio_l.addWidget(omega2_box)

        # SLM IR shaping parameters
        slm_row = QHBoxLayout()
        self.alpha_le = QLineEdit(f"{DEFAULT_ALPHA}")
        self.alpha_le.textChanged.connect(self._on_beam_or_range_changed)
        self.beta_b_le = QLineEdit("0.0")
        self.slm_screen_le = QLineEdit("3")
        slm_row.addWidget(QLabel("alpha (A/B split, 0 = single focus + dump):"))
        slm_row.addWidget(self.alpha_le)
        slm_row.addWidget(QLabel("beta_b (fixed dump B):"))
        slm_row.addWidget(self.beta_b_le)
        slm_row.addWidget(QLabel("SLM screen:"))
        slm_row.addWidget(self.slm_screen_le)
        slm_row.addStretch()
        ratio_l.addWidget(self._wrap_row(slm_row, f"IR shaping ({SLM_CLASS_NAME}, peak A useful)"))

        # Ratio range -> drives the auto-filled I_tot
        ratio_params = QHBoxLayout()
        self.ratio_start = QLineEdit("0.0")
        self.ratio_start.textChanged.connect(self._on_beam_or_range_changed)
        self.ratio_end = QLineEdit("0.9")
        self.ratio_end.textChanged.connect(self._on_beam_or_range_changed)
        self.ratio_step = QLineEdit("0.05")
        ratio_params.addWidget(QLabel("Ratio R start:"))
        ratio_params.addWidget(self.ratio_start)
        ratio_params.addWidget(QLabel("end:"))
        ratio_params.addWidget(self.ratio_end)
        ratio_params.addWidget(QLabel("step:"))
        ratio_params.addWidget(self.ratio_step)
        # Live mixing set for signal check (completely outside any scan).
        self.set_rmin_btn = QPushButton("Set Rmin")
        self.set_rmin_btn.setToolTip(
            "Apply the IR/green mixing for R start right now (SLM + waveplate), "
            "so you can check for signal before scanning. Not part of the scan."
        )
        self.set_rmin_btn.clicked.connect(lambda: self._apply_mixing_for_edit(self.ratio_start))
        self.set_rmax_btn = QPushButton("Set Rmax")
        self.set_rmax_btn.setToolTip(
            "Apply the IR/green mixing for R end right now (SLM + waveplate), "
            "so you can check for signal before scanning. Not part of the scan."
        )
        self.set_rmax_btn.clicked.connect(lambda: self._apply_mixing_for_edit(self.ratio_end))
        ratio_params.addWidget(self.set_rmin_btn)
        ratio_params.addWidget(self.set_rmax_btn)
        ratio_l.addLayout(ratio_params)

        # I_tot: auto-filled to the max conservable value, but editable so you
        # can run a series at lower I_tot. Editing it re-validates the range.
        itot_row = QHBoxLayout()
        self.total_intensity_le = QLineEdit("")
        self.total_intensity_le.textChanged.connect(self._on_itot_edited)
        self.reset_itot_btn = QPushButton("Reset to max")
        self.reset_itot_btn.clicked.connect(self._fill_itot_max)
        itot_row.addWidget(QLabel("Total peak intensity I_tot (W/cm2):"))
        itot_row.addWidget(self.total_intensity_le, 1)
        itot_row.addWidget(self.reset_itot_btn)
        ratio_l.addLayout(itot_row)

        # Conservability readout: per-arm intensities at both ends + validity.
        self.ratio_range_label = QLabel("Conservable range: --")
        self.ratio_range_label.setStyleSheet("QLabel { color: darkblue; font-weight: bold; }")
        self.ratio_range_label.setWordWrap(True)
        ratio_l.addWidget(self.ratio_range_label)

        main.addWidget(ratio_box)

        # Phase setpoints
        sp_box = QGroupBox("Phase Setpoints")
        sp_l = QHBoxLayout(sp_box)
        self.sp_start = QLineEdit("-3.14159")
        self.sp_end = QLineEdit("3.14159")
        self.sp_step = QLineEdit("0.2")
        sp_l.addWidget(QLabel("Start [rad]:"))
        sp_l.addWidget(self.sp_start)
        sp_l.addWidget(QLabel("End [rad]:"))
        sp_l.addWidget(self.sp_end)
        sp_l.addWidget(QLabel("Step [rad]:"))
        sp_l.addWidget(self.sp_step)

        self.serpentine_cb = QCheckBox("Serpentine")
        self.serpentine_cb.setChecked(True)
        self.serpentine_cb.setToolTip(
            "Alternate the sweep direction between successive R values:\n"
            "  -pi -> pi, change R, pi -> -pi, change R, -pi -> pi, ...\n"
            "Avoids waiting for the lock to travel the whole span back to the "
            "start of each ramp. Rows are written to the log in acquisition "
            "order; the setpoint column still tells you which point is which."
        )
        sp_l.addWidget(self.serpentine_cb)
        main.addWidget(sp_box)

        # Detectors (Andor MCP camera)
        det_box = QGroupBox("Detector (MCP camera)")
        det_l = QVBoxLayout(det_box)
        det_pick = QHBoxLayout()
        self.det_picker = QComboBox()
        self.add_det_btn = QPushButton("Add Detector")
        self.add_det_btn.clicked.connect(self._add_det_row)
        det_pick.addWidget(QLabel("Detector:"))
        det_pick.addWidget(self.det_picker, 1)
        det_pick.addWidget(self.add_det_btn)
        det_l.addLayout(det_pick)

        self.det_tbl = QTableWidget(0, 3)
        self.det_tbl.setHorizontalHeaderLabels(["DetectorKey", "Exposure_us", "Averages"])
        self.det_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.det_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        det_l.addWidget(self.det_tbl)

        rm_det = QHBoxLayout()
        self.rm_det_btn = QPushButton("Remove Selected")
        self.rm_det_btn.clicked.connect(self._remove_det_row)
        rm_det.addStretch(1)
        rm_det.addWidget(self.rm_det_btn)
        det_l.addLayout(rm_det)
        main.addWidget(det_box)

        # Scan parameters
        params_box = QGroupBox("Scan Parameters")
        params_l = QVBoxLayout(params_box)

        stab_row1 = QHBoxLayout()
        self.max_phase_error_sb = QDoubleSpinBox()
        self.max_phase_error_sb.setDecimals(4)
        self.max_phase_error_sb.setRange(0.0001, 1.0)
        self.max_phase_error_sb.setValue(0.1)
        self.max_phase_error_sb.setSingleStep(0.01)
        self.max_phase_std_sb = QDoubleSpinBox()
        self.max_phase_std_sb.setDecimals(4)
        self.max_phase_std_sb.setRange(0.0001, 1.0)
        self.max_phase_std_sb.setValue(0.1)
        self.max_phase_std_sb.setSingleStep(0.01)
        stab_row1.addWidget(QLabel("Max phase error (rad):"))
        stab_row1.addWidget(self.max_phase_error_sb)
        stab_row1.addWidget(QLabel("Max phase std (rad):"))
        stab_row1.addWidget(self.max_phase_std_sb)
        params_l.addLayout(stab_row1)

        stab_row2 = QHBoxLayout()
        self.stability_samples_sb = QSpinBox()
        self.stability_samples_sb.setRange(2, 200)
        self.stability_samples_sb.setValue(5)
        stab_row2.addWidget(QLabel("Stability samples X:"))
        stab_row2.addWidget(self.stability_samples_sb)
        stab_row2.addStretch()
        params_l.addLayout(stab_row2)

        other_row = QHBoxLayout()
        self.scan_name = QLineEdit("")
        self.comment = QLineEdit("")
        other_row.addWidget(QLabel("Scan name:"))
        other_row.addWidget(self.scan_name, 1)
        other_row.addWidget(QLabel("Comment:"))
        other_row.addWidget(self.comment, 2)
        params_l.addLayout(other_row)
        main.addWidget(params_box)

        # Controls
        ctl = QHBoxLayout()
        self.estimate_btn = QPushButton("Estimate Scan Time")
        self.estimate_btn.clicked.connect(self._estimate_time)
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.clicked.connect(self._start)
        self.monitor_btn = QPushButton("Open Monitor")
        self.monitor_btn.clicked.connect(self._open_monitor)
        self.monitor_btn.setEnabled(False)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort)
        self.skip_btn = QPushButton("Skip Point")
        self.skip_btn.setToolTip(
            "Give up waiting for the phase criterion on the current point and "
            "take the picture now with whatever phase is measured."
        )
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self._skip_point)
        self.prog = QProgressBar()
        self.prog.setMinimum(0)
        self.prog.setValue(0)
        ctl.addWidget(self.estimate_btn)
        ctl.addWidget(self.start_btn)
        ctl.addWidget(self.monitor_btn)
        ctl.addWidget(self.abort_btn)
        ctl.addWidget(self.skip_btn)
        ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

        # Refresh
        rr = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        rr.addStretch(1)
        rr.addWidget(self.refresh_btn)
        main.addLayout(rr)

        self._on_beam_or_range_changed()

    def _wrap_row(self, inner_layout, title: str) -> QGroupBox:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addLayout(inner_layout)
        return box

    # -------------------------------------------------------------------------
    # Beam maxima
    # -------------------------------------------------------------------------

    def _beam_I_max(self) -> Tuple[float, float] | None:
        """Parse beam fields and return (I_max_A, I_max_2omega), or None if invalid.

        Silent variant used by live-update handlers: no popups.
        """
        try:
            I_max_omega = calculate_max_intensity(
                float(self.omega_max_power_le.text()), float(self.omega_waist_le.text()),
                float(self.omega_pulse_duration_le.text()), float(self.omega_rep_rate_le.text()),
            )
            if self.omega_beam_split_cb.isChecked():
                split = float(np.clip(float(self.omega_beam_split_ratio_le.text()), 0.0, 1.0))
                I_max_omega *= (1.0 - split)
            I_max_2omega = calculate_max_intensity(
                float(self.omega2_max_power_le.text()), float(self.omega2_waist_le.text()),
                float(self.omega2_pulse_duration_le.text()), float(self.omega2_rep_rate_le.text()),
            )
        except (ValueError, ZeroDivisionError):
            return None
        if I_max_omega <= 0 or I_max_2omega <= 0:
            return None
        return I_max_omega, I_max_2omega

    # -------------------------------------------------------------------------
    # I_tot auto-fill + conservability readout
    # -------------------------------------------------------------------------

    def _on_beam_or_range_changed(self):
        """Beam params or R range changed -> refresh maxima, refill I_tot to max."""
        beams = self._beam_I_max()
        if beams is None:
            self.intensity_max_label.setText("I_max_IR: --   I_max_green: --")
            self.ratio_range_label.setText("Conservable range: -- (fill valid beam params)")
            self._scan_ok = False
            return
        I_max_ir, I_max_green = beams
        self.intensity_max_label.setText(
            f"I_max_IR: {I_max_ir:.3e}   I_max_green: {I_max_green:.3e} W/cm2"
        )
        self._fill_itot_max()

    def _fill_itot_max(self):
        """Write I_tot with the highest conservable value for the current R range."""
        beams = self._beam_I_max()
        if beams is None:
            return
        I_max_ir, I_max_green = beams
        try:
            R0 = float(self.ratio_start.text())
            R1 = float(self.ratio_end.text())
        except ValueError:
            self._validate_range()
            return

        I_tot_max, binding = max_conservable_itot(R0, R1, I_max_ir, I_max_green)
        if not np.isfinite(I_tot_max):
            self._validate_range()
            return

        self._updating_itot = True
        self.total_intensity_le.setText(f"{I_tot_max:.4e}")
        self._updating_itot = False
        self._validate_range()

    def _on_itot_edited(self):
        """User (or auto-fill) changed I_tot. Skip if we wrote it ourselves."""
        if self._updating_itot:
            return
        self._validate_range()

    def _validate_range(self):
        """Check the current (I_tot, R range) against hardware limits and update
        the readout. Sets self._scan_ok. Shows per-arm intensities at both ends.
        """
        beams = self._beam_I_max()
        if beams is None:
            self.ratio_range_label.setText("Conservable range: -- (fill valid beam params)")
            self._scan_ok = False
            return
        I_max_ir, I_max_green = beams

        try:
            I_tot = float(self.total_intensity_le.text())
            R0 = float(self.ratio_start.text())
            R1 = float(self.ratio_end.text())
        except ValueError:
            self.ratio_range_label.setText("Conservable range: -- (invalid I_tot or R)")
            self._scan_ok = False
            return

        if I_tot <= 0:
            self.ratio_range_label.setText("I_tot must be > 0")
            self._scan_ok = False
            return

        R_lo, R_hi = min(R0, R1), max(R0, R1)

        # Conservable R window at this I_tot (no clip on either arm):
        #   IR    : (1 - R) * I_tot <= I_max_ir     -> R >= 1 - I_max_ir/I_tot
        #   green : R * I_tot       <= I_max_green   -> R <= I_max_green/I_tot
        R_min = max(0.0, 1.0 - I_max_ir / I_tot)
        R_max = min(1.0, I_max_green / I_tot)

        # Per-arm intensities at both ends of the requested range.
        I_A_lo, I_A_hi = (1.0 - R_lo) * I_tot, (1.0 - R_hi) * I_tot
        I_g_lo, I_g_hi = R_lo * I_tot, R_hi * I_tot

        ok = (R_lo >= R_min - CLIP_TOL) and (R_hi <= R_max + CLIP_TOL)
        self._scan_ok = ok

        status = "OK, I_tot conserved" if ok else "OUT OF RANGE - will clip"
        self.ratio_range_label.setText(
            f"I_tot = {I_tot:.3e} W/cm2   |   conservable R at this I_tot: "
            f"[{R_min:.3f}, {R_max:.3f}]   |   {status}\n"
            f"  R={R_lo:.2f}:  I_A={I_A_lo:.2e}  I_2w={I_g_lo:.2e}   "
            f"|   R={R_hi:.2f}:  I_A={I_A_hi:.2e}  I_2w={I_g_hi:.2e}"
        )
        self.ratio_range_label.setStyleSheet(
            "QLabel { color: darkgreen; }" if ok
            else "QLabel { color: red; font-weight: bold; }"
        )

    # -------------------------------------------------------------------------
    # Live mixing set for signal check (Set Rmin / Set Rmax) - NOT part of a scan
    # -------------------------------------------------------------------------

    def _apply_mixing_for_edit(self, ratio_edit: QLineEdit):
        """Drive SLM + waveplate to the mixing for the R typed in `ratio_edit`.

        Uses the exact same commands the scan would, but runs live from the GUI
        so you can look for signal on the MCP before starting. Reads the current
        beam params and I_tot straight from the fields.
        """
        if self._thread is not None:
            QMessageBox.warning(
                self, "Busy", "A scan is running. Stop it before setting a mixing manually."
            )
            return

        beams = self._beam_I_max()
        if beams is None:
            QMessageBox.critical(self, "Beam params", "Fill in valid IR and green beam params first.")
            return
        I_max_ir, I_max_green = beams

        try:
            ratio_R = float(ratio_edit.text())
            I_tot = float(self.total_intensity_le.text())
            alpha = float(self.alpha_le.text())
            beta_b = float(np.clip(float(self.beta_b_le.text()), 0.0, 1.0))
            slm_screen = int(self.slm_screen_le.text())
        except (ValueError, TypeError):
            QMessageBox.critical(self, "Parameters", "Invalid R, I_tot, alpha, beta_b or screen.")
            return

        wp_key = self.wp_2omega_picker.currentText().strip()
        if not wp_key:
            QMessageBox.critical(self, "Waveplate", "Select the 2-omega waveplate.")
            return

        omega2_max_power = self._read_float(self.omega2_max_power_le)
        omega2_waist = self._read_float(self.omega2_waist_le)
        omega2_tau = self._read_float(self.omega2_pulse_duration_le)
        omega2_rep = self._read_float(self.omega2_rep_rate_le)
        if None in (omega2_max_power, omega2_waist, omega2_tau, omega2_rep):
            QMessageBox.critical(self, "Green params", "Invalid green beam parameters.")
            return

        if not (0.0 <= ratio_R <= 1.0):
            QMessageBox.critical(self, "Ratio", "R must be between 0 and 1.")
            return

        I_2omega = ratio_R * I_tot
        I_A = (1.0 - ratio_R) * I_tot

        if I_A > I_max_ir * (1.0 + CLIP_TOL) or I_2omega > I_max_green * (1.0 + CLIP_TOL):
            QMessageBox.critical(
                self, "Not conservable",
                f"At R={ratio_R:.3f}, I_tot={I_tot:.3e}:\n"
                f"I_A={I_A:.3e} (max {I_max_ir:.3e}), I_2w={I_2omega:.3e} (max {I_max_green:.3e}).\n"
                f"Lower I_tot or pick an R inside the conservable range.",
            )
            return

        beta_a = beta_a_for_intensity(I_A, I_max_ir, alpha)
        power_2omega = intensity_to_power(I_2omega, omega2_waist, omega2_tau, omega2_rep)

        try:
            _publish_slm(alpha, beta_a, beta_b, slm_screen)
            angle, frac = _set_waveplate_power(wp_key, power_2omega, omega2_max_power)
        except Exception as e:
            QMessageBox.critical(self, "Set mixing failed", str(e))
            self._log(f"Set mixing failed @ R={ratio_R:.3f}: {e}")
            return

        self._log(
            f"Set mixing @ R={ratio_R:.3f}: beta_a={beta_a:.4f}, "
            f"P_2w={power_2omega:.4f} W ({100 * frac:.1f}%, angle {angle:.2f} deg), "
            f"I_A={I_A:.2e}, I_2w={I_2omega:.2e} W/cm2. Check for signal now."
        )

    @staticmethod
    def _read_float(edit: QLineEdit):
        try:
            return float(edit.text())
        except (ValueError, TypeError):
            return None

    # -------------------------------------------------------------------------
    # Detector table
    # -------------------------------------------------------------------------

    def _add_det_row(self):
        det_key = self.det_picker.currentText().strip()
        if not det_key:
            QMessageBox.warning(self, "Pick a detector", "Select a detector to add.")
            return
        r = self.det_tbl.rowCount()
        self.det_tbl.insertRow(r)
        self.det_tbl.setItem(r, 0, QTableWidgetItem(det_key))
        self.det_tbl.setItem(r, 1, QTableWidgetItem("500000"))
        self.det_tbl.setItem(r, 2, QTableWidgetItem("1"))

    def _remove_det_row(self):
        rows = sorted({i.row() for i in self.det_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.det_tbl.removeRow(r)

    # -------------------------------------------------------------------------
    # Device discovery
    # -------------------------------------------------------------------------

    def _refresh_devices(self):
        self.phase_ctrl_picker.clear()
        for k in REGISTRY.keys("phaselock:"):
            self.phase_ctrl_picker.addItem(k)

        self.det_picker.clear()
        for k in REGISTRY.keys("camera:andor:"):
            if ":index:" not in k:
                self.det_picker.addItem(k)

        self.wp_2omega_picker.clear()
        for k in REGISTRY.keys("stage:"):
            if not k.startswith("stage:serial:"):
                self.wp_2omega_picker.addItem(k)

    # -------------------------------------------------------------------------
    # Parameter collection
    # -------------------------------------------------------------------------

    def _collect_params(self, validate=True):
        phase_ctrl_key = self.phase_ctrl_picker.currentText().strip()
        if validate and not phase_ctrl_key:
            raise ValueError("Select a phase lock controller.")

        try:
            setpoints = generate_positions(
                float(self.sp_start.text()), float(self.sp_end.text()), float(self.sp_step.text())
            )
        except ValueError as e:
            raise ValueError(f"Invalid setpoint parameters: {e}")

        if self.det_tbl.rowCount() == 0:
            raise ValueError("Add the MCP camera as a detector.")
        detector_params = {}
        for r in range(self.det_tbl.rowCount()):
            det = (self.det_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not det:
                raise ValueError(f"Empty detector key at row {r + 1}.")
            p1 = (self.det_tbl.item(r, 1) or QTableWidgetItem("0")).text()
            p2 = (self.det_tbl.item(r, 2) or QTableWidgetItem("1")).text()
            detector_params[det] = (int(float(p1)), int(float(p2)))

        name = self.scan_name.text().strip()
        if validate and not name:
            raise ValueError("Enter a scan name.")

        # Ratio model parameters
        def _pf(le, label, positive=True):
            try:
                v = float(le.text())
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError(f"{label} required and must be a number.")
                return None
            if validate and positive and v <= 0:
                raise ValueError(f"{label} must be positive.")
            return v

        total_intensity = _pf(self.total_intensity_le, "Total intensity")
        omega_max_power = _pf(self.omega_max_power_le, "Omega max power")
        omega_waist = _pf(self.omega_waist_le, "Omega waist")
        omega_tau = _pf(self.omega_pulse_duration_le, "Omega pulse duration")
        omega_rep = _pf(self.omega_rep_rate_le, "Omega rep rate")
        omega2_max_power = _pf(self.omega2_max_power_le, "2-omega max power")
        omega2_waist = _pf(self.omega2_waist_le, "2-omega waist")
        omega2_tau = _pf(self.omega2_pulse_duration_le, "2-omega pulse duration")
        omega2_rep = _pf(self.omega2_rep_rate_le, "2-omega rep rate")

        try:
            alpha = float(self.alpha_le.text())
            if validate and not (0.0 <= alpha < 1.0):
                raise ValueError("alpha must be in [0, 1).")
        except ValueError:
            if validate:
                raise ValueError("Invalid alpha.")
            alpha = DEFAULT_ALPHA

        try:
            beta_b = float(np.clip(float(self.beta_b_le.text()), 0.0, 1.0))
        except (ValueError, AttributeError):
            beta_b = 0.0

        try:
            slm_screen = int(self.slm_screen_le.text())
        except ValueError:
            slm_screen = 3

        try:
            ratio_values = generate_positions(
                float(self.ratio_start.text()), float(self.ratio_end.text()),
                float(self.ratio_step.text()),
            )
            if validate and any(r < 0 or r > 1 for r in ratio_values):
                raise ValueError("Ratio values must be between 0 and 1.")
        except ValueError as e:
            raise ValueError(f"Invalid ratio parameters: {e}")

        omega_beam_split = self.omega_beam_split_cb.isChecked()
        try:
            omega_beam_split_ratio = float(np.clip(
                float(self.omega_beam_split_ratio_le.text()), 0.0, 1.0
            ))
        except (ValueError, AttributeError):
            omega_beam_split_ratio = 0.5

        wp_2omega_key = self.wp_2omega_picker.currentText().strip()
        if validate and not wp_2omega_key:
            raise ValueError("Select the 2-omega waveplate.")
        if validate and wp_2omega_key:
            wp_idx = _wp_index_from_stage_key(wp_2omega_key)
            if wp_idx is None:
                raise ValueError("Invalid 2-omega waveplate.")
            calib = REGISTRY.get(_reg_key_calib(wp_idx))
            if not calib or not isinstance(calib, (tuple, list)) or len(calib) < 2:
                raise ValueError(
                    f"2-Omega waveplate ({wp_2omega_key}) not calibrated.\n"
                    f"Calibrate it in the Waveplate Calibration tab."
                )
            pm = REGISTRY.get(_reg_key_powermode(wp_idx))
            if not isinstance(pm, bool) or not pm:
                raise ValueError(
                    f"2-Omega waveplate ({wp_2omega_key}) power mode OFF.\n"
                    f"Enable Power Mode in the Stage Control window."
                )

        return {
            "phase_ctrl_key": phase_ctrl_key,
            "setpoints": setpoints,
            "serpentine": bool(self.serpentine_cb.isChecked()),
            "detector_params": detector_params,
            "max_phase_error": float(self.max_phase_error_sb.value()),
            "max_phase_std": float(self.max_phase_std_sb.value()),
            "stability_samples": int(self.stability_samples_sb.value()),
            "scan_name": name,
            "comment": self.comment.text(),
            "ratio_values": ratio_values,
            "total_intensity_W_cm2": total_intensity,
            "slm_screen": slm_screen,
            "alpha": alpha,
            "beta_b_fixed": beta_b,
            "omega_max_power_W": omega_max_power,
            "omega_waist_um": omega_waist,
            "omega_pulse_duration_fs": omega_tau,
            "omega_rep_rate_kHz": omega_rep,
            "omega_beam_split": omega_beam_split,
            "omega_beam_split_ratio": omega_beam_split_ratio,
            "wp_2omega_key": wp_2omega_key,
            "omega2_max_power_W": omega2_max_power,
            "omega2_waist_um": omega2_waist,
            "omega2_pulse_duration_fs": omega2_tau,
            "omega2_rep_rate_kHz": omega2_rep,
        }

    # -------------------------------------------------------------------------
    # Scan time estimate
    # -------------------------------------------------------------------------

    def _estimate_time(self):
        try:
            p = self._collect_params(validate=False)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        n_ratios = len(p["ratio_values"])
        n_phases = len(p["setpoints"])
        n_detectors = max(1, len(p["detector_params"]))

        # Lower bound only: with no timeout, a point takes at least X new
        # acquisitions (one get_current_phase() read per ~0.1 s loop) to fill
        # the window, and longer whenever the lock takes time to settle. The
        # first sample is immediate, so filling the window only needs
        # (X - 1) further 0.1 s sleeps.
        min_wait = max(0, p["stability_samples"] - 1) * 0.1

        detector_time = 0.0
        for _det_key, params in p["detector_params"].items():
            exposure = float(params[0]) if len(params) >= 1 else 0.0
            averages = int(params[1]) if len(params) >= 2 else 1
            detector_time += (exposure / 1e6) * averages

        time_per_phase = min_wait + detector_time
        total = n_ratios * n_phases * time_per_phase
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = int(total % 60)

        sweep_mode = "serpentine" if p["serpentine"] else "always forward"
        msg = (
            f"Ratio points: {n_ratios}\n"
            f"Phase points per ratio: {n_phases}\n"
            f"Phase sweep: {sweep_mode}\n"
            f"Detectors: {n_detectors}\n"
            f"Total acquisitions: {n_ratios * n_phases * n_detectors}\n\n"
            f"Min stability wait: {min_wait:.2f} s ({p['stability_samples']} acquisitions)\n"
            f"Detector acquisition: {detector_time:.2f} s\n"
            f"Min per point: {time_per_phase:.2f} s\n\n"
            f"Estimated MINIMUM total: {hours}h {minutes}min {seconds}s\n\n"

        )
        QMessageBox.information(self, "Scan Time Estimate", msg)
        self._log(f"Estimated minimum scan time: {hours}h {minutes}min {seconds}s")

    # -------------------------------------------------------------------------
    # Scan control
    # -------------------------------------------------------------------------

    def _start(self):
        try:
            p = self._collect_params(validate=True)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        # Block (with override) if the requested R range clips at this I_tot.
        self._validate_range()
        if not self._scan_ok:
            reply = QMessageBox.warning(
                self, "R range will clip",
                "The requested R range exceeds what's conservable at this I_tot.\n"
                "Points outside the range will clip and I_tot won't be conserved,\n"
                "and the worker will abort when it hits the first such point.\n\n"
                "Start anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        n_points = len(p["ratio_values"]) * len(p["setpoints"])
        if not confirm_large_scan(self, n_points, len(p["detector_params"])):
            return

        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(background=False, existing=None)
        self._log(
            "Scan started"
            + (" (serpentine phase sweep)..." if p["serpentine"] else "...")
        )

    def _open_monitor(self):
        if self._monitor_window is not None:
            self._monitor_window.show()
            self._monitor_window.raise_()
            self._monitor_window.activateWindow()

    def _launch(self, background, existing):
        p = self._cached_params
        if not p:
            return

        if not background:
            self._monitor_window = MonitorWindow(p["ratio_values"], p["setpoints"], parent=None)
            self._monitor_window.show()
            self.monitor_btn.setEnabled(True)

        self._thread = QThread(self)
        self._worker = TwoColorScanWorker(
            phase_ctrl_key=p["phase_ctrl_key"],
            setpoints=p["setpoints"],
            serpentine=p["serpentine"],
            detector_params=p["detector_params"],
            max_phase_error_rad=p["max_phase_error"],
            max_phase_std_rad=p["max_phase_std"],
            stability_samples=p["stability_samples"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            ratio_values=p["ratio_values"],
            total_intensity_W_cm2=p["total_intensity_W_cm2"],
            slm_screen=p["slm_screen"],
            alpha=p["alpha"],
            beta_b_fixed=p["beta_b_fixed"],
            omega_max_power_W=p["omega_max_power_W"],
            omega_waist_um=p["omega_waist_um"],
            omega_pulse_duration_fs=p["omega_pulse_duration_fs"],
            omega_rep_rate_kHz=p["omega_rep_rate_kHz"],
            omega_beam_split=p["omega_beam_split"],
            omega_beam_split_ratio=p["omega_beam_split_ratio"],
            wp_2omega_key=p["wp_2omega_key"],
            omega2_max_power_W=p["omega2_max_power_W"],
            omega2_waist_um=p["omega2_waist_um"],
            omega2_pulse_duration_fs=p["omega2_pulse_duration_fs"],
            omega2_rep_rate_kHz=p["omega2_rep_rate_kHz"],
            background=background,
            existing_scan_log=existing,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        if self._monitor_window is not None and not background:
            self._worker.monitor_update.connect(self._monitor_window.update_data)
        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        # Skip only makes sense during the foreground (phase-locked) scan.
        self.skip_btn.setEnabled(not background)

        n_ratios = len(p["ratio_values"])
        n_phases = len(p["setpoints"]) if not background else 1
        total = n_ratios * n_phases * len(p["detector_params"])
        self.prog.setMaximum(total)
        self.prog.setValue(0)

        self._thread.start()

    def _abort(self):
        if self._worker:
            self._worker.abort = True
            self.abort_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)

    def _skip_point(self):
        """Give up waiting for stability on the current point; take it now."""
        if self._worker:
            self._worker.skip_point = True
            self._log("Skip requested: taking current point without meeting criterion.")

    def _on_progress(self, i, n):
        self.prog.setMaximum(n)
        self.prog.setValue(i)

    def _finished(self, log_path):
        if log_path:
            self._last_scan_log_path = log_path
            self._log(f"Scan finished: {log_path}")
        else:
            self._log("Scan finished with errors.")
            self._last_scan_log_path = None

        self.abort_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.monitor_btn.setEnabled(False)

        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

        if not self._doing_background and self._last_scan_log_path is not None:
            reply = QMessageBox.question(
                self,
                "Run Background Scan?",
                "The scan finished.\n\nRun the BACKGROUND scan now?\n"
                "If yes, cut the gas and wait 3-5 min before continuing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._doing_background = True
                self._log("Launching background scan...")
                self._launch(background=True, existing=self._last_scan_log_path)
                return

        self._doing_background = False

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        scrollbar = self.log.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        self.log.append(f"[{ts}] {msg}")
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        if self._log_panel is not None:
            try:
                self._log_panel.log(msg, source="TwoColorScan")
            except Exception:
                pass