from __future__ import annotations
# TODO : fix alpha into alpha
import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit, QCheckBox, QRadioButton,
    QButtonGroup, QFileDialog
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import logging
logger = logging.getLogger("dlab.scans.two_color_scan_tab")


def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


def _save_png_with_meta(folder: Path, filename: str, frame_u16: np.ndarray, meta: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    img = Image.fromarray(frame_u16, mode="I;16")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path


def _detector_display_name(det_key, dev, meta):
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


def power_to_angle(power_fraction, _amp_unused, phase_deg):
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0


def angle_to_power(angle_deg, phase_deg):
    y = 0.5 * (1.0 + float(np.cos(2.0 * np.pi / 90.0 * (float(angle_deg) - float(phase_deg)))))
    return float(np.clip(y, 0.0, 1.0))


def _wp_index_from_stage_key(stage_key):
    try:
        if not stage_key.startswith("stage:"):
            return None
        n = int(stage_key.split(":")[1])
        if 1 <= n <= 10:
            return n
    except (ValueError, IndexError):
        pass
    return None


def _reg_key_powermode(wp_index):
    return f"waveplate:powermode:{wp_index}"


def _reg_key_calib(wp_index):
    return f"waveplate:calib:{wp_index}"


def _reg_key_maxvalue(wp_index):
    return f"waveplate:max_value:{wp_index}"


def calculate_max_intensity(max_power_W: float, waist_um: float, 
                            pulse_duration_fs: float, rep_rate_kHz: float) -> float:
    """
    Calculate maximum peak intensity for given beam parameters.
    
    Args:
        max_power_W: Maximum average power [W]
        waist_um: Beam waist at focus [µm]
        pulse_duration_fs: Pulse duration [fs]
        rep_rate_kHz: Repetition rate [kHz]
    
    Returns:
        I_max_peak: Maximum peak intensity [W/cm²]
    """
    waist_cm = waist_um * 1e-4
    pulse_duration_s = pulse_duration_fs * 1e-15
    rep_rate_Hz = rep_rate_kHz * 1e3
    
    # P_peak = P_avg / (f_rep × τ)
    P_peak = max_power_W / (rep_rate_Hz * pulse_duration_s)
    
    # I_peak = 2 × P_peak / (π × w0²) for Gaussian beam
    area_cm2 = np.pi * waist_cm**2
    I_peak = 2.0 * P_peak / area_cm2
    
    return I_peak


def intensity_to_power(I_peak_W_cm2: float, waist_um: float,
                       pulse_duration_fs: float, rep_rate_kHz: float) -> float:
    """
    Calculate required average power to achieve target peak intensity.
    
    Args:
        I_peak_W_cm2: Target peak intensity [W/cm²]
        waist_um: Beam waist at focus [µm]
        pulse_duration_fs: Pulse duration [fs]
        rep_rate_kHz: Repetition rate [kHz]
    
    Returns:
        P_avg: Required average power [W]
    """
    waist_cm = waist_um * 1e-4
    pulse_duration_s = pulse_duration_fs * 1e-15
    rep_rate_Hz = rep_rate_kHz * 1e3
    
    # I_peak = 2 × P_peak / (π × w0²)
    # P_peak = I_peak × π × w0² / 2
    area_cm2 = np.pi * waist_cm**2
    P_peak = I_peak_W_cm2 * area_cm2 / 2.0
    
    # P_avg = P_peak × f_rep × τ
    P_avg = P_peak * rep_rate_Hz * pulse_duration_s
    
    return P_avg


def load_slm_calibration(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SLM calibration file: alpha_dump_a vs normalized intensity fraction (0 to 1)
    Note: The file format has intensity decreasing as alpha increases
    
    Returns:
        (alpha_dump_a_values, intensity_fraction_values)
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    alpha_dump_a_val = float(parts[0])
                    intensity_frac = float(parts[1])
                    data.append((alpha_dump_a_val, intensity_frac))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("No valid data found in calibration file")
    
    alpha_dump_a_vals = np.array([d[0] for d in data])
    intensity_fracs = np.array([d[1] for d in data])
    
    if np.any(intensity_fracs < 0) or np.any(intensity_fracs > 1):
        raise ValueError("Intensity fraction values must be between 0 and 1")
    
    if np.any(alpha_dump_a_vals < 0) or np.any(alpha_dump_a_vals > 1):
        raise ValueError("alpha_dump_a values must be between 0 and 1")
    
    return alpha_dump_a_vals, intensity_fracs


class MonitorWindow(QWidget):
    """Real-time monitoring window for phase error and std during scan"""
    def __init__(self, ratio_values, setpoints, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Monitor - Phase Error & Std")
        self.resize(1200, 500)
        
        # Store scan dimensions
        self.ratio_values = np.array(ratio_values)
        self.setpoints = np.array(setpoints)
        self.n_ratios = len(ratio_values)
        self.n_phases = len(setpoints)
        
        # Initialize data arrays with NaN
        self.error_data = np.full((self.n_ratios, self.n_phases), np.nan)
        self.std_data = np.full((self.n_ratios, self.n_phases), np.nan)
        
        # Build UI
        layout = QHBoxLayout(self)
        
        # Create figure with 2 subplots
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        self.ax_error = self.figure.add_subplot(121)
        self.ax_std = self.figure.add_subplot(122)
        
        # Initial plot setup
        extent = [self.setpoints[0], self.setpoints[-1], 
                 self.ratio_values[0], self.ratio_values[-1]]
        
        self.im_error = self.ax_error.imshow(
            self.error_data, aspect='auto', origin='lower',
            extent=extent, cmap='RdYlGn_r', interpolation='nearest'
        )
        self.ax_error.set_xlabel('Phase setpoint [rad]')
        self.ax_error.set_ylabel('Ratio R')
        self.ax_error.set_title('Phase Error [rad]')
        self.figure.colorbar(self.im_error, ax=self.ax_error)
        
        self.im_std = self.ax_std.imshow(
            self.std_data, aspect='auto', origin='lower',
            extent=extent, cmap='RdYlGn_r', interpolation='nearest'
        )
        self.ax_std.set_xlabel('Phase setpoint [rad]')
        self.ax_std.set_ylabel('Ratio R')
        self.ax_std.set_title('Phase Std [rad]')
        self.figure.colorbar(self.im_std, ax=self.ax_std)
        
        self.figure.tight_layout()
        
        layout.addWidget(self.canvas)
    
    def update_data(self, ratio_idx: int, phase_idx: int, error: float, std: float):
        """Update data point and refresh display"""
        self.error_data[ratio_idx, phase_idx] = abs(error)
        self.std_data[ratio_idx, phase_idx] = std
        
        # Update images efficiently
        self.im_error.set_data(self.error_data)
        self.im_std.set_data(self.std_data)
        
        # Auto-scale color limits based on valid (non-NaN) data
        valid_errors = self.error_data[~np.isnan(self.error_data)]
        valid_stds = self.std_data[~np.isnan(self.std_data)]
        
        if len(valid_errors) > 0:
            self.im_error.set_clim(vmin=0, vmax=np.percentile(valid_errors, 95))
        if len(valid_stds) > 0:
            self.im_std.set_clim(vmin=0, vmax=np.percentile(valid_stds, 95))
        
        # Redraw canvas
        self.canvas.draw_idle()


class TwoColorScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    monitor_update = pyqtSignal(int, int, float, float)  # ratio_idx, phase_idx, error, std

    def __init__(
        self,
        phase_ctrl_key: str,
        setpoints: List[float],
        detector_params: Dict[str, tuple],
        max_phase_error_rad: float,
        max_phase_std_rad: float,
        stability_check_window_s: float,
        stability_timeout_s: float,
        phase_avg_s: float,
        scan_name: str,
        comment: str,
        # Ratio scan parameters
        enable_ratio_scan: bool = False,
        omega_control_mode: str = "waveplate",  # "waveplate" or "slm"
        wp_omega_key: str = None,
        wp_2omega_key: str = None,
        total_intensity_W_cm2: float = None,
        ratio_values: List[float] = None,
        # Laser parameters for omega
        omega_max_power_W: float = None,
        omega_waist_um: float = None,
        omega_pulse_duration_fs: float = None,
        omega_rep_rate_kHz: float = None,
        omega_beam_split: bool = False,
        omega_beam_split_ratio: float = 0.5,  # Fraction in B beam (A gets 1-ratio)
        # SLM parameters for omega
        slm_class_name: str = None,
        slm_field_name: str = None,
        slm_screen: int = 3,
        slm_calib_alpha: np.ndarray = None,
        slm_calib_intensity: np.ndarray = None,
        # Laser parameters for 2omega
        omega2_max_power_W: float = None,
        omega2_waist_um: float = None,
        omega2_pulse_duration_fs: float = None,
        omega2_rep_rate_kHz: float = None,
        # Background scan
        background: bool = False,
        existing_scan_log: str = None,
        # Background with reference
        background_w_ref: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.phase_ctrl_key = phase_ctrl_key
        self.setpoints = setpoints
        self.detector_params = detector_params
        self.max_phase_error_rad = float(max_phase_error_rad)
        self.max_phase_std_rad = float(max_phase_std_rad)
        self.stability_check_window_s = float(stability_check_window_s)
        self.stability_timeout_s = float(stability_timeout_s)
        self.phase_avg_s = float(phase_avg_s)
        self.scan_name = scan_name
        self.comment = comment
        self.abort = False
        
        # Ratio scan
        self.enable_ratio_scan = enable_ratio_scan
        self.omega_control_mode = omega_control_mode
        self.wp_omega_key = wp_omega_key
        self.wp_2omega_key = wp_2omega_key
        self.total_intensity_W_cm2 = total_intensity_W_cm2
        self.ratio_values = ratio_values or [0.5]
        
        # Laser parameters
        self.omega_max_power_W = omega_max_power_W
        self.omega_waist_um = omega_waist_um
        self.omega_pulse_duration_fs = omega_pulse_duration_fs
        self.omega_rep_rate_kHz = omega_rep_rate_kHz
        self.omega_beam_split = omega_beam_split
        self.omega_beam_split_ratio = omega_beam_split_ratio  # Fraction in B beam (A gets 1-ratio)
        
        # SLM parameters
        self.slm_class_name = slm_class_name
        self.slm_field_name = slm_field_name
        self.slm_screen = slm_screen
        self.slm_calib_alpha = slm_calib_alpha
        self.slm_calib_intensity = slm_calib_intensity
        
        self.omega2_max_power_W = omega2_max_power_W
        self.omega2_waist_um = omega2_waist_um
        self.omega2_pulse_duration_fs = omega2_pulse_duration_fs
        self.omega2_rep_rate_kHz = omega2_rep_rate_kHz
        
        # Background
        self.background = background
        self.existing_scan_log = existing_scan_log
        self.background_w_ref = background_w_ref
        
        self.data_root = _data_root()
        self.timestamp = datetime.datetime.now()

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def _set_waveplate_power(self, stage_key: str, power_W: float, max_power_W: float) -> None:
        """Set waveplate to achieve desired power output using power mode"""
        wp_index = _wp_index_from_stage_key(stage_key)
        if wp_index is None:
            raise ValueError(f"Invalid waveplate key: {stage_key}")
        
        # Get calibration phase
        amp_off = REGISTRY.get(_reg_key_calib(wp_index)) or (None, None)
        if amp_off[1] is None:
            raise ValueError(f"{stage_key}: No calibration phase found")
        phase_deg = float(amp_off[1])
        
        # Register max power value
        REGISTRY.register(_reg_key_maxvalue(wp_index), float(max_power_W))
        
        # Calculate fraction and angle
        power_fraction = power_W / float(max_power_W)
        power_fraction = np.clip(power_fraction, 0.0, 1.0)
        angle = power_to_angle(power_fraction, 1.0, phase_deg)
        
        # Move stage
        stage = REGISTRY.get(stage_key)
        if stage is None:
            raise ValueError(f"Stage not found: {stage_key}")
        
        stage.move_to(float(angle), blocking=True)
        self._emit(f"  {stage_key} → {power_W:.6f} W / {max_power_W:.6f} W ({100*power_fraction:.1f}%, angle: {angle:.3f}°)")

    def _set_slm_intensity(self, I_peak_W_cm2: float, I_max_W_cm2: float) -> float:
        """
        Set SLM alpha to achieve desired peak intensity
        
        Returns:
            Actual alpha value set
        """
        # Calculate required intensity fraction
        fraction = I_peak_W_cm2 / I_max_W_cm2
        fraction = np.clip(fraction, 0.0, 1.0)
        
        # Interpolate to find alpha: given intensity fraction, find alpha
        # Since intensity typically decreases with increasing alpha, we interpolate
        alpha_val = np.interp(fraction, self.slm_calib_intensity, self.slm_calib_alpha)
        
        # Set SLM
        active_classes = REGISTRY.get("slm:red:active_classes") or []
        if self.slm_class_name not in active_classes:
            raise RuntimeError(
                f"SLM class '{self.slm_class_name}' is not active.\n"
                f"Active classes: {active_classes}"
            )
        
        widgets = REGISTRY.get("slm:red:widgets") or []
        phase_widget = None
        for w in widgets:
            if getattr(w, "name_", lambda: "")() == self.slm_class_name:
                phase_widget = w
                break
        
        if phase_widget is None:
            raise RuntimeError(f"SLM widget for '{self.slm_class_name}' not found")
        
        if not hasattr(phase_widget, self.slm_field_name):
            raise RuntimeError(
                f"Field '{self.slm_field_name}' not found in '{self.slm_class_name}'"
            )
        
        widget = getattr(phase_widget, self.slm_field_name)
        widget.setText(str(alpha_val))
        
        # Compose and publish
        slm_window = REGISTRY.get("slm:red:window")
        if slm_window is None:
            raise RuntimeError("SLM window not found")
        
        levels = slm_window.compose_levels()
        
        slm_red = REGISTRY.get("slm:red:controller")
        if slm_red is None:
            raise RuntimeError("Red SLM controller not found")
        
        slm_red.publish(levels, screen_num=self.slm_screen)
        
        self._emit(f"  SLM {self.slm_class_name}:{self.slm_field_name} = {alpha_val:.6f} (intensity fraction: {fraction:.4f})")
        
        return float(alpha_val)

    def _wait_for_stability(self, phase_ctrl, setpoint: float) -> Tuple[float, float, float, bool]:
        """
        Wait for phase lock to stabilize before acquisition.
        
        Returns:
            (avg_phase, std_phase, phase_error, timed_out)
        """
        start_time = time.time()
        check_count = 0
        
        while True:
            if self.abort:
                return 0.0, 0.0, 0.0, True
            
            # Check stability
            avg_phase, std_phase = phase_ctrl.get_phase_average(self.stability_check_window_s)
            phase_error = abs(avg_phase - setpoint)
            
            check_count += 1
            elapsed = time.time() - start_time
            
            # Check if stable
            error_ok = phase_error < self.max_phase_error_rad
            std_ok = std_phase < self.max_phase_std_rad
            
            if error_ok and std_ok:
                self._emit(f"  Phase stable after {elapsed:.2f}s (error={phase_error:.4f} rad, std={std_phase:.4f} rad)")
                return avg_phase, std_phase, phase_error, False
            
            # Check timeout
            if elapsed > self.stability_timeout_s:
                self._emit(f"  WARNING: Stability timeout after {elapsed:.2f}s (error={phase_error:.4f} rad, std={std_phase:.4f} rad)")
                return avg_phase, std_phase, phase_error, True
            
            # Log progress every few checks
            if check_count % 3 == 0:
                self._emit(f"  Waiting for stability... ({elapsed:.1f}s) error={phase_error:.4f}, std={std_phase:.4f}")
            
            # Small sleep between checks
            time.sleep(0.1)

    def run(self) -> None:
        # Phase controller only needed if not background
        phase_ctrl = None
        if not self.background:
            phase_ctrl = REGISTRY.get(self.phase_ctrl_key)
            if phase_ctrl is None:
                self._emit(f"Phase controller '{self.phase_ctrl_key}' not found.")
                self.finished.emit("")
                return

            if not hasattr(phase_ctrl, 'set_target'):
                self._emit(f"Phase controller doesn't have required API methods.")
                self.finished.emit("")
                return
            
            if not phase_ctrl.is_locked():
                self._emit(f"Phase controller is not locked. Please enable lock first.")
                self.finished.emit("")
                return

        # Validate ratio scan setup if enabled
        if self.enable_ratio_scan:
            if self.omega_control_mode == "waveplate":
                if not self.wp_omega_key:
                    self._emit("Omega waveplate not specified.")
                    self.finished.emit("")
                    return
            elif self.omega_control_mode == "slm":
                if not self.slm_class_name or not self.slm_field_name:
                    self._emit("SLM class/field not specified.")
                    self.finished.emit("")
                    return
                if self.slm_calib_alpha is None or self.slm_calib_intensity is None:
                    self._emit("SLM calibration not loaded.")
                    self.finished.emit("")
                    return
            
            if not self.wp_2omega_key:
                self._emit("2-Omega waveplate not specified.")
                self.finished.emit("")
                return
            
            if self.total_intensity_W_cm2 is None or self.total_intensity_W_cm2 <= 0:
                self._emit("Total intensity invalid.")
                self.finished.emit("")
                return

        detectors = {}
        for det_key, params in self.detector_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                self._emit(f"Detector '{det_key}' not found.")
                self.finished.emit("")
                return

            is_camera = hasattr(dev, "grab_frame_for_scan")
            is_spectro = hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan")
            is_pow = hasattr(dev, "fetch_power") or hasattr(dev, "read_power")

            if not (is_camera or is_spectro or is_pow):
                self._emit(f"Detector '{det_key}' doesn't expose a scan API.")
                self.finished.emit("")
                return

            try:
                if is_camera:
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

        # Use existing log for background, or create new one
        if self.existing_scan_log:
            scan_log = Path(self.existing_scan_log)
        else:
            date_str = f"{self.timestamp:%Y-%m-%d}"
            idx = 1
            while True:
                candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
                if not candidate.exists():
                    break
                idx += 1
            scan_log = candidate

            with open(scan_log, "w", encoding="utf-8") as f:
                if self.enable_ratio_scan:
                    if self.omega_control_mode == "slm":
                        header = ["Ratio_R", "alpha", "Power_2omega_W",
                                 "I_omega_peak_W_cm2", "I_2omega_peak_W_cm2",
                                 "Setpoint_rad", "MeasuredPhase_rad", "PhaseStd_rad", "PhaseError_rad",
                                 "DetectorKey", "DataFile", "Exposure_or_IntTime", "Averages"]
                    else:
                        header = ["Ratio_R", "Power_omega_W", "Power_2omega_W",
                                 "I_omega_peak_W_cm2", "I_2omega_peak_W_cm2",
                                 "Setpoint_rad", "MeasuredPhase_rad", "PhaseStd_rad", "PhaseError_rad",
                                 "DetectorKey", "DataFile", "Exposure_or_IntTime", "Averages"]
                else:
                    header = ["Setpoint_rad", "MeasuredPhase_rad", "PhaseStd_rad", "PhaseError_rad",
                             "DetectorKey", "DataFile", "Exposure_or_IntTime", "Averages"]
                f.write("\t".join(header) + "\n")
                f.write(f"# {self.comment}\n")
                f.write(f"# Phase stability criteria:\n")
                f.write(f"#   Max error: {self.max_phase_error_rad} rad\n")
                f.write(f"#   Max std: {self.max_phase_std_rad} rad\n")
                f.write(f"#   Check window: {self.stability_check_window_s} s\n")
                f.write(f"#   Timeout: {self.stability_timeout_s} s\n")
                f.write(f"# Phase averaging: {self.phase_avg_s} s\n")
                if self.enable_ratio_scan:
                    f.write(f"# Ratio scan enabled: R = I_2omega / (I_omega + I_2omega)\n")
                    f.write(f"# Total peak intensity: {self.total_intensity_W_cm2:.6e} W/cm²\n")
                    
                    if self.omega_control_mode == "slm":
                        f.write(f"# Omega control: SLM {self.slm_class_name}:{self.slm_field_name}\n")
                        f.write(f"#   Screen: {self.slm_screen}\n")
                        f.write(f"#   Max power: {self.omega_max_power_W} W\n")
                        f.write(f"#   Waist: {self.omega_waist_um} µm\n")
                        f.write(f"#   Pulse duration: {self.omega_pulse_duration_fs} fs\n")
                        f.write(f"#   Rep rate: {self.omega_rep_rate_kHz} kHz\n")
                        I_max_omega = calculate_max_intensity(
                            self.omega_max_power_W, self.omega_waist_um,
                            self.omega_pulse_duration_fs, self.omega_rep_rate_kHz
                        )
                        if self.omega_beam_split:
                            # beam_split_ratio is fraction in B, so A gets (1 - ratio)
                            I_max_omega *= (1.0 - self.omega_beam_split_ratio)
                            f.write(f"#   Beam split: YES (B beam fraction: {self.omega_beam_split_ratio:.3f}, A gets {1.0-self.omega_beam_split_ratio:.3f})\n")
                        # Apply max calibration intensity
                        if self.slm_calib_intensity is not None:
                            max_calib_intensity = float(self.slm_calib_intensity.max())
                            I_max_omega *= max_calib_intensity
                            f.write(f"#   Max calibration intensity: {max_calib_intensity:.4f}\n")
                        f.write(f"#   Max intensity: {I_max_omega:.6e} W/cm²\n")
                    else:
                        f.write(f"# Waveplate omega: {self.wp_omega_key}\n")
                        f.write(f"#   Max power: {self.omega_max_power_W} W\n")
                        f.write(f"#   Waist: {self.omega_waist_um} µm\n")
                        f.write(f"#   Pulse duration: {self.omega_pulse_duration_fs} fs\n")
                        f.write(f"#   Rep rate: {self.omega_rep_rate_kHz} kHz\n")
                        I_max_omega = calculate_max_intensity(
                            self.omega_max_power_W, self.omega_waist_um,
                            self.omega_pulse_duration_fs, self.omega_rep_rate_kHz
                        )
                        if self.omega_beam_split:
                            # beam_split_ratio is fraction in B, so A gets (1 - ratio)
                            I_max_omega *= (1.0 - self.omega_beam_split_ratio)
                            f.write(f"#   Beam split: YES (B beam fraction: {self.omega_beam_split_ratio:.3f}, A gets {1.0-self.omega_beam_split_ratio:.3f})\n")
                        f.write(f"#   Max intensity: {I_max_omega:.6e} W/cm²\n")
                    
                    f.write(f"# Waveplate 2omega: {self.wp_2omega_key}\n")
                    f.write(f"#   Max power: {self.omega2_max_power_W} W\n")
                    f.write(f"#   Waist: {self.omega2_waist_um} µm\n")
                    f.write(f"#   Pulse duration: {self.omega2_pulse_duration_fs} fs\n")
                    f.write(f"#   Rep rate: {self.omega2_rep_rate_kHz} kHz\n")
                    I_max_2omega = calculate_max_intensity(
                        self.omega2_max_power_W, self.omega2_waist_um,
                        self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz
                    )
                    f.write(f"#   Max intensity: {I_max_2omega:.6e} W/cm²\n")

        # Calculate total points
        n_ratios = len(self.ratio_values) if self.enable_ratio_scan else 1
        n_phases = len(self.setpoints) if not self.background else 1
        total_points = n_ratios * n_phases * len(detectors)
        done = 0

        try:
            # Outer loop: ratio values (if enabled)
            ratio_loop = self.ratio_values if self.enable_ratio_scan else [None]
            
            for ratio_idx, ratio_R in enumerate(ratio_loop):
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return
                
                # Set powers/intensities for this ratio
                power_omega = None
                alpha_val = None
                power_2omega = None
                I_omega_peak = None
                I_2omega_peak = None
                
                if self.enable_ratio_scan:
                    # Calculate intensities
                    I_2omega_peak = ratio_R * self.total_intensity_W_cm2
                    I_omega_peak = (1.0 - ratio_R) * self.total_intensity_W_cm2
                    
                    # Calculate max intensities
                    I_max_omega = calculate_max_intensity(
                        self.omega_max_power_W, self.omega_waist_um,
                        self.omega_pulse_duration_fs, self.omega_rep_rate_kHz
                    )
                    if self.omega_beam_split:
                        # beam_split_ratio is fraction in B, so A gets (1 - ratio)
                        I_max_omega *= (1.0 - self.omega_beam_split_ratio)
                    
                    # Apply max calibration intensity for SLM
                    if self.omega_control_mode == "slm" and self.slm_calib_intensity is not None:
                        I_max_omega *= float(self.slm_calib_intensity.max())
                    
                    I_max_2omega = calculate_max_intensity(
                        self.omega2_max_power_W, self.omega2_waist_um,
                        self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz
                    )
                    
                    # Check limits
                    if I_omega_peak > I_max_omega:
                        self._emit(f"Warning: Required omega intensity ({I_omega_peak:.3e} W/cm²) exceeds max ({I_max_omega:.3e} W/cm²)")
                        I_omega_peak = I_max_omega
                    
                    if I_2omega_peak > I_max_2omega:
                        self._emit(f"Warning: Required 2-omega intensity ({I_2omega_peak:.3e} W/cm²) exceeds max ({I_max_2omega:.3e} W/cm²)")
                        I_2omega_peak = I_max_2omega
                    
                    self._emit(f"\n=== Setting ratio R = {ratio_R:.4f} ===")
                    self._emit(f"  Total peak intensity: {self.total_intensity_W_cm2:.6e} W/cm²")
                    self._emit(f"  I_omega:  {I_omega_peak:.6e} W/cm² ({100*(1-ratio_R):.1f}%)")
                    self._emit(f"  I_2omega: {I_2omega_peak:.6e} W/cm² ({100*ratio_R:.1f}%)")
                    
                    try:
                        if self.omega_control_mode == "slm":
                            # Set SLM
                            alpha_val = self._set_slm_intensity(I_omega_peak, I_max_omega)
                        else:
                            # Set waveplate
                            power_omega = intensity_to_power(
                                I_omega_peak, self.omega_waist_um,
                                self.omega_pulse_duration_fs, self.omega_rep_rate_kHz
                            )
                            self._emit(f"  P_omega: {power_omega:.6f} W")
                            self._set_waveplate_power(self.wp_omega_key, power_omega, self.omega_max_power_W)
                        
                        # Always set 2-omega waveplate
                        power_2omega = intensity_to_power(
                            I_2omega_peak, self.omega2_waist_um,
                            self.omega2_pulse_duration_fs, self.omega2_rep_rate_kHz
                        )
                        self._emit(f"  P_2omega: {power_2omega:.6f} W")
                        self._set_waveplate_power(self.wp_2omega_key, power_2omega, self.omega2_max_power_W)
                    except Exception as e:
                        self._emit(f"Failed to set intensities: {e}")
                        self.finished.emit("")
                        return
                
                # Inner loop: phase setpoints (or single point for background)
                phase_loop = [None] if self.background else self.setpoints
                
                for phase_idx, sp in enumerate(phase_loop):
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    avg_phase = std_phase = phase_error = None
                    timed_out = False
                    
                    if not self.background:
                        phase_ctrl.set_target(float(sp))
                        if self.enable_ratio_scan:
                            self._emit(f"R={ratio_R:.3f}, Phase setpoint: {sp:.4f} rad")
                        else:
                            self._emit(f"Phase setpoint: {sp:.4f} rad")

                        # Wait for stability
                        avg_phase, std_phase, phase_error, timed_out = self._wait_for_stability(phase_ctrl, sp)
                        
                        if self.abort:
                            self._emit("Scan aborted.")
                            self.finished.emit("")
                            return
                        
                        # Do final averaging for acquisition
                        avg_phase, std_phase = phase_ctrl.get_phase_average(self.phase_avg_s)
                        phase_error = avg_phase - sp
                        
                        # Emit monitor update signal
                        if self.enable_ratio_scan:
                            self.monitor_update.emit(ratio_idx, phase_idx, phase_error, std_phase)
                    else:
                        scan_type = "background with reference (alpha=1)" if self.background_w_ref else "background"
                        if self.enable_ratio_scan:
                            self._emit(f"Capturing {scan_type} @ R={ratio_R:.3f}")
                        else:
                            self._emit(f"Capturing {scan_type}")

                    for det_key, dev in detectors.items():
                        if self.abort:
                            self._emit("Scan aborted.")
                            self.finished.emit("")
                            return

                        params = self.detector_params.get(det_key, (0, 1))

                        try:
                            if hasattr(dev, "grab_frame_for_scan"):
                                exposure_or_int = int(params[0]) if len(params) >= 1 else 0
                                averages = int(params[1]) if len(params) >= 2 else 1
                                
                                try:
                                    frame_u16, meta = dev.grab_frame_for_scan(
                                        averages=int(averages),
                                        background=self.background,
                                        dead_pixel_cleanup=True,
                                        exposure_us=int(exposure_or_int),
                                    )
                                except TypeError:
                                    frame_u16, meta = dev.grab_frame_for_scan(
                                        averages=int(averages),
                                        background=self.background,
                                        dead_pixel_cleanup=True,
                                    )
                                
                                exp_meta = int((meta or {}).get("Exposure_us", exposure_or_int))
                                
                                # Save with grid_scan style naming
                                det_name = _detector_display_name(det_key, dev, meta)
                                det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / det_name
                                ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                
                                if self.background and self.background_w_ref:
                                    tag = "Background_w_ref"
                                elif self.background:
                                    tag = "Background"
                                else:
                                    tag = "Image"
                                fn = f"{det_name}_{tag}_{ts_ms}.png"
                                
                                meta_dict = {
                                    "Exposure_us": exp_meta,
                                    "Comment": self.comment
                                }
                                
                                if self.enable_ratio_scan:
                                    meta_dict.update({
                                        "Ratio_R": ratio_R,
                                        "I_omega_peak_W_cm2": I_omega_peak,
                                        "I_2omega_peak_W_cm2": I_2omega_peak,
                                    })
                                    if self.omega_control_mode == "slm":
                                        meta_dict["alpha"] = alpha_val
                                    else:
                                        meta_dict["Power_omega_W"] = power_omega
                                    meta_dict["Power_2omega_W"] = power_2omega
                                
                                if not self.background:
                                    meta_dict.update({
                                        "Setpoint_rad": sp,
                                        "MeasuredPhase_rad": avg_phase,
                                        "PhaseStd_rad": std_phase,
                                    })
                                
                                _save_png_with_meta(det_day, fn, frame_u16, meta_dict)
                                data_fn = fn
                                saved_label = f"exp {exp_meta} µs"

                            elif hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"):
                                exposure_or_int = float(params[0]) if len(params) >= 1 else 0.0
                                averages = int(params[1]) if len(params) >= 2 else 1
                                
                                if hasattr(dev, "get_wavelengths"):
                                    wl = np.asarray(dev.get_wavelengths(), dtype=float)
                                else:
                                    wl = np.asarray(getattr(dev, "wavelength", None), dtype=float)

                                if wl is None or wl.size == 0:
                                    self._emit(f"{det_key}: wavelength array empty.")
                                    continue

                                if hasattr(dev, "grab_spectrum_for_scan"):
                                    counts, meta = dev.grab_spectrum_for_scan(
                                        int_ms=float(exposure_or_int),
                                        averages=int(averages)
                                    )
                                    counts = np.asarray(counts, dtype=float)
                                    int_ms = float((meta or {}).get("Integration_ms", float(exposure_or_int)))
                                else:
                                    _buf = []
                                    for _ in range(int(averages)):
                                        _ts, _data = dev.measure_spectrum(float(exposure_or_int), 1)
                                        _buf.append(np.asarray(_data, dtype=float))
                                        time.sleep(0.01)
                                    counts = np.mean(np.stack(_buf, axis=0), axis=0)
                                    int_ms = float(exposure_or_int)

                                # Save with grid_scan style naming
                                det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "Avaspec"
                                safe_name = _detector_display_name(det_key, dev, None).replace(" ", "")
                                ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                
                                if self.background and self.background_w_ref:
                                    tag = "Background_w_ref"
                                elif self.background:
                                    tag = "Background"
                                else:
                                    tag = "Spectrum"
                                fn = f"{safe_name}_{tag}_{ts_ms}.txt"
                                
                                file_header = {
                                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "IntegrationTime_ms": int_ms,
                                    "Averages": averages,
                                    "Comment": self.comment,
                                }
                                
                                if self.enable_ratio_scan:
                                    file_header.update({
                                        "Ratio_R": ratio_R,
                                        "I_omega_peak_W_cm2": I_omega_peak,
                                        "I_2omega_peak_W_cm2": I_2omega_peak,
                                    })
                                    if self.omega_control_mode == "slm":
                                        file_header["alpha"] = alpha_val
                                    else:
                                        file_header["Power_omega_W"] = power_omega
                                    file_header["Power_2omega_W"] = power_2omega
                                
                                if not self.background:
                                    file_header.update({
                                        "Setpoint_rad": sp,
                                        "MeasuredPhase_rad": avg_phase,
                                        "PhaseStd_rad": std_phase,
                                    })
                                
                                det_day.mkdir(parents=True, exist_ok=True)
                                path = det_day / fn
                                lines = [f"# {k}: {v}" for k, v in file_header.items()]
                                lines.append("Wavelength_nm;Counts")
                                with open(path, "w", encoding="utf-8") as f:
                                    f.write("\n".join(lines) + "\n")
                                    for xv, yv in zip(wl, counts):
                                        f.write(f"{float(xv):.6f};{float(yv):.6f}\n")
                                
                                data_fn = fn
                                saved_label = f"int {int_ms:.0f} ms"

                            else:
                                period_ms = float(params[0]) if len(params) >= 1 else 100.0
                                averages = int(params[1]) if len(params) >= 2 else 1

                                vals = []
                                n_avg = max(1, int(averages))
                                for i in range(n_avg):
                                    if hasattr(dev, "read_power"):
                                        v = float(dev.read_power())
                                    else:
                                        v = float(dev.fetch_power())
                                    vals.append(v)
                                    if i + 1 < n_avg:
                                        time.sleep(period_ms / 1000.0)

                                power = float(np.mean(vals)) if vals else float("nan")
                                data_fn = f"{power:.9f}"
                                saved_label = f"P={power:.3e} W"

                            if self.enable_ratio_scan:
                                if self.omega_control_mode == "slm":
                                    row = [
                                        f"{float(ratio_R):.9f}",
                                        f"{float(alpha_val):.9f}",
                                        f"{float(power_2omega):.9f}",
                                        f"{float(I_omega_peak):.9e}",
                                        f"{float(I_2omega_peak):.9e}",
                                    ]
                                else:
                                    row = [
                                        f"{float(ratio_R):.9f}",
                                        f"{float(power_omega):.9f}",
                                        f"{float(power_2omega):.9f}",
                                        f"{float(I_omega_peak):.9e}",
                                        f"{float(I_2omega_peak):.9e}",
                                    ]
                            else:
                                row = []
                            
                            if not self.background:
                                row += [
                                    f"{float(sp):.9f}",
                                    f"{float(avg_phase):.9f}",
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
                                scan_type = "BACKGROUND_W_REF" if self.background_w_ref else "BACKGROUND"
                                if self.enable_ratio_scan:
                                    self._emit(f"Saved {data_fn} @ R={ratio_R:.3f} {scan_type} ({saved_label})")
                                else:
                                    self._emit(f"Saved {data_fn} {scan_type} ({saved_label})")
                            else:
                                if self.enable_ratio_scan:
                                    self._emit(
                                        f"Saved {data_fn} @ R={ratio_R:.3f}, SP={sp:.4f} rad, "
                                        f"Phase={avg_phase:.4f}±{std_phase:.4f} rad ({saved_label})"
                                    )
                                else:
                                    self._emit(
                                        f"Saved {data_fn} @ SP={sp:.4f} rad, "
                                        f"Phase={avg_phase:.4f}±{std_phase:.4f} rad ({saved_label})"
                                    )

                        except Exception as e:
                            if self.background:
                                self._emit(f"Capture failed (background) on {det_key}: {e}")
                            else:
                                if self.enable_ratio_scan:
                                    self._emit(f"Capture failed @ R={ratio_R:.3f}, SP={sp:.4f} rad on {det_key}: {e}")
                                else:
                                    self._emit(f"Capture failed @ SP={sp:.4f} rad on {det_key}: {e}")

                        done += 1
                        self.progress.emit(done, total_points)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            import traceback
            self._emit(traceback.format_exc())
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


class TwoColorScanTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._monitor_window = None
        self._doing_background = False
        self._cached_params = None
        self._last_scan_log_path = None
        self._slm_calib_alpha = None
        self._slm_calib_intensity = None
        self._build_ui()
        self._refresh_devices()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Phase controller
        ctrl_box = QGroupBox("Phase Lock Controller")
        ctrl_l = QHBoxLayout(ctrl_box)
        self.phase_ctrl_picker = QComboBox()
        ctrl_l.addWidget(QLabel("Controller:"))
        ctrl_l.addWidget(self.phase_ctrl_picker, 1)
        main.addWidget(ctrl_box)

        # Ratio scan settings
        ratio_box = QGroupBox("Ratio Scan (Optional)")
        ratio_l = QVBoxLayout(ratio_box)
        
        enable_row = QHBoxLayout()
        self.enable_ratio_cb = QCheckBox("Enable ratio scan: R = I_2ω / (I_ω + I_2ω)")
        self.enable_ratio_cb.toggled.connect(self._on_ratio_toggle)
        enable_row.addWidget(self.enable_ratio_cb)
        enable_row.addStretch()
        ratio_l.addLayout(enable_row)
        
        # Omega control mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Omega control:"))
        self.omega_mode_group = QButtonGroup()
        self.omega_wp_radio = QRadioButton("Waveplate")
        self.omega_slm_radio = QRadioButton("SLM")
        self.omega_wp_radio.setChecked(True)
        self.omega_mode_group.addButton(self.omega_wp_radio)
        self.omega_mode_group.addButton(self.omega_slm_radio)
        self.omega_wp_radio.toggled.connect(self._on_omega_mode_toggle)
        mode_row.addWidget(self.omega_wp_radio)
        mode_row.addWidget(self.omega_slm_radio)
        mode_row.addStretch()
        ratio_l.addLayout(mode_row)
        
        # Waveplate selection (for waveplate mode)
        self.wp_group = QGroupBox("Waveplate mode")
        wp_l = QVBoxLayout(self.wp_group)
        wp_row = QHBoxLayout()
        self.wp_omega_picker = QComboBox()
        self.wp_2omega_picker = QComboBox()
        wp_row.addWidget(QLabel("Waveplate ω:"))
        wp_row.addWidget(self.wp_omega_picker, 1)
        wp_row.addWidget(QLabel("Waveplate 2ω:"))
        wp_row.addWidget(self.wp_2omega_picker, 1)
        wp_l.addLayout(wp_row)
        ratio_l.addWidget(self.wp_group)
        
        # SLM selection (for SLM mode)
        self.slm_group = QGroupBox("SLM mode")
        slm_l = QVBoxLayout(self.slm_group)
        
        slm_row1 = QHBoxLayout()
        self.slm_class_le = QLineEdit("TwoFociStochastic")
        self.slm_field_le = QLineEdit("le_alpha_dump_A")
        self.slm_screen_le = QLineEdit("3")
        slm_row1.addWidget(QLabel("Class:"))
        slm_row1.addWidget(self.slm_class_le)
        slm_row1.addWidget(QLabel("Field:"))
        slm_row1.addWidget(self.slm_field_le)
        slm_row1.addWidget(QLabel("Screen:"))
        slm_row1.addWidget(self.slm_screen_le)
        slm_l.addLayout(slm_row1)
        
        slm_row2 = QHBoxLayout()
        self.slm_calib_path_le = QLineEdit("")
        self.slm_calib_browse_btn = QPushButton("Browse...")
        self.slm_calib_browse_btn.clicked.connect(self._browse_slm_calib)
        slm_row2.addWidget(QLabel("Calibration file:"))
        slm_row2.addWidget(self.slm_calib_path_le, 1)
        slm_row2.addWidget(self.slm_calib_browse_btn)
        slm_l.addLayout(slm_row2)
        
        slm_row3 = QHBoxLayout()
        self.slm_calib_load_btn = QPushButton("Load Calibration")
        self.slm_calib_load_btn.clicked.connect(self._load_slm_calib)
        self.slm_calib_status = QLabel("Status: Not loaded")
        slm_row3.addWidget(self.slm_calib_load_btn)
        slm_row3.addWidget(self.slm_calib_status)
        slm_row3.addStretch()
        slm_l.addLayout(slm_row3)
        
        # Background with reference option
        slm_row4 = QHBoxLayout()
        self.slm_background_ref_cb = QCheckBox("Take reference with beam dumped (alpha=1) before background")
        slm_row4.addWidget(self.slm_background_ref_cb)
        slm_row4.addStretch()
        slm_l.addLayout(slm_row4)
        
        # 2omega waveplate (always needed)
        slm_wp2_row = QHBoxLayout()
        slm_wp2_row.addWidget(QLabel("Waveplate 2ω:"))
        slm_wp2_row.addWidget(self.wp_2omega_picker, 1)
        slm_wp2_row.addStretch()
        slm_l.addLayout(slm_wp2_row)
        
        ratio_l.addWidget(self.slm_group)
        
        # Total intensity
        intensity_row = QHBoxLayout()
        self.total_intensity_le = QLineEdit("1e14")
        self.total_intensity_le.textChanged.connect(self._update_max_intensity)
        self.intensity_max_label = QLabel("I_tot_max: -- W/cm²")
        self.intensity_max_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        intensity_row.addWidget(QLabel("Total peak intensity I_tot (W/cm²):"))
        intensity_row.addWidget(self.total_intensity_le)
        intensity_row.addWidget(self.intensity_max_label)
        intensity_row.addStretch()
        ratio_l.addLayout(intensity_row)
        
        # Omega parameters
        omega_box = QGroupBox("Omega (ω) beam parameters")
        omega_l = QVBoxLayout(omega_box)
        
        omega_row1 = QHBoxLayout()
        self.omega_max_power_le = QLineEdit("")
        self.omega_max_power_le.textChanged.connect(self._update_max_intensity)
        self.omega_waist_le = QLineEdit("22")
        self.omega_waist_le.textChanged.connect(self._update_max_intensity)
        omega_row1.addWidget(QLabel("Max power (W):"))
        omega_row1.addWidget(self.omega_max_power_le)
        omega_row1.addWidget(QLabel("Waist at focus (µm):"))
        omega_row1.addWidget(self.omega_waist_le)
        omega_l.addLayout(omega_row1)
        
        omega_row2 = QHBoxLayout()
        self.omega_pulse_duration_le = QLineEdit("170")
        self.omega_pulse_duration_le.textChanged.connect(self._update_max_intensity)
        self.omega_rep_rate_le = QLineEdit("10")
        self.omega_rep_rate_le.textChanged.connect(self._update_max_intensity)
        omega_row2.addWidget(QLabel("Pulse duration (fs):"))
        omega_row2.addWidget(self.omega_pulse_duration_le)
        omega_row2.addWidget(QLabel("Rep rate (kHz):"))
        omega_row2.addWidget(self.omega_rep_rate_le)
        omega_l.addLayout(omega_row2)
        
        omega_row3 = QHBoxLayout()
        self.omega_beam_split_cb = QCheckBox("Beam split (A/B configuration)")
        self.omega_beam_split_cb.toggled.connect(self._update_max_intensity)
        self.omega_beam_split_ratio_le = QLineEdit("0.5")
        self.omega_beam_split_ratio_le.setEnabled(False)
        self.omega_beam_split_ratio_le.textChanged.connect(self._update_max_intensity)
        self.omega_beam_split_cb.toggled.connect(
            lambda checked: self.omega_beam_split_ratio_le.setEnabled(checked)
        )
        omega_row3.addWidget(self.omega_beam_split_cb)
        omega_row3.addWidget(QLabel("Fraction in B beam:"))
        omega_row3.addWidget(self.omega_beam_split_ratio_le)
        omega_row3.addWidget(QLabel("(0.5 = 50/50, 0.8 = 80% to B / 20% to A)"))
        omega_row3.addStretch()
        omega_l.addLayout(omega_row3)
        
        ratio_l.addWidget(omega_box)
        
        # 2-Omega parameters
        omega2_box = QGroupBox("2-Omega (2ω) beam parameters")
        omega2_l = QVBoxLayout(omega2_box)
        
        omega2_row1 = QHBoxLayout()
        self.omega2_max_power_le = QLineEdit("")
        self.omega2_max_power_le.textChanged.connect(self._update_max_intensity)
        self.omega2_waist_le = QLineEdit("20")
        self.omega2_waist_le.textChanged.connect(self._update_max_intensity)
        omega2_row1.addWidget(QLabel("Max power (W):"))
        omega2_row1.addWidget(self.omega2_max_power_le)
        omega2_row1.addWidget(QLabel("Waist at focus (µm):"))
        omega2_row1.addWidget(self.omega2_waist_le)
        omega2_l.addLayout(omega2_row1)
        
        omega2_row2 = QHBoxLayout()
        self.omega2_pulse_duration_le = QLineEdit("140")
        self.omega2_pulse_duration_le.textChanged.connect(self._update_max_intensity)
        self.omega2_rep_rate_le = QLineEdit("10")
        self.omega2_rep_rate_le.textChanged.connect(self._update_max_intensity)
        omega2_row2.addWidget(QLabel("Pulse duration (fs):"))
        omega2_row2.addWidget(self.omega2_pulse_duration_le)
        omega2_row2.addWidget(QLabel("Rep rate (kHz):"))
        omega2_row2.addWidget(self.omega2_rep_rate_le)
        omega2_l.addLayout(omega2_row2)
        
        ratio_l.addWidget(omega2_box)
        
        # Ratio range
        ratio_params = QHBoxLayout()
        self.ratio_start = QLineEdit("0.0")
        self.ratio_end = QLineEdit("1.0")
        self.ratio_step = QLineEdit("0.1")
        ratio_params.addWidget(QLabel("Ratio start:"))
        ratio_params.addWidget(self.ratio_start)
        ratio_params.addWidget(QLabel("end:"))
        ratio_params.addWidget(self.ratio_end)
        ratio_params.addWidget(QLabel("step:"))
        ratio_params.addWidget(self.ratio_step)
        ratio_l.addLayout(ratio_params)
        
        main.addWidget(ratio_box)

        # Phase setpoints
        sp_box = QGroupBox("Phase Setpoints")
        sp_l = QVBoxLayout(sp_box)
        
        sp_params = QHBoxLayout()
        self.sp_start = QLineEdit("-3.14159")
        self.sp_end = QLineEdit("3.14159")
        self.sp_step = QLineEdit("0.5")
        sp_params.addWidget(QLabel("Start [rad]:"))
        sp_params.addWidget(self.sp_start)
        sp_params.addWidget(QLabel("End [rad]:"))
        sp_params.addWidget(self.sp_end)
        sp_params.addWidget(QLabel("Step [rad]:"))
        sp_params.addWidget(self.sp_step)
        sp_l.addLayout(sp_params)
        main.addWidget(sp_box)

        # Detectors
        det_box = QGroupBox("Detectors")
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
        self.det_tbl.setHorizontalHeaderLabels(["DetectorKey", "Exposure_us/Int_ms", "Averages"])
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
        
        # Stability parameters row 1
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
        
        # Stability parameters row 2
        stab_row2 = QHBoxLayout()
        self.stability_check_window_sb = QDoubleSpinBox()
        self.stability_check_window_sb.setDecimals(2)
        self.stability_check_window_sb.setRange(0.1, 5.0)
        self.stability_check_window_sb.setValue(0.3)
        self.stability_check_window_sb.setSingleStep(0.1)
        
        self.stability_timeout_sb = QDoubleSpinBox()
        self.stability_timeout_sb.setDecimals(1)
        self.stability_timeout_sb.setRange(1.0, 300.0)
        self.stability_timeout_sb.setValue(15.0)
        self.stability_timeout_sb.setSingleStep(1.0)
        
        stab_row2.addWidget(QLabel("Check window (s):"))
        stab_row2.addWidget(self.stability_check_window_sb)
        stab_row2.addWidget(QLabel("Timeout (s):"))
        stab_row2.addWidget(self.stability_timeout_sb)
        params_l.addLayout(stab_row2)
        
        # Phase averaging and other params
        other_row = QHBoxLayout()
        self.phase_avg_le = QLineEdit("1.0")
        self.scan_name = QLineEdit("")
        self.comment = QLineEdit("")
        
        other_row.addWidget(QLabel("Phase avg (s):"))
        other_row.addWidget(self.phase_avg_le)
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
        self.prog = QProgressBar()
        self.prog.setMinimum(0)
        self.prog.setValue(0)
        ctl.addWidget(self.estimate_btn)
        ctl.addWidget(self.start_btn)
        ctl.addWidget(self.monitor_btn)
        ctl.addWidget(self.abort_btn)
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
        
        # Initial state
        self._on_ratio_toggle(False)
        self._on_omega_mode_toggle(True)

    def _browse_slm_calib(self):
        """Browse for SLM calibration file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SLM Calibration File", 
            str(_data_root()), 
            "Text files (*.txt *.dat);;All files (*.*)"
        )
        if path:
            self.slm_calib_path_le.setText(path)

    def _load_slm_calib(self):
        """Load SLM calibration from file"""
        path = self.slm_calib_path_le.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a calibration file first.")
            return
        
        try:
            self._slm_calib_alpha, self._slm_calib_intensity = load_slm_calibration(path)
            n_points = len(self._slm_calib_alpha)
            alpha_range = f"[{self._slm_calib_alpha.min():.3f}, {self._slm_calib_alpha.max():.3f}]"
            intensity_range = f"[{self._slm_calib_intensity.min():.3f}, {self._slm_calib_intensity.max():.3f}]"
            self.slm_calib_status.setText(
                f"Status: Loaded {n_points} points, alpha {alpha_range}, intensity {intensity_range}"
            )
            self.slm_calib_status.setStyleSheet("QLabel { color: green; }")
            self._update_max_intensity()
            self._log(f"SLM calibration loaded: {n_points} points from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load calibration:\n{str(e)}")
            self.slm_calib_status.setText("Status: Load failed")
            self.slm_calib_status.setStyleSheet("QLabel { color: red; }")

    def _on_omega_mode_toggle(self, checked):
        """Toggle between waveplate and SLM mode for omega"""
        is_wp_mode = self.omega_wp_radio.isChecked()
        self.wp_group.setEnabled(is_wp_mode)
        self.slm_group.setEnabled(not is_wp_mode)
        
        # Check the background_ref checkbox by default when switching to SLM mode
        if not is_wp_mode and self.enable_ratio_cb.isChecked():
            self.slm_background_ref_cb.setChecked(True)
        
        self._update_max_intensity()

    def _on_ratio_toggle(self, checked):
        self.omega_wp_radio.setEnabled(checked)
        self.omega_slm_radio.setEnabled(checked)
        self.wp_group.setEnabled(checked and self.omega_wp_radio.isChecked())
        self.slm_group.setEnabled(checked and self.omega_slm_radio.isChecked())
        self.total_intensity_le.setEnabled(checked)
        self.omega_max_power_le.setEnabled(checked)
        self.omega_waist_le.setEnabled(checked)
        self.omega_pulse_duration_le.setEnabled(checked)
        self.omega_rep_rate_le.setEnabled(checked)
        self.omega_beam_split_cb.setEnabled(checked)
        self.omega_beam_split_ratio_le.setEnabled(checked and self.omega_beam_split_cb.isChecked())
        self.omega2_max_power_le.setEnabled(checked)
        self.omega2_waist_le.setEnabled(checked)
        self.omega2_pulse_duration_le.setEnabled(checked)
        self.omega2_rep_rate_le.setEnabled(checked)
        self.ratio_start.setEnabled(checked)
        self.ratio_end.setEnabled(checked)
        self.ratio_step.setEnabled(checked)
        self._update_max_intensity()

    def _update_max_intensity(self):
        """Calculate and display maximum achievable intensity"""
        if not self.enable_ratio_cb.isChecked():
            self.intensity_max_label.setText("I_tot_max: -- W/cm²")
            return
        
        try:
            # Get omega parameters
            omega_max_power_W = float(self.omega_max_power_le.text())
            omega_waist_um = float(self.omega_waist_le.text())
            omega_pulse_duration_fs = float(self.omega_pulse_duration_le.text())
            omega_rep_rate_kHz = float(self.omega_rep_rate_le.text())
            
            # Get 2omega parameters
            omega2_max_power_W = float(self.omega2_max_power_le.text())
            omega2_waist_um = float(self.omega2_waist_le.text())
            omega2_pulse_duration_fs = float(self.omega2_pulse_duration_le.text())
            omega2_rep_rate_kHz = float(self.omega2_rep_rate_le.text())
            
            # Calculate max intensity for omega
            I_max_omega = calculate_max_intensity(
                omega_max_power_W, omega_waist_um,
                omega_pulse_duration_fs, omega_rep_rate_kHz
            )
            
            # Apply beam split factor if enabled
            if self.omega_beam_split_cb.isChecked():
                try:
                    split_ratio = float(self.omega_beam_split_ratio_le.text())
                    split_ratio = np.clip(split_ratio, 0.0, 1.0)
                    # split_ratio is fraction in B, so A gets (1 - split_ratio)
                    I_max_omega *= (1.0 - split_ratio)
                except (ValueError, AttributeError):
                    pass
            
            # For SLM mode, multiply by max calibration intensity
            if self.omega_slm_radio.isChecked() and self._slm_calib_intensity is not None:
                I_max_omega *= float(self._slm_calib_intensity.max())
            
            I_max_2omega = calculate_max_intensity(
                omega2_max_power_W, omega2_waist_um,
                omega2_pulse_duration_fs, omega2_rep_rate_kHz
            )
            
            # Total max intensity is the MINIMUM of the two
            I_tot_max = min(I_max_omega, I_max_2omega)
            
            # Check if user's requested I_tot exceeds limit
            try:
                I_tot_requested = float(self.total_intensity_le.text())
                
                if I_tot_requested > I_tot_max:
                    # Calculate achievable R range
                    R_min = max(0.0, 1.0 - I_max_omega / I_tot_requested)
                    R_max = min(1.0, I_max_2omega / I_tot_requested)
                    
                    self.intensity_max_label.setText(
                        f"I_tot_max: {I_tot_max:.3e} W/cm² | "
                        f"R ∈ [{R_min:.3f}, {R_max:.3f}]"
                    )
                    self.intensity_max_label.setStyleSheet(
                        "QLabel { color: red; font-weight: bold; }"
                    )
                else:
                    self.intensity_max_label.setText(f"I_tot_max: {I_tot_max:.3e} W/cm²")
                    self.intensity_max_label.setStyleSheet(
                        "QLabel { color: blue; font-weight: bold; }"
                    )
            except (ValueError, ZeroDivisionError):
                self.intensity_max_label.setText(f"I_tot_max: {I_tot_max:.3e} W/cm²")
                self.intensity_max_label.setStyleSheet(
                    "QLabel { color: blue; font-weight: bold; }"
                )
        
        except (ValueError, ZeroDivisionError):
            self.intensity_max_label.setText("I_tot_max: -- W/cm²")

    def _add_det_row(self):
        det_key = self.det_picker.currentText().strip()
        if not det_key:
            QMessageBox.warning(self, "Pick a detector", "Select a detector to add.")
            return
        r = self.det_tbl.rowCount()
        self.det_tbl.insertRow(r)
        self.det_tbl.setItem(r, 0, QTableWidgetItem(det_key))
        self.det_tbl.setItem(r, 1, QTableWidgetItem("5000"))
        self.det_tbl.setItem(r, 2, QTableWidgetItem("1"))

    def _remove_det_row(self):
        rows = sorted({i.row() for i in self.det_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.det_tbl.removeRow(r)

    def _refresh_devices(self):
        self.phase_ctrl_picker.clear()
        for k in REGISTRY.keys("phaselock:"):
            self.phase_ctrl_picker.addItem(k)
        
        self.det_picker.clear()
        for prefix in ("camera:daheng:", "camera:andor:", "spectrometer:avaspec:", "powermeter:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" not in k:
                    self.det_picker.addItem(k)
        
        # Populate waveplate pickers
        self.wp_omega_picker.clear()
        self.wp_2omega_picker.clear()
        for k in REGISTRY.keys("stage:"):
            if not k.startswith("stage:serial:"):
                self.wp_omega_picker.addItem(k)
                self.wp_2omega_picker.addItem(k)

    def _positions(self, start, end, step):
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if end >= start:
            n = int((end - start) / step)
            vals = [start + i * step for i in range(n + 1)]
            if abs(vals[-1] - end) > 1e-9:
                vals.append(end)
        else:
            n = int((start - end) / step)
            vals = [start - i * step for i in range(n + 1)]
            if abs(vals[-1] - end) > 1e-9:
                vals.append(end)
        return vals

    def _estimate_time(self):
        """Estimate total scan time"""
        try:
            p = self._collect_params(validate=False)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return
        
        # Calculate timing
        n_ratios = len(p["ratio_values"]) if p["enable_ratio"] else 1
        n_phases = len(p["setpoints"])
        n_detectors = len(p["detector_params"])
        
        # Time per point
        avg_stability_time = p["stability_timeout"] / 2.0
        phase_avg_time = p["phase_avg"]
        
        # Detector acquisition time
        detector_time = 0
        for det_key, params in p["detector_params"].items():
            exposure_or_int = float(params[0]) if len(params) >= 1 else 0
            averages = int(params[1]) if len(params) >= 2 else 1
            
            if det_key.startswith("camera:"):
                t = (exposure_or_int / 1e6) * averages
            elif det_key.startswith("spectrometer:"):
                t = (exposure_or_int / 1e3) * averages
            elif det_key.startswith("powermeter:"):
                t = (exposure_or_int / 1e3) * averages
            else:
                t = 1.0
            
            detector_time += t
        
        time_per_phase = avg_stability_time + phase_avg_time + detector_time
        total_scan_time = n_ratios * n_phases * time_per_phase
        
        hours = int(total_scan_time // 3600)
        minutes = int((total_scan_time % 3600) // 60)
        seconds = int(total_scan_time % 60)
        
        msg = f"**Scan Configuration:**\n\n"
        
        if p["enable_ratio"]:
            msg += f"• Ratio points: {n_ratios}\n"
            msg += f"• Phase points per ratio: {n_phases}\n"
        else:
            msg += f"• Phase points: {n_phases}\n"
        
        msg += f"• Detectors: {n_detectors}\n"
        msg += f"• Total acquisitions: {n_ratios * n_phases * n_detectors}\n\n"
        
        msg += f"**Timing per point (estimated):**\n\n"
        msg += f"• Avg stability wait: ~{avg_stability_time:.2f} s\n"
        msg += f"• Phase averaging: {phase_avg_time:.2f} s\n"
        msg += f"• Detector acquisition: {detector_time:.2f} s\n"
        msg += f"• Total per point: ~{time_per_phase:.2f} s\n"
        
        msg += f"\n**Estimated total time: "
        if hours > 0:
            msg += f"{hours}h {minutes}min {seconds}s**"
        elif minutes > 0:
            msg += f"{minutes}min {seconds}s**"
        else:
            msg += f"{seconds}s**"
        
        msg += f"\n\nNote: Actual time may vary based on phase lock stability."
        
        QMessageBox.information(self, "Scan Time Estimate", msg)
        self._log(f"Estimated scan time: {hours}h {minutes}min {seconds}s (approximate)")

    def _collect_params(self, validate=True):
        """Collect all scan parameters and validate"""
        phase_ctrl_key = self.phase_ctrl_picker.currentText().strip()
        if not phase_ctrl_key:
            raise ValueError("Select a phase lock controller.")

        try:
            start = float(self.sp_start.text())
            end = float(self.sp_end.text())
            step = float(self.sp_step.text())
            setpoints = self._positions(start, end, step)
        except ValueError as e:
            raise ValueError(f"Invalid setpoint parameters: {e}")

        if self.det_tbl.rowCount() == 0:
            raise ValueError("Add at least one detector.")

        detector_params = {}
        for r in range(self.det_tbl.rowCount()):
            det = (self.det_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not det:
                raise ValueError(f"Empty detector key at row {r+1}.")
            p1 = (self.det_tbl.item(r, 1) or QTableWidgetItem("0")).text()
            p2 = (self.det_tbl.item(r, 2) or QTableWidgetItem("1")).text()
            detector_params[det] = (int(float(p1)), int(float(p2)))

        max_phase_error = float(self.max_phase_error_sb.value())
        max_phase_std = float(self.max_phase_std_sb.value())
        stability_check_window = float(self.stability_check_window_sb.value())
        stability_timeout = float(self.stability_timeout_sb.value())
        
        try:
            phase_avg = float(self.phase_avg_le.text())
            if phase_avg <= 0:
                raise ValueError("Phase averaging time must be positive")
        except ValueError:
            raise ValueError("Invalid phase averaging time")
        
        name = self.scan_name.text().strip()
        if not name:
            raise ValueError("Enter a scan name.")
        comment = self.comment.text()

        # Ratio scan parameters
        enable_ratio = self.enable_ratio_cb.isChecked()
        omega_control_mode = "slm" if self.omega_slm_radio.isChecked() else "waveplate"
        
        wp_omega_key = None
        wp_2omega_key = None
        slm_class_name = None
        slm_field_name = None
        slm_screen = 3
        total_intensity_W_cm2 = None
        ratio_values = None
        omega_max_power_W = None
        omega_waist_um = None
        omega_pulse_duration_fs = None
        omega_rep_rate_kHz = None
        omega_beam_split = False
        omega_beam_split_ratio = 0.5  # Fraction in B beam (A gets 1-ratio)
        omega2_max_power_W = None
        omega2_waist_um = None
        omega2_pulse_duration_fs = None
        omega2_rep_rate_kHz = None
        background_w_ref = False

        if enable_ratio:
            # Common parameters
            try:
                total_intensity_W_cm2 = float(self.total_intensity_le.text())
                if validate and total_intensity_W_cm2 <= 0:
                    raise ValueError("Total intensity must be positive")
            except ValueError:
                if validate:
                    raise ValueError("Invalid total intensity value.")
                total_intensity_W_cm2 = None
            
            # Omega mode-specific validation
            if omega_control_mode == "waveplate":
                wp_omega_key = self.wp_omega_picker.currentText().strip()
                if validate and not wp_omega_key:
                    raise ValueError("Select omega waveplate.")
                
                # Validate waveplate calibration
                if validate and wp_omega_key:
                    wp_omega_idx = _wp_index_from_stage_key(wp_omega_key)
                    if wp_omega_idx is None:
                        raise ValueError("Invalid omega waveplate.")
                    
                    calib = REGISTRY.get(_reg_key_calib(wp_omega_idx))
                    if not calib or not isinstance(calib, (tuple, list)) or len(calib) < 2:
                        raise ValueError(
                            f"Omega waveplate ({wp_omega_key}) not calibrated.\n"
                            f"Please calibrate in 'Waveplate Calibration' tab."
                        )
                    
                    pm = REGISTRY.get(_reg_key_powermode(wp_omega_idx))
                    if not isinstance(pm, bool) or not pm:
                        raise ValueError(
                            f"Omega waveplate ({wp_omega_key}) power mode OFF.\n"
                            f"Enable 'Power Mode' in Stage Control window."
                        )
            else:  # SLM mode
                slm_class_name = self.slm_class_le.text().strip()
                slm_field_name = self.slm_field_le.text().strip()
                background_w_ref = self.slm_background_ref_cb.isChecked()
                try:
                    slm_screen = int(self.slm_screen_le.text())
                except ValueError:
                    slm_screen = 3
                
                if validate:
                    if not slm_class_name or not slm_field_name:
                        raise ValueError("SLM class and field names required.")
                    
                    if self._slm_calib_alpha is None or self._slm_calib_intensity is None:
                        raise ValueError("Load SLM calibration file first.")
            
            # 2-omega waveplate (always needed)
            wp_2omega_key = self.wp_2omega_picker.currentText().strip()
            if validate and not wp_2omega_key:
                raise ValueError("Select 2-omega waveplate.")
            
            # Validate 2-omega waveplate
            if validate and wp_2omega_key:
                wp_2omega_idx = _wp_index_from_stage_key(wp_2omega_key)
                if wp_2omega_idx is None:
                    raise ValueError("Invalid 2-omega waveplate.")
                
                calib = REGISTRY.get(_reg_key_calib(wp_2omega_idx))
                if not calib or not isinstance(calib, (tuple, list)) or len(calib) < 2:
                    raise ValueError(
                        f"2-Omega waveplate ({wp_2omega_key}) not calibrated.\n"
                        f"Please calibrate in 'Waveplate Calibration' tab."
                    )
                
                pm = REGISTRY.get(_reg_key_powermode(wp_2omega_idx))
                if not isinstance(pm, bool) or not pm:
                    raise ValueError(
                        f"2-Omega waveplate ({wp_2omega_key}) power mode OFF.\n"
                        f"Enable 'Power Mode' in Stage Control window."
                    )
            
            # Laser parameters
            try:
                omega_max_power_W = float(self.omega_max_power_le.text())
                if validate and omega_max_power_W <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("Omega max power required and must be positive.")
                omega_max_power_W = None
            
            try:
                omega_waist_um = float(self.omega_waist_le.text())
                if validate and omega_waist_um <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("Omega waist required and must be positive.")
                omega_waist_um = None
            
            try:
                omega_pulse_duration_fs = float(self.omega_pulse_duration_le.text())
                if validate and omega_pulse_duration_fs <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("Omega pulse duration required and must be positive.")
                omega_pulse_duration_fs = None
            
            try:
                omega_rep_rate_kHz = float(self.omega_rep_rate_le.text())
                if validate and omega_rep_rate_kHz <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("Omega rep rate required and must be positive.")
                omega_rep_rate_kHz = None
            
            try:
                omega2_max_power_W = float(self.omega2_max_power_le.text())
                if validate and omega2_max_power_W <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("2-Omega max power required and must be positive.")
                omega2_max_power_W = None
            
            try:
                omega2_waist_um = float(self.omega2_waist_le.text())
                if validate and omega2_waist_um <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("2-Omega waist required and must be positive.")
                omega2_waist_um = None
            
            try:
                omega2_pulse_duration_fs = float(self.omega2_pulse_duration_le.text())
                if validate and omega2_pulse_duration_fs <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("2-Omega pulse duration required and must be positive.")
                omega2_pulse_duration_fs = None
            
            try:
                omega2_rep_rate_kHz = float(self.omega2_rep_rate_le.text())
                if validate and omega2_rep_rate_kHz <= 0:
                    raise ValueError()
            except (ValueError, AttributeError):
                if validate:
                    raise ValueError("2-Omega rep rate required and must be positive.")
                omega2_rep_rate_kHz = None
            
            try:
                r_start = float(self.ratio_start.text())
                r_end = float(self.ratio_end.text())
                r_step = float(self.ratio_step.text())
                ratio_values = self._positions(r_start, r_end, r_step)
                
                if validate and any(r < 0 or r > 1 for r in ratio_values):
                    raise ValueError("Ratio values must be between 0 and 1.")
            except ValueError as e:
                if validate:
                    raise ValueError(f"Invalid ratio parameters: {e}")
                ratio_values = [0.5]
            
            # Get beam split option and ratio
            omega_beam_split = self.omega_beam_split_cb.isChecked()
            try:
                # Ratio is fraction in B beam (A gets 1 - ratio)
                omega_beam_split_ratio = float(self.omega_beam_split_ratio_le.text())
                omega_beam_split_ratio = np.clip(omega_beam_split_ratio, 0.0, 1.0)
            except (ValueError, AttributeError):
                omega_beam_split_ratio = 0.5

            # Validate total intensity against max
            if validate and all([
                omega_max_power_W, omega_waist_um, omega_pulse_duration_fs, omega_rep_rate_kHz,
                omega2_max_power_W, omega2_waist_um, omega2_pulse_duration_fs, omega2_rep_rate_kHz
            ]):
                try:
                    I_max_omega = calculate_max_intensity(
                        omega_max_power_W, omega_waist_um,
                        omega_pulse_duration_fs, omega_rep_rate_kHz
                    )
                    
                    # Apply beam split (ratio is fraction in B, so A gets 1 - ratio)
                    if omega_beam_split:
                        I_max_omega *= (1.0 - omega_beam_split_ratio)
                    
                    # Apply SLM calibration max intensity
                    if omega_control_mode == "slm" and self._slm_calib_intensity is not None:
                        I_max_omega *= float(self._slm_calib_intensity.max())
                    
                    I_max_2omega = calculate_max_intensity(
                        omega2_max_power_W, omega2_waist_um,
                        omega2_pulse_duration_fs, omega2_rep_rate_kHz
                    )
                    I_tot_max = min(I_max_omega, I_max_2omega)
                    
                except ZeroDivisionError:
                    raise ValueError("Invalid beam parameters (check waist, pulse duration, rep rate).")

        return {
            "phase_ctrl_key": phase_ctrl_key,
            "setpoints": setpoints,
            "detector_params": detector_params,
            "max_phase_error": max_phase_error,
            "max_phase_std": max_phase_std,
            "stability_check_window": stability_check_window,
            "stability_timeout": stability_timeout,
            "phase_avg": phase_avg,
            "scan_name": name,
            "comment": comment,
            "enable_ratio": enable_ratio,
            "omega_control_mode": omega_control_mode,
            "wp_omega_key": wp_omega_key,
            "wp_2omega_key": wp_2omega_key,
            "slm_class_name": slm_class_name,
            "slm_field_name": slm_field_name,
            "slm_screen": slm_screen,
            "total_intensity_W_cm2": total_intensity_W_cm2,
            "ratio_values": ratio_values,
            "omega_max_power_W": omega_max_power_W,
            "omega_waist_um": omega_waist_um,
            "omega_pulse_duration_fs": omega_pulse_duration_fs,
            "omega_rep_rate_kHz": omega_rep_rate_kHz,
            "omega_beam_split": omega_beam_split,
            "omega_beam_split_ratio": omega_beam_split_ratio,
            "omega2_max_power_W": omega2_max_power_W,
            "omega2_waist_um": omega2_waist_um,
            "omega2_pulse_duration_fs": omega2_pulse_duration_fs,
            "omega2_rep_rate_kHz": omega2_rep_rate_kHz,
            "background_w_ref": background_w_ref,
        }

    def _start(self):
        try:
            p = self._collect_params(validate=True)
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return
        
        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(False, None, False)
        self._log("Scan started...")

    def _open_monitor(self):
        """Open or bring to front the monitor window"""
        if self._monitor_window is not None:
            self._monitor_window.show()
            self._monitor_window.raise_()
            self._monitor_window.activateWindow()

    def _launch(self, background, existing, background_w_ref=False):
        p = self._cached_params
        if not p:
            return

        # Create monitor window for ratio scans
        if p["enable_ratio"] and not background:
            self._monitor_window = MonitorWindow(p["ratio_values"], p["setpoints"], parent=None)
            self._monitor_window.show()
            self.monitor_btn.setEnabled(True)

        self._thread = QThread(self)
        self._worker = TwoColorScanWorker(
            phase_ctrl_key=p["phase_ctrl_key"],
            setpoints=p["setpoints"],
            detector_params=p["detector_params"],
            max_phase_error_rad=p["max_phase_error"],
            max_phase_std_rad=p["max_phase_std"],
            stability_check_window_s=p["stability_check_window"],
            stability_timeout_s=p["stability_timeout"],
            phase_avg_s=p["phase_avg"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            enable_ratio_scan=p["enable_ratio"],
            omega_control_mode=p["omega_control_mode"],
            wp_omega_key=p["wp_omega_key"],
            wp_2omega_key=p["wp_2omega_key"],
            total_intensity_W_cm2=p["total_intensity_W_cm2"],
            ratio_values=p["ratio_values"],
            omega_max_power_W=p["omega_max_power_W"],
            omega_waist_um=p["omega_waist_um"],
            omega_pulse_duration_fs=p["omega_pulse_duration_fs"],
            omega_rep_rate_kHz=p["omega_rep_rate_kHz"],
            omega_beam_split=p["omega_beam_split"],
            omega_beam_split_ratio=p["omega_beam_split_ratio"],
            slm_class_name=p["slm_class_name"],
            slm_field_name=p["slm_field_name"],
            slm_screen=p["slm_screen"],
            slm_calib_alpha=self._slm_calib_alpha,
            slm_calib_intensity=self._slm_calib_intensity,
            omega2_max_power_W=p["omega2_max_power_W"],
            omega2_waist_um=p["omega2_waist_um"],
            omega2_pulse_duration_fs=p["omega2_pulse_duration_fs"],
            omega2_rep_rate_kHz=p["omega2_rep_rate_kHz"],
            background=background,
            existing_scan_log=existing,
            background_w_ref=background_w_ref,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        
        # Connect monitor updates
        if self._monitor_window is not None:
            self._worker.monitor_update.connect(self._monitor_window.update_data)

        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        n_ratios = len(p["ratio_values"]) if p["enable_ratio"] else 1
        n_phases = len(p["setpoints"]) if not background else 1
        total = n_ratios * n_phases * len(p["detector_params"])
        self.prog.setMaximum(total)
        self.prog.setValue(0)

        self._thread.start()

    def _launch_background_w_ref(self, existing):
        """Launch background scan with alpha=1 (beam dumped) - single acquisition, no ratios"""
        p = self._cached_params
        if not p:
            return
        
        # Set alpha to 1.0 (beam fully dumped)
        try:
            slm_class_name = p["slm_class_name"]
            slm_field_name = p["slm_field_name"]
            slm_screen = p["slm_screen"]
            
            # Set SLM alpha = 1.0
            active_classes = REGISTRY.get("slm:red:active_classes") or []
            if slm_class_name not in active_classes:
                self._log(f"ERROR: SLM class '{slm_class_name}' is not active.")
                return
            
            widgets = REGISTRY.get("slm:red:widgets") or []
            phase_widget = None
            for w in widgets:
                if getattr(w, "name_", lambda: "")() == slm_class_name:
                    phase_widget = w
                    break
            
            if phase_widget is None:
                self._log(f"ERROR: SLM widget for '{slm_class_name}' not found")
                return
            
            if not hasattr(phase_widget, slm_field_name):
                self._log(f"ERROR: Field '{slm_field_name}' not found in '{slm_class_name}'")
                return
            
            widget = getattr(phase_widget, slm_field_name)
            widget.setText("1.0")
            
            # Compose and publish
            slm_window = REGISTRY.get("slm:red:window")
            if slm_window is None:
                self._log("ERROR: SLM window not found")
                return
            
            levels = slm_window.compose_levels()
            
            slm_red = REGISTRY.get("slm:red:controller")
            if slm_red is None:
                self._log("ERROR: Red SLM controller not found")
                return
            
            slm_red.publish(levels, screen_num=slm_screen)
            
            self._log(f"Set SLM {slm_class_name}:{slm_field_name} = 1.0 (beam fully dumped)")
            
            # Wait a bit for SLM to settle
            import time
            time.sleep(0.5)
            
        except Exception as e:
            self._log(f"ERROR setting SLM to alpha=1: {e}")
            return
        
        # Set 2omega waveplate to zero power
        try:
            if p["wp_2omega_key"]:
                self._log("Setting 2omega waveplate to zero power...")
                wp_2omega_idx = _wp_index_from_stage_key(p["wp_2omega_key"])
                if wp_2omega_idx:
                    amp_off = REGISTRY.get(_reg_key_calib(wp_2omega_idx)) or (None, None)
                    if amp_off[1] is not None:
                        phase_deg = float(amp_off[1])
                        REGISTRY.register(_reg_key_maxvalue(wp_2omega_idx), float(p["omega2_max_power_W"]))
                        angle = power_to_angle(0.0, 1.0, phase_deg)
                        stage = REGISTRY.get(p["wp_2omega_key"])
                        if stage:
                            stage.move_to(float(angle), blocking=True)
                            self._log(f"  {p['wp_2omega_key']} → 0 W (angle: {angle:.3f}°)")
        except Exception as e:
            self._log(f"Warning: Could not set 2omega to zero: {e}")
        
        # Launch a single acquisition (no ratio scan, no phase scan)
        # Use ratio_values=[None] to indicate single point
        self._thread = QThread(self)
        self._worker = TwoColorScanWorker(
            phase_ctrl_key=p["phase_ctrl_key"],
            setpoints=p["setpoints"],
            detector_params=p["detector_params"],
            max_phase_error_rad=p["max_phase_error"],
            max_phase_std_rad=p["max_phase_std"],
            stability_check_window_s=p["stability_check_window"],
            stability_timeout_s=p["stability_timeout"],
            phase_avg_s=p["phase_avg"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            enable_ratio_scan=False,  # Disable ratio scan for this reference
            omega_control_mode=p["omega_control_mode"],
            wp_omega_key=p["wp_omega_key"],
            wp_2omega_key=p["wp_2omega_key"],
            total_intensity_W_cm2=p["total_intensity_W_cm2"],
            ratio_values=[0.0],  # Dummy value
            omega_max_power_W=p["omega_max_power_W"],
            omega_waist_um=p["omega_waist_um"],
            omega_pulse_duration_fs=p["omega_pulse_duration_fs"],
            omega_rep_rate_kHz=p["omega_rep_rate_kHz"],
            omega_beam_split=p["omega_beam_split"],
            omega_beam_split_ratio=p["omega_beam_split_ratio"],
            slm_class_name=p["slm_class_name"],
            slm_field_name=p["slm_field_name"],
            slm_screen=p["slm_screen"],
            slm_calib_alpha=self._slm_calib_alpha,
            slm_calib_intensity=self._slm_calib_intensity,
            omega2_max_power_W=p["omega2_max_power_W"],
            omega2_waist_um=p["omega2_waist_um"],
            omega2_pulse_duration_fs=p["omega2_pulse_duration_fs"],
            omega2_rep_rate_kHz=p["omega2_rep_rate_kHz"],
            background=True,
            existing_scan_log=existing,
            background_w_ref=True,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished_background_w_ref)  # Different handler!

        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        # Single point acquisition
        total = len(p["detector_params"])
        self.prog.setMaximum(total)
        self.prog.setValue(0)

        self._thread.start()

    def _abort(self):
        if self._worker:
            self._worker.abort = True
            self.abort_btn.setEnabled(False)

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
        self.start_btn.setEnabled(True)
        self.monitor_btn.setEnabled(False)

        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

        """
        # Ask for background_w_ref scan first (SLM mode only)
        if not self._doing_background and self._last_scan_log_path is not None:
            p = self._cached_params
            if p and p["omega_control_mode"] == "slm" and p["background_w_ref"]:
                reply = QMessageBox.question(
                    self,
                    "Run Reference Scan?",
                    "The scan finished.\n\nDo you want to run the REFERENCE scan now (alpha=1, on-axis beam dumped)?\n",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self._log("Launching background with reference scan (alpha=1)...")
                    self._launch_background_w_ref(self._last_scan_log_path)
                    return
        """
        # Ask for normal background scan
        if not self._doing_background and self._last_scan_log_path is not None:
            reply = QMessageBox.question(
                self,
                "Run Background Scan?",
                "The scan finished.\n\nDo you want to run the BACKGROUND scan now?\n"
                "If yes, cut the gas and wait 3-5min before continuing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._doing_background = True
                self._log("Launching background scan...")
                self._launch(background=True, existing=self._last_scan_log_path, background_w_ref=False)
                return

        self._doing_background = False

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        
        scrollbar = self.log.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        
        self.log.append(f"[{ts}] {msg}")
        
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())