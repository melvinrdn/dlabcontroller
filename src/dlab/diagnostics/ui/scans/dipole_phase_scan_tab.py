"""Dipole phase scan: two-foci beta_a scan at fixed waveplate, focus B the reference.

Only beta_a is scanned. alpha is set to its starting value and beta_b is left
as found (except on the A-only frames below), so focus B is the reference,
constant over the scan.

The waveplate sets focus A to I_end at beta_a = 0, and beta_a = 1 - I_n / I_end
scans A down the grid I_n = I_start + n * dI, from high to low intensity.

Optionally the jet position z (one SmarAct axis, absolute positions) is an outer
loop: the whole scan is repeated at each z. Each z can hold single-focus blocks
at its start, middle and end: A only (beta_b = 1) at a few A intensities, then
B only (beta_a = 1). They give the drift of each focus during the scan, and the
tab writes the signal of each one, relative to the first, to the log panel.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from scipy.ndimage import median_filter
from scipy.ndimage import rotate as sp_rotate

from dlab.core.device_registry import REGISTRY
from dlab.diagnostics.ui.scans.grid_scan_tab import (
    GridScanTab,
    _reg_key_calib,
    _reg_key_maxvalue,
    _wp_index_from_stage_key,
    power_to_angle,
)
from dlab.diagnostics.ui.scans.scan_utils import confirm_large_scan, generate_positions


SLM_CLASS_NAME = "TwoFociStochastic"
SLM_FIELD_BETA_A = "le_beta_a"
SLM_FIELD_ALPHA = "le_alpha"
SLM_FIELD_BETA_B = "le_beta_b"
ANDOR_KEY = "camera:andor:andorcam_1"

GAUSSIAN_PEAK_POWER_FACTOR = 0.94  # P_peak = 0.94 * E / tau for a Gaussian pulse (tau = FWHM)
INTENSITY_UNIT = 1e14  # W/cm2, unit of the intensity fields
UM = 1e-6  # SmarAct positions are in m, the z fields in µm
SMARACT_PREFIX = "stage:smaract:"
REL_TOL = 1e-9
BLOCKS = ("start", "mid", "end")  # single-focus blocks of each z
LINE_FLOOR_PX = 25  # running-median window along the dispersion for the line check (lines are ~5 px wide)

# 1 mm uncoated fused silica at 1030 nm, normal incidence: n = 1.450 (Malitson),
# R = ((n-1)/(n+1))^2 = 3.37 % per surface, T = (1-R)^2. Internal reflections
# exit ~10 ps later as separate pulses and do not add to the peak intensity.
DEFAULT_TRANSMISSION = 0.934


# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------


def focus_peak_intensity(
    p_avg_W: float,
    rep_rate_Hz: float,
    tau_s: float,
    transmission: float,
    efficiency: float,
    energy_share: float,
    w_focus_um: float,
) -> float:
    """Peak intensity [W/cm2] of one focus with the waveplate at full power and beta = 0.

    energy_share is the fraction of the energy in that focus: (1 - alpha) for A,
    alpha for B. w_focus is the 1/e^2 intensity radius.
    """
    energy_focus = p_avg_W / rep_rate_Hz * transmission * efficiency * energy_share
    p_peak_focus = GAUSSIAN_PEAK_POWER_FACTOR * energy_focus / tau_s
    w_cm = w_focus_um * 1e-4
    return 2.0 * p_peak_focus / (np.pi * w_cm**2)


def a_grid(I_start: float, I_end: float, dI: float, I_focus_max: float) -> list[tuple[int, float, float]]:
    """(grid index n, I_n, beta_a) of focus A, from high to low intensity; beta_a = 0 at I_end."""
    if not (0.0 < I_start < I_end):
        raise ValueError("Need 0 < I_start < I_end.")
    if dI <= 0:
        raise ValueError("Intensity step must be > 0.")
    if not I_focus_max > 0:
        raise ValueError("Max focus intensity must be > 0.")
    if I_end > I_focus_max * (1.0 + REL_TOL):
        raise ValueError(f"I_end = {I_end:.4e} W/cm2 exceeds the max focus intensity {I_focus_max:.4e} W/cm2.")
    n_max = int(np.floor((I_end - I_start) / dI + REL_TOL))
    return [(n, I_start + n * dI, max(0.0, 1.0 - (I_start + n * dI) / I_end)) for n in range(n_max, -1, -1)]


def single_focus_block(grid: list[tuple[int, float, float]], n_a_only: int) -> list[tuple[str, int, float, float, float | None]]:
    """(frame, grid index, I_A, beta_a, beta_b) of one single-focus block.

    A only (beta_b = 1) at n_a_only of the A intensities, evenly spread from the
    top to the bottom of the grid, then B only (beta_a = 1), so beta_b is back
    to its value at the end of the block. beta_b is None where it stays as set
    on the SLM.
    """
    picks = np.unique(np.linspace(0, len(grid) - 1, n_a_only).round().astype(int)) if n_a_only > 0 and grid else []
    frames = [("A_only", grid[i][0], grid[i][1], grid[i][2], 1.0) for i in picks]
    return frames + [("B_only", -1, 0.0, 1.0, None)]


def scan_points(
    grid: list[tuple[int, float, float]], wp_fraction: float, I_ref: float, I_B: float, beta_b: float,
    z_um: list[float] | None = None, n_a_only: int | None = None,
) -> tuple[list[list[float]], list[dict]]:
    """Explicit worker points [wp_fraction, beta_a, beta_b] and their log extras.

    With z_um (absolute, in µm), the scan is repeated at each jet position and
    every point starts with z in m: [z, wp_fraction, beta_a, beta_b]. With
    n_a_only, each z holds a single-focus block at its start, in the middle of
    the two-foci frames and at its end.
    """
    half = len(grid) // 2
    ab = [("AB", n, I_n, beta, None) for n, I_n, beta in grid]
    points, extras = [], []
    for j, z in enumerate([None] if z_um is None else z_um):
        z_point = [] if z is None else [z * UM]
        z_extra = {} if z is None else {"z_index": j, "z_um": f"{z:.3f}"}
        if n_a_only is None:
            seq = [(f, "") for f in ab]
        else:
            block = single_focus_block(grid, n_a_only)
            seq = ([(f, "start") for f in block] + [(f, "") for f in ab[:half]] + [(f, "mid") for f in block]
                   + [(f, "") for f in ab[half:]] + [(f, "end") for f in block])
        for (frame, n, I_n, beta_a, b_off), blk in seq:
            points.append(z_point + [wp_fraction, beta_a, beta_b if b_off is None else b_off])
            extras.append({
                **z_extra,
                "frame": frame,
                "block": blk,
                "grid_index": n,
                "I_ref_Wcm2": f"{I_ref:.6e}",
                "I_target_Wcm2": f"{I_n:.6e}",
                "I_B_Wcm2": f"{0.0 if b_off is not None else I_B:.6e}",
            })
    return points, extras


def roi_signal(frame: np.ndarray, pre: dict) -> float:
    """Counts of the harmonic lines in the Andor display ROI (rotation + crop of the live window).

    The lines are narrow along the dispersion (x of the display); a running
    median along x, wider than a line, is the floor (camera offset, MCP
    background, broad glow). Static narrow features, such as the edges of the
    MCP ring, are counted too: keep the crop on the lines.
    """
    img = np.asarray(frame, dtype=float)
    if pre["enabled"]:
        if abs(pre["angle"]) > 1e-9:
            img = sp_rotate(img, pre["angle"], reshape=True, order=3, mode="nearest")
        img = img[pre["y0"]:pre["y1"], pre["x0"]:pre["x1"]]
    return float((img - median_filter(img, size=(1, LINE_FLOOR_PX))).sum())


def make_frame_check(extras: list[dict], pre: dict, det_key: str):
    """Frame check for the worker: signal of each single-focus frame relative to the first of its kind."""
    first: dict[tuple, float] = {}

    def check(point_idx: int | None, key: str, frame: np.ndarray) -> str | None:
        if point_idx is None or key != det_key:
            return None
        e = extras[point_idx]
        if e["frame"] == "AB":
            return None
        s = roi_signal(frame, pre)
        kind = (e["frame"], e["grid_index"])
        ref = first.setdefault(kind, s)
        what = "B only" if e["frame"] == "B_only" else f"A only {float(e['I_target_Wcm2']) / INTENSITY_UNIT:.3g}"
        where = (f"z = {e['z_um']} µm, " if "z_um" in e else "") + e["block"]
        ratio = f"{s / ref:.2f} × first" if ref > 0 else "first has no signal"
        return f"Check {where}: {what}: {s:.3g} counts, {ratio}"

    return check


# -----------------------------------------------------------------------------
# Tab
# -----------------------------------------------------------------------------


class DipolePhaseScanTab(GridScanTab):
    """beta_a scan of focus A at fixed waveplate, at each jet position z."""

    _log_source = "DipolePhaseScan"

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)
        main.addWidget(self._create_laser_group())
        main.addWidget(self._create_intensity_group())
        main.addWidget(self._create_z_group())
        main.addWidget(self._create_detectors_group())
        main.addWidget(self._create_parameters_group())
        main.addLayout(self._create_controls_row())

        self._cam_tbl.itemChanged.connect(self._update_preview)
        self._cam_tbl.model().rowsInserted.connect(self._update_preview)
        self._cam_tbl.model().rowsRemoved.connect(self._update_preview)

        refresh_row = QHBoxLayout()
        btn_go_start = QPushButton("Go to min I_A (beta_a max)")
        btn_go_start.clicked.connect(lambda: self._go_to_extreme(start=True))
        refresh_row.addWidget(btn_go_start)
        btn_go_end = QPushButton("Go to max I_A (beta_a min)")
        btn_go_end.clicked.connect(lambda: self._go_to_extreme(start=False))
        refresh_row.addWidget(btn_go_end)
        refresh_row.addStretch(1)
        btn_refresh = QPushButton("Refresh Devices")
        btn_refresh.clicked.connect(self._refresh_devices)
        refresh_row.addWidget(btn_refresh)
        main.addLayout(refresh_row)

        self._ready = True
        self._update_preview()

    def _line(self, text: str) -> QLineEdit:
        le = QLineEdit(text)
        le.textChanged.connect(self._update_preview)
        return le

    def _create_laser_group(self) -> QGroupBox:
        group = QGroupBox("Laser and focus A")
        grid = QGridLayout(group)

        self._p_max_le = self._line("3.5")
        self._rep_le = self._line("8")
        self._tau_le = self._line("220")
        self._trans_le = self._line(f"{DEFAULT_TRANSMISSION}")
        self._eff_le = self._line("0.8")
        self._alpha_le = self._line("0.5")
        self._w_focus_le = self._line("16")  # 13 x 19.5 µm on the IR camera, 2026-10-08

        fields = [
            ("Max power before chamber (W, waveplate at 1)", self._p_max_le),
            ("Rep rate (kHz)", self._rep_le),
            ("Pulse duration FWHM (fs)", self._tau_le),
            ("Transmission", self._trans_le),
            ("Efficiency", self._eff_le),
            ("alpha (A gets 1 - alpha, B gets alpha)", self._alpha_le),
            ("w_focus, 1/e² radius (µm)", self._w_focus_le),
        ]
        for i, (label, le) in enumerate(fields):
            grid.addWidget(QLabel(label), i // 2, 2 * (i % 2))
            grid.addWidget(le, i // 2, 2 * (i % 2) + 1)

        row = (len(fields) + 1) // 2
        grid.addWidget(QLabel("Waveplate"), row, 0)
        self._wp_combo = QComboBox()
        self._wp_combo.currentIndexChanged.connect(self._update_preview)
        grid.addWidget(self._wp_combo, row, 1)
        grid.addWidget(QLabel("SLM screen"), row, 2)
        self._screen_le = self._line("3")
        grid.addWidget(self._screen_le, row, 3)

        self._imax_label = QLabel("")
        grid.addWidget(self._imax_label, row + 1, 0, 1, 4)
        return group

    def _create_intensity_group(self) -> QGroupBox:
        group = QGroupBox("Focus A intensity (1e14 W/cm²): waveplate at I end, beta_a scans A down by up to beta_max")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self._i_end_le = self._line("1.70")
        self._di_le = self._line("0.01")
        self._beta_max_le = self._line("0.35")
        for label, le in [
            ("I end ('max': max of focus A)", self._i_end_le),
            ("I step", self._di_le),
            ("beta_max", self._beta_max_le),
        ]:
            row.addWidget(QLabel(label))
            row.addWidget(le)
        layout.addLayout(row)

        row = QHBoxLayout()
        self._single_cb = QCheckBox("Single-focus blocks at the start, middle and end of each z: A only (beta_b = 1) at")
        self._single_cb.setChecked(True)
        self._single_cb.toggled.connect(self._update_preview)
        row.addWidget(self._single_cb)
        self._n_a_only_le = self._line("3")
        row.addWidget(self._n_a_only_le)
        row.addWidget(QLabel("A intensities, then B only (beta_a = 1)"))
        row.addStretch(1)
        layout.addLayout(row)

        self._scan_label = QLabel("")
        self._scan_label.setWordWrap(True)
        layout.addWidget(self._scan_label)
        return group

    def _create_z_group(self) -> QGroupBox:
        group = QGroupBox("Jet position z (SmarAct, absolute; outer loop: the whole scan at each z)")
        row = QHBoxLayout(group)
        self._z_cb = QCheckBox("Scan z")
        self._z_cb.setChecked(True)
        self._z_cb.toggled.connect(self._update_preview)
        row.addWidget(self._z_cb)
        self._z_combo = QComboBox()
        self._z_combo.currentIndexChanged.connect(self._update_preview)
        row.addWidget(self._z_combo, 1)
        self._z_start_le = self._line("-250")
        self._z_end_le = self._line("750")
        self._z_step_le = self._line("250")
        for label, le in [("start (µm)", self._z_start_le), ("end (µm)", self._z_end_le), ("step (µm)", self._z_step_le)]:
            row.addWidget(QLabel(label))
            row.addWidget(le)
        self._z_label = QLabel("")
        row.addWidget(self._z_label)
        return group

    # -------------------------------------------------------------------------
    # Devices
    # -------------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        current = self._wp_combo.currentText()
        self._wp_combo.blockSignals(True)
        self._wp_combo.clear()
        for k in REGISTRY.keys("stage:"):
            if _wp_index_from_stage_key(k) is not None:
                self._wp_combo.addItem(k)
        idx = self._wp_combo.findText(current)
        if idx >= 0:
            self._wp_combo.setCurrentIndex(idx)
        self._wp_combo.blockSignals(False)

        current = self._z_combo.currentText()
        self._z_combo.blockSignals(True)
        self._z_combo.clear()
        self._z_combo.addItems(REGISTRY.keys(SMARACT_PREFIX))
        idx = self._z_combo.findText(current)
        if idx >= 0:
            self._z_combo.setCurrentIndex(idx)
        self._z_combo.blockSignals(False)

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
        self._update_preview()

    def _sync_power_mode_from_registry(self) -> None:
        pass  # no axes table: the waveplate always runs in power mode here

    # -------------------------------------------------------------------------
    # Scan computation and preview
    # -------------------------------------------------------------------------

    def _beam(self) -> dict:
        def num(le: QLineEdit, name: str) -> float:
            try:
                return float(le.text())
            except ValueError:
                raise ValueError(f"Invalid {name}.")

        beam = {
            "p_max_W": num(self._p_max_le, "max power"),
            "rep_rate_kHz": num(self._rep_le, "rep rate"),
            "tau_fs": num(self._tau_le, "pulse duration"),
            "transmission": num(self._trans_le, "transmission"),
            "efficiency": num(self._eff_le, "efficiency"),
            "alpha": num(self._alpha_le, "alpha"),
            "w_focus_um": num(self._w_focus_le, "w_focus"),
        }
        for key in ("p_max_W", "rep_rate_kHz", "tau_fs", "transmission", "efficiency", "w_focus_um"):
            if not beam[key] > 0:
                raise ValueError(f"{key} must be > 0.")
        if not 0.0 < beam["alpha"] < 1.0:
            raise ValueError("alpha must be in (0, 1): both foci need energy.")

        def peak(share: float) -> float:
            return focus_peak_intensity(
                beam["p_max_W"], beam["rep_rate_kHz"] * 1e3, beam["tau_fs"] * 1e-15,
                beam["transmission"], beam["efficiency"], share, beam["w_focus_um"],
            )

        beam["I_A_max"] = peak(1.0 - beam["alpha"])
        beam["I_B_max"] = peak(beam["alpha"])
        return beam

    def _scan(self) -> tuple[dict, dict, list[tuple[int, float, float]]]:
        beam = self._beam()
        try:
            i_end = self._i_end_le.text().strip().lower()
            scan = {
                "I_end": beam["I_A_max"] if i_end == "max" else float(i_end) * INTENSITY_UNIT,
                "dI": float(self._di_le.text()) * INTENSITY_UNIT,
            }
            beta_max = float(self._beta_max_le.text())
        except ValueError:
            raise ValueError("Invalid intensity value.")
        if not 0.0 < beta_max < 1.0:
            raise ValueError("beta_max must be in (0, 1).")
        # lowest grid point (down from I_end in steps of dI) with beta_a <= beta_max
        scan["I_start"] = scan["I_end"] - np.floor(beta_max * scan["I_end"] / scan["dI"] + REL_TOL) * scan["dI"]
        scan["beta_max"] = beta_max
        grid = a_grid(scan["I_start"], scan["I_end"], scan["dI"], beam["I_A_max"])
        scan["wp_fraction"] = min(1.0, scan["I_end"] / beam["I_A_max"])
        return beam, scan, grid

    def _z_positions(self) -> list[float] | None:
        """Jet positions in µm, or None when z is not scanned."""
        if not self._z_cb.isChecked():
            return None
        try:
            start, end, step = (float(le.text()) for le in (self._z_start_le, self._z_end_le, self._z_step_le))
        except ValueError:
            raise ValueError("Invalid z value.")
        return generate_positions(start, end, step)

    def _n_a_only(self) -> int | None:
        """Number of A-only frames per single-focus block, or None without the blocks."""
        if not self._single_cb.isChecked():
            return None
        try:
            n = int(self._n_a_only_le.text())
        except ValueError:
            raise ValueError("Invalid number of A-only frames.")
        if n < 0:
            raise ValueError("Number of A-only frames must be >= 0.")
        return n

    @staticmethod
    def _slm_widget():
        for w in REGISTRY.get("slm:red:widgets") or []:
            if getattr(w, "name_", lambda: "")() == SLM_CLASS_NAME:
                return w
        return None

    def _slm_beta_b(self) -> float | None:
        """beta_b as currently set on the SLM (left untouched by the scan), or None."""
        w = self._slm_widget()
        try:
            return float(getattr(w, SLM_FIELD_BETA_B).text()) if w is not None else None
        except (AttributeError, ValueError):
            return None

    def _slm_settings(self) -> str:
        """All fields and checkboxes of the TwoFociStochastic widget, for the log header."""
        w = self._slm_widget()
        if w is None:
            return "not available"
        vals = []
        for name in sorted(vars(w)):
            obj = getattr(w, name)
            if name.startswith("le_") and isinstance(obj, QLineEdit):
                vals.append(f"{name[3:]}={obj.text()}")
            elif name.startswith("cb_") and isinstance(obj, QCheckBox):
                vals.append(f"{name[3:]}={obj.isChecked()}")
        return ", ".join(vals)

    @staticmethod
    def _andor_preprocess() -> dict | None:
        """Rotation and crop of the Andor live window display (its saved ROI), or None if it is not open."""
        w = REGISTRY.get(ANDOR_KEY)
        try:
            return {
                "enabled": w._pre_enable_cb.isChecked(),
                "angle": float(w._pre_angle_sb.value()),
                **{k: int(getattr(w, f"_pre_{k}_sb").value()) for k in ("x0", "x1", "y0", "y1")},
            }
        except AttributeError:
            return None

    def _update_preview(self) -> None:
        if not getattr(self, "_ready", False):
            return  # still building the UI
        try:
            beam = self._beam()
            self._imax_label.setText(
                f"Max intensity (waveplate at 1, beta = 0): "
                f"A {beam['I_A_max'] / INTENSITY_UNIT:.4g}, B {beam['I_B_max'] / INTENSITY_UNIT:.4g} × 1e14 W/cm²"
            )
            _, scan, grid = self._scan()
            n_a_only = self._n_a_only()
            z_um = self._z_positions()
        except ValueError as e:
            self._scan_label.setText(str(e))
            return

        warnings = []
        beta_b = self._slm_beta_b()
        if beta_b is None:
            warnings.append("SLM not available: I_B shown for beta_b = 0")
            beta_b = 0.0
        n_single = 0 if n_a_only is None else len(BLOCKS) * len(single_focus_block(grid, n_a_only))
        per_z = len(grid) + n_single
        I_B = beam["I_B_max"] * scan["wp_fraction"] * (1.0 - beta_b)
        text = (f"Waveplate {scan['wp_fraction']:.4f} ({scan['wp_fraction'] * beam['p_max_W']:.4g} W), "
                f"beta_a {grid[0][2]:.4f} – {grid[-1][2]:.4f}, I_B {I_B / INTENSITY_UNIT:.4g}. "
                f"{len(grid)} two-foci points" + (f" + {n_single} single-focus frames" if n_single else "")
                + f" = {per_z} points" + ("." if z_um is None else f" per z, × {len(z_um)} z = {per_z * len(z_um)} points."))
        self._z_label.setText("" if z_um is None else f"{len(z_um)} positions")
        if z_um is not None and not self._z_combo.currentText():
            warnings.append("no SmarAct axis for z (activate the SmarAct window, then Refresh Devices)")
        if not self._wp_calibrated():
            warnings.append(f"waveplate {self._wp_combo.currentText() or '(none)'} has no calibration")
        if n_a_only is not None and self._andor_preprocess() is None:
            warnings.append("Andor live window not open: no check of the single-focus frames")
        if warnings:
            text += "\nWarning: " + "; ".join(warnings) + "."
        self._scan_label.setText(text)

    def _wp_calibrated(self) -> bool:
        wp = _wp_index_from_stage_key(self._wp_combo.currentText())
        if wp is None:
            return False
        amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
        return amp_off[1] is not None

    # -------------------------------------------------------------------------
    # Scan parameters
    # -------------------------------------------------------------------------

    def _collect_params(self) -> dict:
        beam, scan, grid = self._scan()
        wp_key = self._wp_combo.currentText().strip()
        wp = _wp_index_from_stage_key(wp_key)
        if wp is None:
            raise ValueError("Select the waveplate.")
        if not self._wp_calibrated():
            raise ValueError(f"{wp_key}: no calibration loaded.")
        try:
            screen = int(self._screen_le.text())
        except ValueError:
            raise ValueError("Invalid SLM screen.")

        beta_b = self._slm_beta_b()
        beta_b_note = "as set on the SLM"
        if beta_b is None:
            beta_b, beta_b_note = 0.0, "SLM not available, assumed"
        elif beta_b >= 1.0:
            raise ValueError("beta_b = 1 on the SLM: focus B is off (left by an aborted scan?). Set it back first.")

        z_um = self._z_positions()
        z_key = self._z_combo.currentText().strip()
        if z_um is not None and not z_key.startswith(SMARACT_PREFIX):
            raise ValueError("Select the SmarAct axis for z.")
        n_a_only = self._n_a_only()

        I_B = beam["I_B_max"] * scan["wp_fraction"] * (1.0 - beta_b)
        points, extras = scan_points(grid, scan["wp_fraction"], scan["I_end"], I_B, beta_b, z_um, n_a_only)
        if not points:
            raise ValueError("The scan has no points.")

        ax_a = f"slm:{SLM_CLASS_NAME}:{SLM_FIELD_BETA_A}"
        ax_b = f"slm:{SLM_CLASS_NAME}:{SLM_FIELD_BETA_B}"
        axes = [(wp_key, []), (ax_a, []), (ax_b, [])]  # same order as scan_points
        axes_meta = {
            wp_key: {"pm": True, "max_value_W": beam["p_max_W"]},
            ax_a: {"param": SLM_FIELD_BETA_A, "screen": screen},
            ax_b: {"param": SLM_FIELD_BETA_B, "screen": screen},
        }
        if z_um is not None:
            axes.insert(0, (z_key, []))  # outer loop, in m
            axes_meta[z_key] = {"pm": False}

        pre = self._andor_preprocess() if n_a_only is not None else None
        notes = [
            f"Dipole phase scan ({SLM_CLASS_NAME}): only beta_a scanned, focus B is the reference",
            f"  P_max={beam['p_max_W']:.6g} W | rep={beam['rep_rate_kHz']:.6g} kHz | "
            f"tau={beam['tau_fs']:.6g} fs | T={beam['transmission']:.6g} | eff={beam['efficiency']:.6g} | "
            f"alpha={beam['alpha']:.6g} | w_focus={beam['w_focus_um']:.6g} um",
            f"  I_A_max={beam['I_A_max']:.6e} | I_B_max={beam['I_B_max']:.6e} W/cm2 (waveplate at 1, beta=0)",
            f"  A: I_start={scan['I_start']:.6e} | I_end={scan['I_end']:.6e} (beta_a=0) | dI={scan['dI']:.6e} W/cm2 | "
            f"beta_a {grid[0][2]:.6g} to {grid[-1][2]:.6g} (beta_max={scan['beta_max']:.6g}) | {len(grid)} points | wp_fraction={scan['wp_fraction']:.6f}",
            f"  B: beta_b={beta_b:.6g} ({beta_b_note}; 1 only on the A-only frames) | I_B={I_B:.6e} W/cm2",
            "  single-focus frames: none" if n_a_only is None else
            f"  single-focus blocks at the start, middle and end of each z: A only (beta_b=1) at {n_a_only} A intensities, "
            "then B only (beta_a=1)",
            f"  SLM {SLM_CLASS_NAME}: {self._slm_settings()}",
        ]
        if pre is not None:
            notes.append("  check ROI (Andor display): " + ", ".join(f"{k}={v}" for k, v in pre.items()))
        if z_um is not None:
            notes.append(f"  z: {z_key}, {len(z_um)} jet positions (um): " + ", ".join(f"{z:.6g}" for z in z_um)
                         + "; outer loop, the whole scan at each z")

        return {
            "axes": axes,
            "axes_meta": axes_meta,
            "points": points,
            "point_extras": extras,
            "header_notes": notes,
            "frame_check": None if pre is None else make_frame_check(extras, pre, ANDOR_KEY),
            "alpha": beam["alpha"],
            "wp_key": wp_key,
            "n_grid": len(grid),
            "z_um": z_um,
            **self._collect_common_params(),
        }

    def _points_summary(self, p: dict) -> str:
        text = f"{p['n_grid']} A intensities, {len(p['points'])} points"
        if p["z_um"] is not None:
            text += f"\nz: the scan at {len(p['z_um'])} jet positions"
        return text

    # -------------------------------------------------------------------------
    # Scan control
    # -------------------------------------------------------------------------

    def _set_slm_alpha(self, alpha: float) -> None:
        """Write alpha into the TwoFociStochastic widget; the first beta move publishes it."""
        if SLM_CLASS_NAME not in (REGISTRY.get("slm:red:active_classes") or []):
            raise RuntimeError(f"SLM class '{SLM_CLASS_NAME}' is not active on the red SLM.")
        w = self._slm_widget()
        if w is None:
            raise RuntimeError(f"SLM widget for '{SLM_CLASS_NAME}' not found.")
        getattr(w, SLM_FIELD_ALPHA).setText(f"{alpha:.6f}")

    def _go_to_extreme(self, start: bool) -> None:
        """Move to the lowest (start) or highest (end) A intensity of the scan, to check contrast."""
        if not self._start_btn.isEnabled():
            QMessageBox.warning(self, "Scan running", "Wait for the scan to finish.")
            return
        label = "start" if start else "end"
        try:
            beam, scan, grid = self._scan()
            _, I_n, beta_a = grid[-1] if start else grid[0]

            wp_key = self._wp_combo.currentText().strip()
            wp = _wp_index_from_stage_key(wp_key)
            if wp is None:
                raise ValueError("Select the waveplate.")
            amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
            if amp_off[1] is None:
                raise ValueError(f"{wp_key}: no calibration loaded.")
            stage = REGISTRY.get(wp_key)
            if stage is None:
                raise ValueError(f"Stage '{wp_key}' not found.")
            try:
                screen = int(self._screen_le.text())
            except ValueError:
                raise ValueError("Invalid SLM screen.")

            slm_window = REGISTRY.get("slm:red:window")
            slm_red = REGISTRY.get("slm:red:controller")
            if slm_window is None or slm_red is None:
                raise RuntimeError("Red SLM is not active: publish it once from the SLM window.")

            self._set_slm_alpha(beam["alpha"])
            getattr(self._slm_widget(), SLM_FIELD_BETA_A).setText(f"{beta_a:.6f}")
            slm_red.publish(slm_window.compose_levels(), screen_num=screen)

            angle = power_to_angle(scan["wp_fraction"], float(amp_off[1]))
            stage.move_to(angle, blocking=False)
        except Exception as e:
            QMessageBox.critical(self, f"Go to {label}", str(e))
            return

        self._log_message(
            f"Go to {label}: WP fraction {scan['wp_fraction']:.4f} "
            f"({scan['wp_fraction'] * beam['p_max_W']:.4g} W, {angle:.2f} deg), beta_a {beta_a:.4f}, "
            f"I_A {I_n / INTENSITY_UNIT:.4g}e14 W/cm2, alpha {beam['alpha']:.4g}"
        )

    def _on_start(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        if not confirm_large_scan(self, self._total_points(p), len(p["camera_params"])):
            return

        try:
            self._set_slm_alpha(p["alpha"])
        except Exception as e:
            QMessageBox.critical(self, "SLM", str(e))
            return

        wp_key = p["wp_key"]  # not axes[0]: the z axis goes first when scanned
        REGISTRY.register(_reg_key_maxvalue(_wp_index_from_stage_key(wp_key)), float(p["axes_meta"][wp_key]["max_value_W"]))

        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(background=False, existing=None)
        self._log_message("Dipole phase scan started…" + ("" if p["frame_check"] else " (no single-focus check: Andor live window not open)"))
