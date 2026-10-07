"""Two-foci beta_a scan on a common intensity grid, stitched over a waveplate ladder.

Only beta_a is scanned. alpha is set to its starting value and beta_b is left
as found, so focus B is the reference: constant on each rung, and scaled with
focus A by the waveplate from one rung to the next.

One beta_a scan at fixed laser energy covers A intensities
I = (1 - beta_a) * I_ref, i.e. [1 - beta_max, 1] * I_ref. A ladder of reference
intensities I_ref,k (set with the waveplate) covers [I_start, I_end]. Every rung
samples the same grid I_n = I_start + n * dI, so beta_a = 1 - I_n / I_ref,k and
the step in beta_a grows as I_ref,k drops. Overlap is kept minimal: the next
rung's I_ref is the grid point n_shared steps up from the bottom of the current
window, so consecutive rungs share exactly n_shared grid points, measured on
both on purpose to stitch the rungs together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from dlab.core.device_registry import REGISTRY
from dlab.diagnostics.ui.scans.grid_scan_tab import (
    GridScanTab,
    _reg_key_calib,
    _detector_time_estimate_s,
    _reg_key_maxvalue,
    _wp_index_from_stage_key,
    power_to_angle,
)
from dlab.diagnostics.ui.scans.scan_utils import confirm_large_scan


SLM_CLASS_NAME = "TwoFociStochastic"
SLM_FIELD_BETA_A = "le_beta_a"
SLM_FIELD_ALPHA = "le_alpha"
SLM_FIELD_BETA_B = "le_beta_b"

GAUSSIAN_PEAK_POWER_FACTOR = 0.94  # P_peak = 0.94 * E / tau for a Gaussian pulse (tau = FWHM)
INTENSITY_UNIT = 1e14  # W/cm2, unit of the intensity fields
REL_TOL = 1e-9

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


@dataclass
class Rung:
    k: int
    I_ref: float  # W/cm2, intensity at beta_a = 0 on this rung
    wp_fraction: float  # waveplate power fraction giving I_ref
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (grid index n, I_n, beta_a)

    @property
    def grid_indices(self) -> set[int]:
        return {n for n, _, _ in self.points}


def build_ladder(
    I_start: float, I_end: float, dI: float, beta_max: float, n_shared: int, I_focus_max: float
) -> list[Rung]:
    """Rungs from I_end downwards until a window reaches I_start.

    Consecutive rungs share exactly n_shared grid points. Within a rung, points
    go from high to low intensity (beta_a increasing).
    """
    if not (0.0 < I_start < I_end):
        raise ValueError("Need 0 < I_start < I_end.")
    if dI <= 0:
        raise ValueError("Intensity step must be > 0.")
    if not (0.0 < beta_max < 1.0):
        raise ValueError("beta_max must be in (0, 1).")
    if n_shared < 1:
        raise ValueError("Shared points must be >= 1 to stitch the rungs.")
    if not I_focus_max > 0:
        raise ValueError("Max focus intensity must be > 0.")
    if I_end > I_focus_max * (1.0 + REL_TOL):
        raise ValueError(
            f"I_end = {I_end:.4e} W/cm2 exceeds the max focus intensity {I_focus_max:.4e} W/cm2."
        )

    n_max = int(np.floor((I_end - I_start) / dI + REL_TOL))
    grid = [(n, I_start + n * dI) for n in range(n_max + 1)]
    eps = REL_TOL * I_end

    rungs: list[Rung] = []
    k = 0
    I_ref = I_end
    while True:
        lo = (1.0 - beta_max) * I_ref
        rung = Rung(k=k, I_ref=I_ref, wp_fraction=min(1.0, I_ref / I_focus_max))
        for n, I_n in reversed(grid):
            if lo - eps <= I_n <= I_ref + eps:
                beta = float(np.clip(1.0 - I_n / I_ref, 0.0, beta_max))
                rung.points.append((n, I_n, beta))
        rungs.append(rung)
        if lo <= I_start + eps:
            break
        if len(rung.points) <= n_shared:
            raise ValueError(
                f"Rung {k} window holds only {len(rung.points)} grid point(s), not more than the "
                f"{n_shared} shared; the ladder cannot move down. Decrease the intensity step."
            )
        I_ref = rung.points[-n_shared][1]
        k += 1
        if k > 1000:
            raise ValueError("Ladder has more than 1000 rungs; check the step and beta_max.")
    return rungs


def reference_intensity(rung: Rung, I_B_max: float, beta_b: float) -> float:
    """Focus B intensity on a rung: constant while beta_a is scanned."""
    return I_B_max * rung.wp_fraction * (1.0 - beta_b)


def ladder_to_points(
    rungs: list[Rung], I_B_max: float, beta_b: float
) -> tuple[list[list[float]], list[dict]]:
    """Explicit worker points [wp_fraction, beta_a] and their log extras."""
    points, extras = [], []
    for rung in rungs:
        I_B = reference_intensity(rung, I_B_max, beta_b)
        for n, I_n, beta in rung.points:
            points.append([rung.wp_fraction, beta])
            extras.append({
                "rung": rung.k,
                "grid_index": n,
                "I_ref_Wcm2": f"{rung.I_ref:.6e}",
                "I_target_Wcm2": f"{I_n:.6e}",
                "I_B_Wcm2": f"{I_B:.6e}",
            })
    return points, extras


# -----------------------------------------------------------------------------
# Tab
# -----------------------------------------------------------------------------


class LadderPlanWindow(QWidget):
    """Planned ladder (I_A vs I_B per rung) and planned cumulative time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Intensity Ladder Plan")
        self.resize(820, 360)
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 3.4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_i, self.ax_t = self.figure.subplots(1, 2)
        layout.addWidget(self.canvas)

    def show_message(self, msg: str) -> None:
        for ax in (self.ax_i, self.ax_t):
            ax.clear()
            ax.set_axis_off()
        self.ax_i.text(0.0, 0.5, msg, transform=self.ax_i.transAxes, wrap=True)
        self.canvas.draw_idle()

    def update_plan(self, rungs: list[Rung], I_B_max: float, beta_b: float, t_point_s: float) -> None:
        ax_i, ax_t = self.ax_i, self.ax_t
        for ax in (ax_i, ax_t):
            ax.clear()
            ax.set_axis_on()
        u = 1.0 / INTENSITY_UNIT

        n_done = 0
        i_max = 0.0
        for rung in rungs:
            color = f"C{rung.k % 10}"
            I_A = np.array([I for _, I, _ in rung.points]) * u
            I_B = reference_intensity(rung, I_B_max, beta_b) * u
            ax_i.plot(np.full_like(I_A, I_B), I_A, "o-", ms=3, lw=1, color=color, label=f"rung {rung.k}")
            i_max = max(i_max, I_B, I_A.max(initial=0.0))
            idx = np.arange(n_done + 1, n_done + len(rung.points) + 1)
            ax_t.plot(idx, idx * t_point_s / 60.0, ".", ms=3, color=color)
            n_done += len(rung.points)

        ax_i.plot([0, i_max * 1.05], [0, i_max * 1.05], "k--", lw=0.8, label="I_A = I_B")
        ax_i.set_xlabel("I_B, reference (1e14 W/cm$^2$)")
        ax_i.set_ylabel("I_A, scanned (1e14 W/cm$^2$)")
        ax_i.set_title(f"{len(rungs)} rungs, {n_done} points", fontsize=9)
        ax_i.legend(fontsize=7, loc="upper left")
        ax_i.grid(alpha=0.3)

        total = n_done * t_point_s
        h, m, sec = int(total // 3600), int(total % 3600 // 60), int(total % 60)
        ax_t.set_xlabel("Point")
        ax_t.set_ylabel("Cumulative time (min)")
        ax_t.set_title(f"Planned minimum: {h}h {m}min {sec}s ({t_point_s:.2f} s/point)", fontsize=9)
        ax_t.grid(alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class IntensityLadderScanTab(GridScanTab):
    """beta_a scan on a common intensity grid over a waveplate ladder (focus A)."""

    _log_source = "IntensityLadderScan"

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)
        main.addWidget(self._create_laser_group())
        main.addWidget(self._create_ladder_group())
        main.addWidget(self._create_detectors_group())
        main.addWidget(self._create_parameters_group())
        main.addLayout(self._create_controls_row())

        self._plan_window: LadderPlanWindow | None = None
        self._settle_sb.valueChanged.connect(self._update_preview)
        self._cam_tbl.itemChanged.connect(self._update_preview)
        self._cam_tbl.model().rowsInserted.connect(self._update_preview)
        self._cam_tbl.model().rowsRemoved.connect(self._update_preview)

        refresh_row = QHBoxLayout()
        btn_plan = QPushButton("Plan Plot")
        btn_plan.clicked.connect(self._show_plan_window)
        refresh_row.addWidget(btn_plan)
        btn_go_start = QPushButton("Go to start (WP min, beta_a max)")
        btn_go_start.clicked.connect(lambda: self._go_to_extreme(start=True))
        refresh_row.addWidget(btn_go_start)
        btn_go_end = QPushButton("Go to end (WP max, beta_a min)")
        btn_go_end.clicked.connect(lambda: self._go_to_extreme(start=False))
        refresh_row.addWidget(btn_go_end)
        refresh_row.addStretch(1)
        btn_refresh = QPushButton("Refresh Devices")
        btn_refresh.clicked.connect(self._refresh_devices)
        refresh_row.addWidget(btn_refresh)
        main.addLayout(refresh_row)

        self._update_preview()

    def _line(self, text: str) -> QLineEdit:
        le = QLineEdit(text)
        le.textChanged.connect(self._update_preview)
        return le

    def _create_laser_group(self) -> QGroupBox:
        group = QGroupBox("Laser and focus A")
        grid = QGridLayout(group)

        self._p_max_le = self._line("2.90")
        self._rep_le = self._line("8")
        self._tau_le = self._line("220")
        self._trans_le = self._line(f"{DEFAULT_TRANSMISSION}")
        self._eff_le = self._line("0.8")
        self._alpha_le = self._line("0.5")
        self._w_focus_le = self._line("20")

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

    def _create_ladder_group(self) -> QGroupBox:
        group = QGroupBox("Intensity ladder (intensities in 1e14 W/cm²)")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self._i_start_le = self._line("0.5")
        self._i_end_le = self._line("2.0")
        self._di_le = self._line("0.05")
        self._beta_max_le = self._line("0.35")
        self._n_shared_le = self._line("1")
        for label, le in [
            ("A: I start", self._i_start_le),
            ("I end", self._i_end_le),
            ("I step", self._di_le),
            ("beta_max", self._beta_max_le),
            ("Shared points", self._n_shared_le),
        ]:
            row.addWidget(QLabel(label))
            row.addWidget(le)
        layout.addLayout(row)

        self._ladder_tbl = QTableWidget(0, 8)
        self._ladder_tbl.setHorizontalHeaderLabels(
            ["Rung", "I_ref", "WP fraction", "WP power (W)", "beta_a range", "I_B (ref)", "Points", "Shared with next"]
        )
        self._ladder_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._ladder_tbl)

        self._ladder_label = QLabel("")
        self._ladder_label.setWordWrap(True)
        layout.addWidget(self._ladder_label)
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
    # Ladder computation and preview
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

    def _ladder(self) -> tuple[dict, dict, list[Rung]]:
        beam = self._beam()
        try:
            ladder = {
                "I_start": float(self._i_start_le.text()) * INTENSITY_UNIT,
                "I_end": float(self._i_end_le.text()) * INTENSITY_UNIT,
                "dI": float(self._di_le.text()) * INTENSITY_UNIT,
                "beta_max": float(self._beta_max_le.text()),
                "n_shared": int(self._n_shared_le.text()),
            }
        except ValueError:
            raise ValueError("Invalid ladder value.")
        rungs = build_ladder(I_focus_max=beam["I_A_max"], **ladder)
        return beam, ladder, rungs

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

    def _update_preview(self) -> None:
        if not hasattr(self, "_plan_window"):
            return  # still building the UI
        self._plan = None
        self._update_ladder_table()
        self._refresh_plan_window()

    def _show_plan_window(self) -> None:
        if self._plan_window is None:
            self._plan_window = LadderPlanWindow(self)
        self._refresh_plan_window()
        self._plan_window.show()
        self._plan_window.raise_()

    def _refresh_plan_window(self) -> None:
        if self._plan_window is None:
            return
        if self._plan is None:
            self._plan_window.show_message(self._ladder_label.text() or self._imax_label.text())
        else:
            self._plan_window.update_plan(*self._plan, self._time_per_point_s())

    def _time_per_point_s(self) -> float:
        """Settle + detector acquisition per point; lenient, for the live plan only."""
        t = float(self._settle_sb.value())
        for r in range(self._cam_tbl.rowCount()):
            try:
                key = self._cam_tbl.item(r, 0).text().strip()
                p1 = float(self._cam_tbl.item(r, 1).text())
                avg = int(float(self._cam_tbl.item(r, 3).text()))
            except (AttributeError, ValueError):
                continue
            t += _detector_time_estimate_s(key, (p1, avg))
        return t

    def _update_ladder_table(self) -> None:
        self._ladder_tbl.setRowCount(0)
        try:
            beam = self._beam()
            self._imax_label.setText(
                f"Max intensity (waveplate at 1, beta = 0): "
                f"A {beam['I_A_max'] / INTENSITY_UNIT:.4g}, B {beam['I_B_max'] / INTENSITY_UNIT:.4g} "
                f"× 1e14 W/cm²"
            )
        except ValueError as e:
            self._imax_label.setText(str(e))
            self._ladder_label.setText("")
            return
        try:
            _, _, rungs = self._ladder()
        except ValueError as e:
            self._ladder_label.setText(str(e))
            return

        p_max = beam["p_max_W"]
        warnings = []
        beta_b = self._slm_beta_b()
        if beta_b is None:
            warnings.append("SLM not available: I_B shown for beta_b = 0")
            beta_b = 0.0
        for i, rung in enumerate(rungs):
            shared = len(rung.grid_indices & rungs[i + 1].grid_indices) if i + 1 < len(rungs) else None
            betas = [b for _, _, b in rung.points]
            beta_txt = f"{min(betas):.4f} – {max(betas):.4f}" if betas else "–"
            cells = [
                str(rung.k),
                f"{rung.I_ref / INTENSITY_UNIT:.4g}",
                f"{rung.wp_fraction:.4f}",
                f"{rung.wp_fraction * p_max:.4g}",
                beta_txt,
                f"{reference_intensity(rung, beam['I_B_max'], beta_b) / INTENSITY_UNIT:.4g}",
                str(len(rung.points)),
                "–" if shared is None else str(shared),
            ]
            self._ladder_tbl.insertRow(i)
            for c, text in enumerate(cells):
                self._ladder_tbl.setItem(i, c, QTableWidgetItem(text))
            if not rung.points:
                warnings.append(f"rung {rung.k} has no grid point")

        total = sum(len(r.points) for r in rungs)
        unique = len(set().union(*(r.grid_indices for r in rungs)))
        text = f"{len(rungs)} rungs, {total} points ({unique} distinct A intensities, {total - unique} repeats for stitching)."
        if not self._wp_calibrated():
            warnings.append(f"waveplate {self._wp_combo.currentText() or '(none)'} has no calibration")
        if warnings:
            text += "\nWarning: " + "; ".join(warnings) + "."
        self._ladder_label.setText(text)
        self._plan = (rungs, beam["I_B_max"], beta_b)

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
        beam, ladder, rungs = self._ladder()
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

        points, extras = ladder_to_points(rungs, beam["I_B_max"], beta_b)
        if not points:
            raise ValueError("The ladder has no points.")

        ax_a = f"slm:{SLM_CLASS_NAME}:{SLM_FIELD_BETA_A}"
        axes = [(wp_key, []), (ax_a, [])]  # same order as ladder_to_points
        axes_meta = {
            wp_key: {"pm": True, "max_value_W": beam["p_max_W"]},
            ax_a: {"param": SLM_FIELD_BETA_A, "screen": screen},
        }

        notes = [
            f"Intensity ladder scan ({SLM_CLASS_NAME}): only beta_a scanned, focus B is the reference",
            f"  P_max={beam['p_max_W']:.6g} W | rep={beam['rep_rate_kHz']:.6g} kHz | "
            f"tau={beam['tau_fs']:.6g} fs | T={beam['transmission']:.6g} | eff={beam['efficiency']:.6g} | "
            f"alpha={beam['alpha']:.6g} | w_focus={beam['w_focus_um']:.6g} um",
            f"  I_A_max={beam['I_A_max']:.6e} | I_B_max={beam['I_B_max']:.6e} W/cm2 (waveplate at 1, beta=0)",
            f"  A: I_start={ladder['I_start']:.6e} | I_end={ladder['I_end']:.6e} | dI={ladder['dI']:.6e} W/cm2 | "
            f"beta_max={ladder['beta_max']:.6g} | shared points={ladder['n_shared']}",
            f"  B: beta_b={beta_b:.6g} ({beta_b_note}, not changed by the scan)",
        ]
        notes += [
            f"  rung {rg.k}: I_ref={rg.I_ref:.6e} W/cm2 | wp_fraction={rg.wp_fraction:.6f} | "
            f"I_B={reference_intensity(rg, beam['I_B_max'], beta_b):.6e} W/cm2 | {len(rg.points)} points"
            for rg in rungs
        ]

        return {
            "axes": axes,
            "axes_meta": axes_meta,
            "points": points,
            "point_extras": extras,
            "header_notes": notes,
            "alpha": beam["alpha"],
            "rungs": rungs,
            **self._collect_common_params(),
        }

    def _points_summary(self, p: dict) -> str:
        rungs = p["rungs"]
        return f"Ladder: {len(rungs)} rungs, " + ", ".join(str(len(r.points)) for r in rungs) + " points per rung"

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
        """Move to the lowest (start) or highest (end) A intensity of the ladder, to check contrast.

        start: last rung (lowest waveplate fraction) at its largest beta_a.
        end: rung 0 (highest waveplate fraction) at beta_a = 0.
        """
        if not self._start_btn.isEnabled():
            QMessageBox.warning(self, "Scan running", "Wait for the scan to finish.")
            return
        label = "start" if start else "end"
        try:
            beam, _, rungs = self._ladder()
            rung = rungs[-1] if start else rungs[0]
            if not rung.points:
                raise ValueError(f"Rung {rung.k} has no grid point.")
            _, I_n, beta_a = rung.points[-1] if start else rung.points[0]

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

            angle = power_to_angle(rung.wp_fraction, float(amp_off[1]))
            stage.move_to(angle, blocking=False)
        except Exception as e:
            QMessageBox.critical(self, f"Go to {label}", str(e))
            return

        self._log_message(
            f"Go to {label}: rung {rung.k}, WP fraction {rung.wp_fraction:.4f} "
            f"({rung.wp_fraction * beam['p_max_W']:.4g} W, {angle:.2f} deg), beta_a {beta_a:.4f}, "
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

        wp_key = p["axes"][0][0]
        REGISTRY.register(_reg_key_maxvalue(_wp_index_from_stage_key(wp_key)), float(p["axes_meta"][wp_key]["max_value_W"]))

        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(background=False, existing=None)
        self._log_message("Intensity ladder scan started…")
