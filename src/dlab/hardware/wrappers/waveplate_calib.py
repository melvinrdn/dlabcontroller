from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFileDialog, QLineEdit
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm

from dlab.boot import ROOT, get_config
from dlab.utils.config_utils import cfg_get
from dlab.utils.paths_utils import ressources_dir
from dlab.utils.yaml_utils import read_yaml, write_yaml
from dlab.utils.log_panel import LogPanel


def _wp_calibration_path(wp_index: int) -> Path | None:
    """Get calibration file path for waveplate from config."""
    rel = cfg_get(f"waveplates.calibration_files.{wp_index}")
    if not rel:
        return None
    return (ressources_dir() / str(rel)).resolve()


def _save_wp_calibration_path(wp_index: int, path: Path) -> None:
    """Remember the calibration file for a waveplate, in config.yaml and in the loaded config."""
    try:
        rel = path.resolve().relative_to(ressources_dir()).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()

    def set_path(data: dict) -> None:
        node = data.get("waveplates") if isinstance(data.get("waveplates"), dict) else {}
        files = node.get("calibration_files") if isinstance(node.get("calibration_files"), dict) else {}
        files[str(wp_index)] = rel
        node["calibration_files"] = files
        data["waveplates"] = node

    cfg_path = ROOT / "config" / "config.yaml"
    data = read_yaml(cfg_path)
    set_path(data)
    write_yaml(cfg_path, data)
    set_path(get_config())  # so a later reload in this session uses the new file


def _load_wp_calibration_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load x, y data from text file with flexible separators."""
    xs: list[float] = []
    ys: list[float] = []

    def _try_float(tok: str) -> float | None:
        try:
            return float(tok)
        except Exception:
            return None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            for sep in (";", ",", "\t"):
                s = s.replace(sep, " ")
            parts = [p for p in s.split(" ") if p]
            if len(parts) < 2:
                continue
            x, y = _try_float(parts[0]), _try_float(parts[1])
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)

    if not xs:
        return np.array([]), np.array([])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _generate_colors(n: int) -> list:
    """Generate n distinct colors from colormap."""
    cmap = cm.get_cmap("tab10") if n <= 10 else cm.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


class WaveplateCalibWidget(QWidget):
    """Widget for waveplate calibration management."""

    def __init__(
        self,
        log_panel: LogPanel | None = None,
        calibration_changed_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Waveplate Calibration")

        self._log = log_panel
        self._calibration_changed_callback = calibration_changed_callback
        self._num_waveplates = int(cfg_get("waveplates.num_waveplates", 7))
        self._colors = _generate_colors(self._num_waveplates)

        self._calibration_params: Dict[int, Tuple[float, float]] = {}
        self._max_abs_power: Dict[int, float] = {}
        self._wp_entries: Dict[int, Dict[str, QLineEdit]] = {}

        self._init_ui()
        self._load_all_calibrations()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        options_group = QGroupBox("Calibration Options")
        options_layout = QVBoxLayout(options_group)

        for i in range(1, self._num_waveplates + 1):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"WP{i}:"), stretch=0)

            row.addWidget(QLabel("Max (norm):"), stretch=0)
            max_edit = QLineEdit("1.00")
            max_edit.setFixedWidth(70)
            max_edit.setReadOnly(True)
            row.addWidget(max_edit, stretch=0)

            row.addWidget(QLabel("Phase (deg):"), stretch=0)
            off_edit = QLineEdit("0")
            off_edit.setFixedWidth(70)
            off_edit.setReadOnly(True)
            row.addWidget(off_edit, stretch=0)

            self._wp_entries[i] = {"max": max_edit, "offset": off_edit}
            options_layout.addLayout(row)

        choose_row = QHBoxLayout()
        choose_row.addWidget(QLabel("Select WP:"), stretch=0)
        self._wp_dropdown = QComboBox()
        self._wp_dropdown.addItems([str(i) for i in range(1, self._num_waveplates + 1)])
        choose_row.addWidget(self._wp_dropdown, stretch=1)
        options_layout.addLayout(choose_row)

        update_btn = QPushButton("Update Calibration File")
        update_btn.clicked.connect(self._update_selected_calibration_file)
        options_layout.addWidget(update_btn)

        main_layout.addWidget(options_group, 1)

        self._fig = Figure(figsize=(5, 6), dpi=100)
        rows = int(np.ceil(np.sqrt(self._num_waveplates)))
        cols = int(np.ceil(self._num_waveplates / rows))
        self._axes = [self._fig.add_subplot(rows, cols, i + 1) for i in range(self._num_waveplates)]
        self._canvas = FigureCanvas(self._fig)
        main_layout.addWidget(self._canvas, 2)

    def _load_all_calibrations(self):
        """Load calibrations for all waveplates from config."""
        for i in range(1, self._num_waveplates + 1):
            self.load_waveplate_calibration(i)

    @staticmethod
    def _cos01(x_deg: np.ndarray | float, phase_deg: float) -> np.ndarray | float:
        return 0.5 * (1.0 + np.cos(2.0 * np.pi / 90.0 * (np.asarray(x_deg) - phase_deg)))

    def _fit_phase_only(self, x: np.ndarray, y01: np.ndarray) -> float:
        def f(xx, phase):
            return self._cos01(xx, phase)

        popt, _ = curve_fit(f, x, y01, p0=(0.0,))
        return float(popt[0])

    def load_waveplate_calibration(self, wp_index: int) -> bool:
        """Load calibration for a waveplate from config path."""
        p = _wp_calibration_path(wp_index)
        if not p or not p.exists():
            self._log_message(f"No calibration file for WP{wp_index}.")
            return False

        self._open_calibration_file(wp_index, p)
        return True

    def _open_calibration_file(self, wp_index: int, filepath: Path) -> bool:
        try:
            angles, powers = _load_wp_calibration_file(filepath)
            if angles.size == 0:
                raise ValueError("Empty calibration file")

            pmax = float(np.nanmax(powers))
            if not np.isfinite(pmax) or pmax <= 0:
                raise ValueError("Invalid max power in calibration file")

            y01 = np.clip(powers / pmax, 0.0, 1.0)
            phase = self._fit_phase_only(angles, y01)

            self._calibration_params[wp_index] = (1.0, phase)
            self._max_abs_power[wp_index] = pmax

            self._wp_entries[wp_index]["max"].setText("1.00")
            self._wp_entries[wp_index]["offset"].setText(f"{phase:.2f}")

            ax = self._axes[wp_index - 1]
            ax.clear()
            color = self._colors[wp_index - 1]
            ax.plot(angles, y01, marker="o", linestyle="None", color=color, label="data (norm)")
            xs = np.linspace(0, 360, 721)
            ax.plot(xs, self._cos01(xs, phase), color=color, label="fit (norm)")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Relative power (0..1)")
            ax.legend(loc="best")

            self._log_message(f"WP{wp_index} loaded (max={pmax:.3g} W)")

            if self._calibration_changed_callback:
                self._calibration_changed_callback(wp_index, self._calibration_params[wp_index])

            try:
                from dlab.core.device_registry import REGISTRY
                REGISTRY.register(f"waveplate:calib:{wp_index}", (1.0, phase))
                REGISTRY.register(f"waveplate:calib_path:{wp_index}", filepath.as_posix())
                REGISTRY.register(f"waveplate:max:{wp_index}", pmax)
            except Exception:
                pass

            self._canvas.draw_idle()
            return True

        except Exception as e:
            self._log_message(f"Failed to load calibration for WP{wp_index}: {e}")
            return False

    def _update_selected_calibration_file(self):
        wp_index = int(self._wp_dropdown.currentText())
        start_dir = _wp_calibration_path(wp_index)
        if start_dir:
            start_dir = start_dir.parent
        else:
            start_dir = ressources_dir()

        fname, _ = QFileDialog.getOpenFileName(
            self,
            f"Select calibration file for WP{wp_index}",
            str(start_dir),
            "Data Files (*.twt *.txt *.csv);;All Files (*)",
        )
        if not fname:
            return

        if not self._open_calibration_file(wp_index, Path(fname)):
            return
        try:
            _save_wp_calibration_path(wp_index, Path(fname))
            self._log_message(f"WP{wp_index} calibration file saved as default: {fname}")
        except Exception as e:
            self._log_message(f"WP{wp_index}: could not save calibration file to config: {e}")

    def _log_message(self, message: str):
        if self._log:
            self._log.log(message, source="WaveplateCalib")