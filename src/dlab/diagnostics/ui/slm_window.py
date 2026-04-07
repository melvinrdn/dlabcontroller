from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from dlab.boot import ROOT, get_config
from dlab.utils.log_panel import LogPanel
from dlab.utils.yaml_utils import read_yaml, write_yaml
from dlab.utils.paths_utils import ressources_dir

from dlab.hardware.wrappers.phase_settings import PhaseSettings
from dlab.hardware.wrappers.slm_controller import SLMController
from dlab.core.device_registry import REGISTRY


def _slm_path(key: str) -> Path:
    """Get a path from slm config section."""
    cfg = get_config() or {}
    rel = (cfg.get("slm", {}) or {}).get(key)
    if not rel:
        raise KeyError(f"Missing 'slm.{key}' in config")
    return (ROOT / rel).resolve()


class SlmWindow(QtWidgets.QMainWindow):
    """
    Control window for red and green SLM devices.

    Provides phase preview, publishing to screens, and settings management.
    """
    closed = pyqtSignal()

    def __init__(self, log_panel: LogPanel | None = None):
        super().__init__()
        self._log = log_panel
        if self._log is not None:
            self._log.installEventFilter(self)

        self.setWindowTitle("SlmWindow")
        self.setMinimumSize(700, 900)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._slm_red = SLMController("red")
        self._slm_green = SLMController("green")

        self._slm_red_status = "closed"
        self._slm_green_status = "closed"

        self._init_ui()

        REGISTRY.register("slm:red:window", self)

        self._log_message("Loading the default parameters...")
        for color in ["red", "green"]:
            self._load_default_parameters(color)

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _init_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self._create_menu_bar()

        self._slm_tabs = QtWidgets.QTabWidget()
        self._slm_tabs.addTab(self._create_slm_panel("red"), "Red SLM")
        self._slm_tabs.addTab(self._create_slm_panel("green"), "Green SLM")

        main_layout.addWidget(self._slm_tabs)
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("Red SLM: closed | Green SLM: closed")

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_background_group(self, color: str) -> QtWidgets.QGroupBox:
        """Hardware flatness correction panel: load file once, toggle on/off."""
        slm: SLMController = getattr(self, f"_slm_{color}")
        group = QtWidgets.QGroupBox("Hardware background correction")
        v = QtWidgets.QVBoxLayout(group)

        h = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Load background")
        cb_on = QtWidgets.QCheckBox("On")
        cb_on.setChecked(False)
        h.addWidget(btn_load)
        h.addWidget(cb_on)
        v.addLayout(h)

        lbl = QtWidgets.QLabel("(no file loaded)")
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        def _load():
            initial = "."
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Background File", initial,
                "CSV Files (*.csv);;Image Files (*.bmp);;Text Files (*.txt);;All Files (*)",
            )
            if not filepath:
                return
            try:
                slm.load_background(filepath)
            except Exception as e:
                self._log_message(f"Failed to load {color} background: {e}")
                return
            lbl.setText(filepath)
            cb_on.setChecked(True)
            self._log_message(f"Loaded {color} background: {filepath}")

        btn_load.clicked.connect(_load)
        cb_on.toggled.connect(slm.set_background_enabled)

        setattr(self, f"_bg_btn_{color}", btn_load)
        setattr(self, f"_bg_cb_{color}", cb_on)
        setattr(self, f"_bg_lbl_{color}", lbl)
        return group

    def _create_slm_panel(self, color: str):
        panel = QtWidgets.QGroupBox(f"{color.capitalize()} SLM Interface")
        layout = QtWidgets.QVBoxLayout(panel)

        top_group = QtWidgets.QGroupBox(f"{color.capitalize()} SLM - Phase Display")
        top_layout = QtWidgets.QVBoxLayout(top_group)

        # Save/Load buttons
        btn_save = QtWidgets.QPushButton(f"Save {color} settings")
        btn_load = QtWidgets.QPushButton(f"Load {color} settings")
        btn_save.clicked.connect(lambda: self._save_settings(color))
        btn_load.clicked.connect(lambda: self._load_settings(color))
        hlayout_save = QtWidgets.QHBoxLayout()
        hlayout_save.addWidget(btn_load)
        hlayout_save.addWidget(btn_save)
        top_layout.addLayout(hlayout_save)

        # Hardware background correction
        top_layout.addWidget(self._create_background_group(color))

        # Display number
        h_layout_display = QtWidgets.QHBoxLayout()
        screens = QtWidgets.QApplication.instance().screens()
        num_screens = len(screens) if screens else 1
        spin_display = QtWidgets.QSpinBox()
        spin_display.setRange(1, num_screens)
        spin_display.setValue(2 if color == "green" and num_screens >= 2 else 1)
        setattr(self, f"_spin_{color}", spin_display)
        h_layout_display.addWidget(QtWidgets.QLabel("Display number:"))
        h_layout_display.addWidget(spin_display)
        top_layout.addLayout(h_layout_display)

        # Status label
        status_label = QtWidgets.QLabel("Status: closed")
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_label.setStyleSheet("background-color: lightgray;")
        setattr(self, f"_status_label_{color}", status_label)
        top_layout.addWidget(status_label)

        # Phase image
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        slm = getattr(self, f"_slm_{color}")
        h, w = slm.slm_size
        extent = (-w / 2, w / 2, -h / 2, h / 2)
        phase_image = ax.imshow(
            np.zeros(slm.slm_size), cmap="hsv", vmin=0, vmax=2 * np.pi, extent=extent
        )
        cbar = fig.colorbar(phase_image, ax=ax, orientation="horizontal", fraction=0.07, pad=0.03)
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", "π", "2π"])
        ax.set_xticks([])
        ax.set_yticks([])
        canvas = FigureCanvas(fig)
        setattr(self, f"_fig_{color}", fig)
        setattr(self, f"_ax_{color}", ax)
        setattr(self, f"_phase_image_{color}", phase_image)
        setattr(self, f"_canvas_{color}", canvas)
        top_layout.addWidget(canvas)

        # Phase checkboxes and tabs
        check_group = QtWidgets.QGroupBox("Phases enabled")
        check_layout = QtWidgets.QVBoxLayout(check_group)
        checkboxes = []
        for typ in PhaseSettings.types:
            cb = QtWidgets.QCheckBox(typ)
            cb.setChecked(False)
            check_layout.addWidget(cb)
            checkboxes.append(cb)

        tab_widget = QtWidgets.QTabWidget()
        phase_refs = []
        for typ in PhaseSettings.types:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            phase_ref = PhaseSettings.new_type(tab, typ)
            tab_layout.addWidget(phase_ref)
            tab_widget.addTab(tab, typ)
            phase_refs.append(phase_ref)

        setattr(self, f"_checkboxes_{color}", checkboxes)
        setattr(self, f"_phase_refs_{color}", phase_refs)
        setattr(self, f"_tab_widget_{color}", tab_widget)

        top_tab_layout = QtWidgets.QHBoxLayout()
        top_tab_layout.addWidget(check_group)
        top_tab_layout.addWidget(tab_widget)
        top_layout.addLayout(top_tab_layout)

        # Action buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        btn_preview = QtWidgets.QPushButton(f"Preview {color}")
        btn_publish = QtWidgets.QPushButton(f"Publish {color}")
        btn_close = QtWidgets.QPushButton(f"Close {color}")
        btn_preview.clicked.connect(lambda: self._get_phase(color))
        btn_publish.clicked.connect(lambda: self._open_publish_win(color))
        btn_close.clicked.connect(lambda: self._close_publish_win(color))
        bottom_layout.addWidget(btn_preview)
        bottom_layout.addWidget(btn_publish)
        bottom_layout.addWidget(btn_close)
        layout.addLayout(bottom_layout)

        layout.addWidget(top_group)
        return panel

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def _update_status_bar(self):
        msg = f"Red SLM: {self._slm_red_status} | Green SLM: {self._slm_green_status}"
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(msg)

        for color in ["red", "green"]:
            status = getattr(self, f"_slm_{color}_status")
            label = getattr(self, f"_status_label_{color}", None)
            if label:
                label.setText(f"Status: {status}")
                label.setStyleSheet(
                    "background-color: lightgreen;" if "displaying" in status
                    else "background-color: lightgray;"
                )

    # -------------------------------------------------------------------------
    # Registry
    # -------------------------------------------------------------------------

    def _update_registry_red(self, publish_types, phase_refs):
        REGISTRY.register("slm:red:active_classes", list(publish_types))
        REGISTRY.register("slm:red:widgets", phase_refs)
        params = {}
        for pref in phase_refs:
            name = pref.name_()
            try:
                params[name] = pref.save_()
            except Exception:
                params[name] = {}
        REGISTRY.register("slm:red:params", params)
        REGISTRY.register("slm:red:last_update", datetime.datetime.now().isoformat())

    # -------------------------------------------------------------------------
    # Phase operations
    # -------------------------------------------------------------------------
    def compose_levels(self):
        """Compose phase levels from active widgets for red SLM."""
        slm = self._slm_red
        active = REGISTRY.get("slm:red:active_classes") or []
        widgets = REGISTRY.get("slm:red:widgets") or []

        composed = np.zeros(slm.slm_size, dtype=np.uint16)
        for w in widgets:
            try:
                if w.name_() not in active:
                    continue
                lv = w.phase()
            except Exception:
                continue
            composed = (composed + lv) % (slm.bit_depth + 1)

        return composed

    @staticmethod
    def _levels_to_radians(levels: np.ndarray, bit_depth: int) -> np.ndarray:
        return levels.astype(np.float64) * (2.0 * np.pi / bit_depth)

    def _get_phase(self, color: str):
        self._log_message(f"Preview requested for {color} SLM.")
        slm: SLMController = getattr(self, f"_slm_{color}")
        phase_refs = getattr(self, f"_phase_refs_{color}")
        checkboxes = getattr(self, f"_checkboxes_{color}")

        total_levels = np.zeros(slm.slm_size, dtype=np.uint16)
        publish_types = []
        active_refs = []

        for cb, phase_ref in zip(checkboxes, phase_refs):
            if cb.isChecked():
                levels = phase_ref.phase()
                total_levels = (total_levels + levels) % (slm.bit_depth + 1)
                publish_types.append(phase_ref.name_())
                active_refs.append(phase_ref)

        if color == "red":
            self._update_registry_red(publish_types, active_refs)

        slm.phase = total_levels
        display_phase = self._levels_to_radians(total_levels, slm.bit_depth)

        phase_image = getattr(self, f"_phase_image_{color}")
        phase_image.set_data(display_phase)
        canvas = getattr(self, f"_canvas_{color}")
        canvas.draw()

        self._log_message(f"Preview updated for {color} SLM. Types: {', '.join(publish_types)}")
        return publish_types

    def _open_publish_win(self, color: str):
        self._log_message(f"Publish requested for {color} SLM.")
        slm: SLMController = getattr(self, f"_slm_{color}")
        spin = getattr(self, f"_spin_{color}")
        screen_num = spin.value()

        # Check screen conflict
        other_color = "green" if color == "red" else "red"
        other_status = getattr(self, f"_slm_{other_color}_status")
        if other_status != "closed" and f"Screen {screen_num}" in other_status:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Screen {screen_num} is already in use by {other_color.capitalize()} SLM."
            )
            return

        publish_types = self._get_phase(color)

        if slm.phase is None:
            QtWidgets.QMessageBox.warning(self, "Error", f"No phase computed for {color} SLM.")
            return

        slm.publish(slm.phase, screen_num)
        REGISTRY.register("slm:red:active_classes", publish_types)
        REGISTRY.register("slm:red:widgets", getattr(self, "_phase_refs_red"))
        REGISTRY.register("slm:red:controller", slm)

        setattr(self, f"_slm_{color}_status", f"displaying (Screen {screen_num})")
        self._log_message(f"Published {color} SLM phase on screen {screen_num}. Types: {', '.join(publish_types)}")
        self._update_status_bar()

    def _close_publish_win(self, color: str):
        self._log_message(f"Close requested for {color} SLM.")
        slm: SLMController = getattr(self, f"_slm_{color}")
        slm.close()
        self._log_message(f"Closed {color} SLM connection.")
        setattr(self, f"_slm_{color}_status", "closed")
        self._update_status_bar()

    # -------------------------------------------------------------------------
    # Settings I/O
    # -------------------------------------------------------------------------

    def _save_settings(self, color: str):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilter("Text Files (*.txt);;All Files (*)")
        dlg.setDirectory(str(_slm_path(f"{color}_saved_dir")))
        if not dlg.exec_():
            return

        filepath = Path(dlg.selectedFiles()[0])
        phase_refs = getattr(self, f"_phase_refs_{color}")
        checkboxes = getattr(self, f"_checkboxes_{color}")
        spin = getattr(self, f"_spin_{color}")
        slm: SLMController = getattr(self, f"_slm_{color}")

        settings = {}
        for phase_ref, cb in zip(phase_refs, checkboxes):
            settings[phase_ref.name_()] = {
                "Enabled": cb.isChecked(),
                "Params": phase_ref.save_(),
            }
        settings["__background__"] = {
            "path": slm.background_path or "",
            "enabled": bool(slm.background_enabled),
        }
        settings["screen_pos"] = spin.value()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(settings, f)

        self._update_default_path(filepath, color)

    def _load_settings(self, color: str, filepath: Path | None = None):
        if filepath is None:
            dlg = QtWidgets.QFileDialog(self)
            dlg.setNameFilter("Text Files (*.txt);;All Files (*)")
            dlg.setDirectory(str(_slm_path(f"{color}_saved_dir")))
            if not dlg.exec_():
                return
            filepath = Path(dlg.selectedFiles()[0])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            phase_refs = getattr(self, f"_phase_refs_{color}")
            checkboxes = getattr(self, f"_checkboxes_{color}")
            spin = getattr(self, f"_spin_{color}")

            refs_by_name = {pr.name_(): (pr, cb) for pr, cb in zip(phase_refs, checkboxes)}
            for key, phase_data in data.items():
                if key in ("screen_pos", "__background__"):
                    continue
                if key in refs_by_name:
                    pr, cb = refs_by_name[key]
                    pr.load_(phase_data["Params"])
                    cb.setChecked(phase_data["Enabled"])

            if "__background__" in data:
                bg_info = data["__background__"]
                slm: SLMController = getattr(self, f"_slm_{color}")
                bg_lbl = getattr(self, f"_bg_lbl_{color}")
                bg_cb = getattr(self, f"_bg_cb_{color}")
                path = bg_info.get("path", "")
                if path:
                    try:
                        slm.load_background(path)
                        bg_lbl.setText(path)
                    except Exception as e:
                        self._log_message(f"Could not reload {color} background {path}: {e}")
                bg_cb.setChecked(bool(bg_info.get("enabled", False)))

            if "screen_pos" in data:
                spin.setValue(data["screen_pos"])

            self._log_message(f"{color.capitalize()} settings loaded successfully")
        except Exception as e:
            self._log_message(f"Error loading settings for {color}: {e}")

    def _update_default_path(self, path: Path, color: str):
        try:
            rel_path = path.resolve().relative_to(ressources_dir()).as_posix()
        except ValueError:
            rel_path = str(path.resolve())

        defaults_path = _slm_path("defaults_file")
        data = read_yaml(defaults_path)
        data[f"{color}_default_path"] = rel_path

        try:
            write_yaml(defaults_path, data)
            self._log_message(f"Default {color} settings path updated to {rel_path}")
        except Exception as e:
            self._log_message(f"Error updating default settings path for {color}: {e}")

    def _load_default_parameters(self, color: str):
        try:
            defaults_path = _slm_path("defaults_file")
            overrides = read_yaml(defaults_path)
            key = f"{color}_default_path"

            if key in overrides and overrides[key]:
                rel_or_abs = overrides[key]
                if Path(rel_or_abs).is_absolute():
                    filepath = Path(rel_or_abs)
                else:
                    filepath = (ressources_dir() / rel_or_abs).resolve()
                self._log_message(f"Loading default {color} settings from override: {rel_or_abs}")
            else:
                filepath = _slm_path(f"{color}_default")
                self._log_message(f"Loading default {color} settings from config")

            self._load_settings(color, filepath)
        except Exception as e:
            self._log_message(f"Error loading default {color} settings: {e}")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, message: str):
        if self._log:
            self._log.log(message, source="SLM")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, a0):
        try:
            self._slm_red.close()
            self._slm_green.close()
        except Exception as e:
            self._log_message(f"Error during shutdown: {e}")

        self.closed.emit()
        super().closeEvent(a0)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SlmWindow()
    window.show()
    sys.exit(app.exec_())