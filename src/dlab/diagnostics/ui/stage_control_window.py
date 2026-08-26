from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QTabWidget,
    QMainWindow,
    QCheckBox,
    QMessageBox,
)
from PyQt5.QtGui import QIntValidator

from dlab.boot import get_config
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.config_utils import cfg_get
from dlab.utils.position_poller import PositionPoller

from dlab.hardware.wrappers.thorlabs_controller import ThorlabsController


NUM_WAVEPLATES = int(cfg_get("waveplates.num_waveplates", 7))


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _wp_index_from_stage_number(stage_number: int) -> int:
    """Convert 0-based stage number to 1-based waveplate index."""
    return stage_number + 1


def _reg_key_powermode(wp_index: int) -> str:
    """Registry key for waveplate power mode flag."""
    return f"waveplate:powermode:{wp_index}"


def _reg_key_calib(wp_index: int) -> str:
    """Registry key for waveplate calibration data."""
    return f"waveplate:calib:{wp_index}"


def power_to_angle(power_fraction: float, _amp_unused: float, phase_deg: float) -> float:
    """Convert power fraction (0-1) to waveplate angle using calibration phase."""
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0


def load_default_ids() -> dict[int, str]:
    """Load default motor IDs from configuration."""
    cfg = get_config() or {}
    defaults = {i: "00000000" for i in range(10)}
    try:
        y = (cfg.get("stages", {}) or {}).get("default_ids", {}) or {}
        for k, v in y.items():
            try:
                defaults[int(k)] = str(v)
            except Exception:
                pass
    except Exception:
        pass
    return defaults


# -----------------------------------------------------------------------------
# StageRow
# -----------------------------------------------------------------------------


class StageRow(QWidget):
    """Single row controlling one Thorlabs rotation/translation stage."""

    def __init__(
        self,
        stage_number: int,
        description: str = "",
        log_panel: LogPanel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_number = stage_number
        self.is_waveplate = stage_number < NUM_WAVEPLATES
        self.controller: ThorlabsController | None = None
        self._log = log_panel

        self.amplitude = 1.0
        self.offset = 0.0

        self._init_ui(description)

        self._poller = PositionPoller(
            get_position=lambda: self.controller.get_position() if self.controller else None,
            target_edit=self._current_edit,
            log=self._log_message,
            parent=self,
        )

    def _init_ui(self, description: str) -> None:
        layout = QHBoxLayout(self)
        layout.setSpacing(5)

        # Labels
        title = f"Waveplate {self.stage_number + 1}:" if self.is_waveplate else f"Stage {self.stage_number + 1}:"
        self._stage_label = QLabel(title)
        self._stage_label.setFixedWidth(120)
        layout.addWidget(self._stage_label)

        self._desc_label = QLabel(description)
        self._desc_label.setFixedWidth(120)
        layout.addWidget(self._desc_label)

        # Motor ID
        self._motor_id_edit = QLineEdit()
        self._motor_id_edit.setFixedWidth(90)
        self._motor_id_edit.setPlaceholderText("Motor ID")
        self._motor_id_edit.setValidator(QIntValidator(0, 99999999, self))
        defaults = load_default_ids()
        self._motor_id_edit.setText(defaults.get(self.stage_number, "00000000"))
        layout.addWidget(self._motor_id_edit)

        # Control buttons
        self._activate_btn = QPushButton("Activate")
        self._activate_btn.clicked.connect(self._on_activate)
        layout.addWidget(self._activate_btn)

        self._home_on_activate_checkbox = QCheckBox("Home on Activate")
        self._home_on_activate_checkbox.setChecked(False)
        layout.addWidget(self._home_on_activate_checkbox)

        self._home_btn = QPushButton("Home")
        self._home_btn.clicked.connect(self._on_home)
        self._home_btn.setEnabled(False)
        layout.addWidget(self._home_btn)

        self._ident_btn = QPushButton("Identify")
        self._ident_btn.clicked.connect(self._on_identify)
        self._ident_btn.setEnabled(False)
        layout.addWidget(self._ident_btn)

        # Target input
        self._target_edit = QLineEdit()
        self._target_edit.setFixedWidth(100)
        layout.addWidget(self._target_edit)

        # Power mode checkbox (waveplates only)
        self._power_mode_checkbox = QCheckBox("Power Mode")
        if self.is_waveplate:
            wp_idx = _wp_index_from_stage_number(self.stage_number)
            pm = REGISTRY.get(_reg_key_powermode(wp_idx))
            if isinstance(pm, bool):
                self._power_mode_checkbox.setChecked(pm)
            self._power_mode_checkbox.toggled.connect(
                lambda chk, wpi=wp_idx: REGISTRY.register(_reg_key_powermode(wpi), bool(chk))
            )
        else:
            self._power_mode_checkbox.setChecked(False)
            self._power_mode_checkbox.setEnabled(False)
            self._power_mode_checkbox.setVisible(False)
        layout.addWidget(self._power_mode_checkbox)

        self._move_btn = QPushButton("Move To")
        self._move_btn.clicked.connect(self._on_move)
        self._move_btn.setEnabled(False)
        layout.addWidget(self._move_btn)

        # Current position display
        self._current_edit = QLineEdit()
        self._current_edit.setPlaceholderText("Current")
        self._current_edit.setFixedWidth(100)
        self._current_edit.setReadOnly(True)
        layout.addWidget(self._current_edit)

        layout.addStretch(1)

        self._refresh_target_placeholder()
        self._power_mode_checkbox.toggled.connect(lambda _: self._refresh_target_placeholder())

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        full_msg = f"Stage {self.stage_number + 1}: {msg}"
        if self._log:
            self._log.log(full_msg, source="Thorlabs")

    # -------------------------------------------------------------------------
    # UI helpers
    # -------------------------------------------------------------------------

    def _refresh_target_placeholder(self) -> None:
        if not self.is_waveplate:
            self._target_edit.setPlaceholderText("Position")
            self._target_edit.setToolTip("")
            return
        if self._power_mode_checkbox.isChecked():
            self._target_edit.setPlaceholderText("Power fraction (0..1)")
            self._target_edit.setToolTip("Enter fraction of max power; converted to angle via calibration.")
        else:
            self._target_edit.setPlaceholderText("Angle (deg)")
            self._target_edit.setToolTip("Enter target angle in degrees.")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._home_btn.setEnabled(enabled)
        self._ident_btn.setEnabled(enabled)
        self._move_btn.setEnabled(enabled)

    # -------------------------------------------------------------------------
    # Stage control
    # -------------------------------------------------------------------------

    def _on_activate(self) -> None:
        from dlab.hardware.wrappers.thorlabs_controller import ThorlabsController

        txt = self._motor_id_edit.text().strip()
        if not txt:
            QMessageBox.warning(self, "Error", "Please enter a motor ID.")
            return
        try:
            motor_id = int(txt)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid motor ID.")
            return

        try:
            self.controller = ThorlabsController(motor_id)
            self.controller.activate(homing=self._home_on_activate_checkbox.isChecked())

            self._activate_btn.setEnabled(False)
            self._motor_id_edit.setEnabled(False)
            self._set_controls_enabled(True)
            self._stage_label.setStyleSheet("background-color: lightgreen;")
            self._log_message("Activated.")

            # Register in device registry
            key_ui = f"stage:{self.stage_number + 1}"
            key_ser = f"stage:serial:{motor_id}"
            for k in (key_ui, key_ser):
                prev = REGISTRY.get(k)
                if prev and prev is not self.controller:
                    self._log_message(f"Registry key '{k}' already in use. Replacing.")
            REGISTRY.register(key_ui, self.controller)
            REGISTRY.register(key_ser, self.controller)

            # Load waveplate calibration if applicable
            if self.is_waveplate:
                wp_idx = _wp_index_from_stage_number(self.stage_number)
                calib_ui = REGISTRY.get("ui:waveplate_calib_widget")
                if calib_ui and hasattr(calib_ui, "load_waveplate_calibration"):
                    ok = calib_ui.load_waveplate_calibration(wp_idx)
                    if ok:
                        calib = REGISTRY.get(_reg_key_calib(wp_idx))
                        if isinstance(calib, (tuple, list)) and len(calib) >= 2:
                            self.amplitude, self.offset = float(calib[0]), float(calib[1])
                            self._log_message(f"WP{wp_idx} calibration loaded (phase={self.offset:.2f}°).")
                    else:
                        self._log_message(f"No calibration found for WP{wp_idx}.")

            self._poller.start()
            self._refresh_target_placeholder()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate stage: {e}")
            self._log_message(f"Activation failed: {e}")
            self.controller = None
            self._stage_label.setStyleSheet("")
            self._poller.stop()

    def _on_home(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self.controller.home(blocking=False)
            self._log_message("Homing…")
            self._poller.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to home stage: {e}")
            self._log_message(f"Home failed: {e}")

    def _on_identify(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self.controller.identify()
            self._log_message("Identify blink.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Identify failed: {e}")
            self._log_message(f"Identify failed: {e}")

    def _on_move(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        t = self._target_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Please enter a target value.")
            return

        try:
            # Power mode for waveplates
            if self.is_waveplate and self._power_mode_checkbox.isChecked():
                requested_frac = float(t)
                frac = float(np.clip(requested_frac, 0.0, 1.0))
                if abs(frac - requested_frac) > 1e-12:
                    self._target_edit.blockSignals(True)
                    self._target_edit.setText(f"{frac:.3f}")
                    self._target_edit.blockSignals(False)

                wp_idx = _wp_index_from_stage_number(self.stage_number)
                phase = float(self.offset)

                mv = REGISTRY.get(f"waveplate:max_value:{wp_idx}")
                if mv is None:
                    mv = REGISTRY.get(f"waveplate:max:{wp_idx}")

                angle_deg = power_to_angle(frac, 1.0, phase)
                self.controller.move_to(angle_deg, blocking=False)

                if mv is not None and np.isfinite(float(mv)):
                    self._log_message(f"Moving to {angle_deg:.3f}° (fraction={frac:.3f}, ~{frac * float(mv):.3g} W)…")
                else:
                    self._log_message(f"Moving to {angle_deg:.3f}° (fraction={frac:.3f})…")

                self._poller.start()
                return

            # Normal angle/position mode
            value = float(t)
            if self.is_waveplate:
                value = value % 360.0
            self.controller.move_to(value, blocking=False)
            self._log_message(f"Moving to {value:.3f}{'°' if self.is_waveplate else ''} …")
            self._poller.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move stage: {e}")
            self._log_message(f"Move failed: {e}")


# -----------------------------------------------------------------------------
# ThorlabsView
# -----------------------------------------------------------------------------


class ThorlabsView(QWidget):
    """Main view containing all Thorlabs stage rows organized by group."""

    def __init__(self, log_panel: LogPanel | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thorlabs Stage Control")
        self._log = log_panel
        self._stage_rows: list[StageRow] = []
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Left panel: stage groups
        left_layout = QVBoxLayout()

        # Group 1: w vs (2w, 3w) mixing
        group1 = QGroupBox("w vs (2w,3w) mixing")
        g1_layout = QVBoxLayout(group1)
        activate_all_g1 = QPushButton("Activate All in Group")
        activate_all_g1.clicked.connect(lambda: self._activate_group([0]))
        g1_layout.addWidget(activate_all_g1)
        row1 = StageRow(0, description="w/(2w,3w)", log_panel=self._log)
        self._stage_rows.append(row1)
        g1_layout.addWidget(row1)
        left_layout.addWidget(group1)

        # Group 2: w, 2w, 3w attenuation
        group2 = QGroupBox("w, 2w, 3w attenuation")
        g2_layout = QVBoxLayout(group2)
        activate_all_g2 = QPushButton("Activate All in Group")
        activate_all_g2.clicked.connect(lambda: self._activate_group([1, 2, 3]))
        g2_layout.addWidget(activate_all_g2)
        for i, desc in enumerate(["w", "2w", "3w"], start=1):
            row = StageRow(i, description=desc, log_panel=self._log)
            self._stage_rows.append(row)
            g2_layout.addWidget(row)
        left_layout.addWidget(group2)

        # Group 3: MPC
        group3 = QGroupBox("MPC")
        g3_layout = QVBoxLayout(group3)
        activate_all_g3 = QPushButton("Activate All in Group")
        activate_all_g3.clicked.connect(lambda: self._activate_group([4, 5]))
        g3_layout.addWidget(activate_all_g3)
        for i, desc in [(4, "Before MPC"), (5, "After MPC")]:
            row = StageRow(i, description=desc, log_panel=self._log)
            self._stage_rows.append(row)
            g3_layout.addWidget(row)
        left_layout.addWidget(group3)

        # Group 4: SFG
        group4 = QGroupBox("SFG")
        g4_layout = QVBoxLayout(group4)
        activate_all_g4 = QPushButton("Activate All in Group")
        activate_all_g4.clicked.connect(lambda: self._activate_group([6]))
        g4_layout.addWidget(activate_all_g4)
        row7 = StageRow(6, description="3w/2w", log_panel=self._log)
        self._stage_rows.append(row7)
        g4_layout.addWidget(row7)
        left_layout.addWidget(group4)

        # Group 5: Translation Stages
        group5 = QGroupBox("Translation Stages")
        g5_layout = QVBoxLayout(group5)
        activate_all_g5 = QPushButton("Activate All in Group")
        activate_all_g5.clicked.connect(lambda: self._activate_group([7, 8, 9]))
        g5_layout.addWidget(activate_all_g5)
        for i, desc in [(7, "Focus"), (8, "Delay 1"), (9, "Delay 2")]:
            row = StageRow(i, description=desc, log_panel=self._log)
            self._stage_rows.append(row)
            g5_layout.addWidget(row)
        left_layout.addWidget(group5)

        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 2)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="Thorlabs")

    # -------------------------------------------------------------------------
    # Group activation
    # -------------------------------------------------------------------------

    def _activate_group(self, indices: list[int]) -> None:
        for idx in indices:
            row = self._stage_rows[idx]
            try:
                row._on_activate()
                if row.controller is not None:
                    row._stage_label.setStyleSheet("background-color: lightgreen;")
                    self._log_message(f"Stage {row.stage_number + 1} activated successfully.")
                else:
                    row._stage_label.setStyleSheet("")
            except Exception as e:
                self._log_message(f"Stage {row.stage_number + 1} activation error: {e}")
                row._stage_label.setStyleSheet("background-color: lightcoral;")

    # -------------------------------------------------------------------------
    # Calibration update
    # -------------------------------------------------------------------------

    def update_stage_calibration(self, wp_index: int, calibration: tuple[float, float]) -> None:
        """Update calibration for a specific waveplate."""
        for row in self._stage_rows:
            if row.is_waveplate and (row.stage_number + 1) == wp_index:
                row.amplitude, row.offset = calibration
                REGISTRY.register(_reg_key_calib(wp_index), (float(calibration[0]), float(calibration[1])))
                self._log_message(f"Updated calibration for Stage {row.stage_number + 1}: phase={calibration[1]:.2f}°")


# -----------------------------------------------------------------------------
# StageControlWindow
# -----------------------------------------------------------------------------


class StageControlWindow(QMainWindow):
    """Main window with tabs for all stage control interfaces."""

    from PyQt5.QtCore import pyqtSignal

    closed = pyqtSignal()

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage Control")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._init_ui()

    def _init_ui(self) -> None:
        # Lazy imports to avoid circular imports
        from dlab.diagnostics.ui.auto_waveplate_calib_window import AutoWaveplateCalibWindow
        from dlab.diagnostics.ui.grating_compressor_window import GratingCompressorWindow
        from dlab.diagnostics.ui.piezojena_window import PiezoJenaStageWindow
        from dlab.diagnostics.ui.smaract_window import SmarActStageWindow
        from dlab.hardware.wrappers.waveplate_calib import WaveplateCalibWidget

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Thorlabs tab
        self._thorlabs_view = ThorlabsView(log_panel=self._log)
        self._tabs.addTab(self._thorlabs_view, "Thorlabs Control")

        # Grating Compressor tab
        self._gc_view = GratingCompressorWindow(log_panel=self._log)
        self._tabs.addTab(self._gc_view, "Grating Compressor")

        # PiezoJena tab
        self._piezojena_view = PiezoJenaStageWindow(log_panel=self._log)
        self._tabs.addTab(self._piezojena_view, "PiezoJena")

        # SmarAct tab
        self._smaract_view = SmarActStageWindow(log_panel=self._log)
        self._tabs.addTab(self._smaract_view, "SmarAct")

        # Waveplate Calibration tab
        self._calib_widget = WaveplateCalibWidget(
            log_panel=self._log,
            calibration_changed_callback=self._update_stage_calibrations,
        )
        self._tabs.addTab(self._calib_widget, "Waveplate Calibration")
        REGISTRY.register("ui:waveplate_calib_widget", self._calib_widget)

        # Auto Waveplate Calibration tab
        self._autocalib_view = AutoWaveplateCalibWindow()
        self._tabs.addTab(self._autocalib_view, "Automatic Waveplate Calibration")

    # -------------------------------------------------------------------------
    # Calibration callback
    # -------------------------------------------------------------------------

    def _update_stage_calibrations(self, wp_index: int, calibration: tuple[float, float]) -> None:
        self._thorlabs_view.update_stage_calibration(wp_index, calibration)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Embedded tab widgets don't get their own closeEvent for free when
        # this window closes, so give the ones that hold live hardware
        # connections a chance to disconnect cleanly.
        for view in (self._thorlabs_view, self._gc_view, self._piezojena_view, self._smaract_view):
            try:
                view.close()
            except Exception:
                pass
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = StageControlWindow()
    window.show()
    sys.exit(app.exec_())