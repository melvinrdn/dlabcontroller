from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QMessageBox,
    QComboBox,
    QInputDialog,
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import ROOT
from dlab.hardware.wrappers.smaract_controller import SmarActAxis, SmarActController
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.config_utils import cfg_get
from dlab.utils.position_poller import PositionPoller
from dlab.utils.yaml_utils import read_yaml, write_yaml


def _config_path():
    return ROOT / "config" / "config.yaml"


# The controller works in meters; this window displays and accepts micrometers.
_UM = 1e-6
_STEPS_UM = (-100, -10, -1, 1, 10, 100)


class SmarActAxisRow(QWidget):
    """Single row controlling one axis of a shared SmarAct MCS2 connection."""

    def __init__(
        self,
        controller: SmarActController,
        axis: int,
        has_sensor: bool,
        log_panel: LogPanel | None = None,
        parent: QWidget | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._axis = axis
        self._has_sensor = has_sensor
        self._log = log_panel
        self._label_text = label or f"Axis {self._axis}:"
        # Last commanded target (m), so rapid step clicks accumulate while moving.
        self._last_target: Optional[float] = None

        self._init_ui()
        self._poller = PositionPoller(
            get_position=self._get_position_um,
            target_edit=self._current_edit,
            log=self._log_message,
            fmt="{:.3f}",
            parent=self,
        )
        if has_sensor:
            self._poller.start()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setSpacing(5)

        label = QLabel(self._label_text)
        label.setFixedWidth(140)
        layout.addWidget(label)

        self._home_btn = QPushButton("Home")
        self._home_btn.clicked.connect(self._on_home)
        layout.addWidget(self._home_btn)

        self._target_edit = QLineEdit()
        self._target_edit.setPlaceholderText("Position (µm)")
        self._target_edit.setValidator(QDoubleValidator(-1e9, 1e9, 3, self))
        self._target_edit.setFixedWidth(120)
        layout.addWidget(self._target_edit)

        self._move_btn = QPushButton("Move To")
        self._move_btn.clicked.connect(self._on_move)
        layout.addWidget(self._move_btn)

        self._current_edit = QLineEdit()
        self._current_edit.setPlaceholderText("Current (µm)")
        self._current_edit.setFixedWidth(120)
        self._current_edit.setReadOnly(True)
        layout.addWidget(self._current_edit)
        layout.addWidget(QLabel("µm"))

        self._step_btns: list[QPushButton] = []
        for step in _STEPS_UM:
            btn = QPushButton(f"{step:+d}")
            btn.setFixedWidth(45)
            btn.setToolTip(f"Step {step:+d} µm")
            btn.clicked.connect(lambda _=False, s=step: self._on_step(s))
            layout.addWidget(btn)
            self._step_btns.append(btn)

        layout.addStretch(1)

        if not self._has_sensor:
            self._home_btn.setEnabled(False)
            self._target_edit.setEnabled(False)
            self._move_btn.setEnabled(False)
            for btn in self._step_btns:
                btn.setEnabled(False)
            self._current_edit.setPlaceholderText("No sensor")

    def _get_position_um(self) -> Optional[float]:
        pos = self._controller.get_position(self._axis)
        return None if pos is None else pos / _UM

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(f"Axis {self._axis}: {msg}", source="SmarAct")

    def _on_home(self) -> None:
        try:
            self._controller.home(self._axis, blocking=False)
            self._log_message("Homing…")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Axis {self._axis}: home failed: {e}")
            self._log_message(f"Home failed: {e}")

    def _on_move(self) -> None:
        t = self._target_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Please enter a target position.")
            return
        try:
            value = float(t)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid position.")
            return
        try:
            self._controller.move_to(self._axis, value * _UM, blocking=False)
            self._last_target = value * _UM
            self._log_message(f"Moving to {value:.3f} µm…")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Axis {self._axis}: move failed: {e}")
            self._log_message(f"Move failed: {e}")

    def _on_step(self, step_um: int) -> None:
        try:
            # While a move is in progress, step from the pending target rather
            # than the in-flight position, so repeated clicks add up exactly.
            if self._last_target is not None and self._controller.is_moving(self._axis):
                base = self._last_target
            else:
                base = self._controller.get_position(self._axis)
            if base is None:
                return
            target = base + step_um * _UM
            self._controller.move_to(self._axis, target, blocking=False)
            self._last_target = target
            self._log_message(f"Step {step_um:+d} µm → {target / _UM:.3f} µm")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Axis {self._axis}: step failed: {e}")
            self._log_message(f"Step failed: {e}")

    def stop_polling(self) -> None:
        self._poller.stop()


class SmarActStageWindow(QWidget):
    """Control window for a SmarAct MCS2 controller and its axes."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SmarAct Stage Control")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._controller: Optional[SmarActController] = None
        self._axis_rows: list[SmarActAxisRow] = []
        self._registry_key = "smaract:mcs2"
        self._axis_registry_keys: list[str] = []

        self._init_ui()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout(conn_group)

        default_locator = cfg_get("smaract.mcs2.locator", "")
        self._locator_edit = QLineEdit(default_locator)
        self._locator_edit.setPlaceholderText("network:sn:MCS2-...")
        conn_layout.addWidget(QLabel("Locator:"))
        conn_layout.addWidget(self._locator_edit, 1)

        self._activate_btn = QPushButton("Connect")
        self._activate_btn.clicked.connect(self._on_activate)
        conn_layout.addWidget(self._activate_btn)

        self._deactivate_btn = QPushButton("Disconnect")
        self._deactivate_btn.setEnabled(False)
        self._deactivate_btn.clicked.connect(self._on_deactivate)
        conn_layout.addWidget(self._deactivate_btn)

        main.addWidget(conn_group)

        self._axes_group = QGroupBox("Axes")
        self._axes_layout = QVBoxLayout(self._axes_group)
        main.addWidget(self._axes_group)

        positions_group = QGroupBox("Saved Positions")
        positions_row = QHBoxLayout(positions_group)
        self._saved_combo = QComboBox()
        self._saved_combo.setMinimumWidth(200)
        positions_row.addWidget(self._saved_combo, 1)

        self._goto_saved_positions_btn = QPushButton("Go To")
        self._goto_saved_positions_btn.setEnabled(False)
        self._goto_saved_positions_btn.clicked.connect(self._on_goto_saved_positions)
        positions_row.addWidget(self._goto_saved_positions_btn)

        self._save_positions_btn = QPushButton("Save Current As…")
        self._save_positions_btn.setEnabled(False)
        self._save_positions_btn.clicked.connect(self._on_save_positions)
        positions_row.addWidget(self._save_positions_btn)

        self._delete_saved_btn = QPushButton("Delete")
        self._delete_saved_btn.clicked.connect(self._on_delete_saved_position)
        positions_row.addWidget(self._delete_saved_btn)

        main.addWidget(positions_group)
        self._refresh_saved_combo()

        main.addStretch(1)

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="SmarAct")

    def _on_activate(self) -> None:
        locator = self._locator_edit.text().strip()
        if not locator:
            QMessageBox.warning(self, "Error", "Please enter a locator.")
            return

        try:
            controller = SmarActController(locator)
            controller.activate()

            for row in self._axis_rows:
                row.stop_polling()
                self._axes_layout.removeWidget(row)
                row.deleteLater()
            self._axis_rows = []

            axis_labels = [
                "Z position:",
                "X position (fwd/backward):",
                "Y position (up/down):",
            ]

            self._axis_registry_keys = []
            for axis in range(controller.naxes):
                has_sensor = controller.has_sensor(axis)
                label = axis_labels[axis] if axis < len(axis_labels) else f"Axis {axis}:"
                row = SmarActAxisRow(
                    controller,
                    axis,
                    has_sensor=has_sensor,
                    log_panel=self._log,
                    parent=self._axes_group,
                    label=label,
                )
                self._axis_rows.append(row)
                self._axes_layout.addWidget(row)

                if has_sensor:
                    axis_key = f"stage:smaract:mcs2:axis{axis}"
                    REGISTRY.register(axis_key, SmarActAxis(controller, axis))
                    self._axis_registry_keys.append(axis_key)

            self._controller = controller
            REGISTRY.register(self._registry_key, self._controller)

            self._activate_btn.setEnabled(False)
            self._locator_edit.setEnabled(False)
            self._deactivate_btn.setEnabled(True)
            self._save_positions_btn.setEnabled(True)
            self._goto_saved_positions_btn.setEnabled(True)
            self._log_message(f"Connected to {locator} ({controller.naxes} axes).")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {e}")
            self._log_message(f"Connection failed: {e}")
            self._controller = None

    def _on_deactivate(self) -> None:
        for row in self._axis_rows:
            row.stop_polling()
            self._axes_layout.removeWidget(row)
            row.deleteLater()
        self._axis_rows = []

        if self._controller is not None:
            try:
                self._controller.disable()
            finally:
                for axis_key in self._axis_registry_keys:
                    try:
                        REGISTRY.unregister(axis_key)
                    except Exception:
                        pass
                self._axis_registry_keys = []
                try:
                    REGISTRY.unregister(self._registry_key)
                except Exception:
                    pass
        self._controller = None

        self._activate_btn.setEnabled(True)
        self._locator_edit.setEnabled(True)
        self._deactivate_btn.setEnabled(False)
        self._save_positions_btn.setEnabled(False)
        self._goto_saved_positions_btn.setEnabled(False)
        self._log_message("Disconnected.")

    @staticmethod
    def _read_saved_positions() -> dict[str, dict[str, float]]:
        """Named saved positions from config: {name: {"axisN": meters}}."""
        data = read_yaml(_config_path())
        saved = (((data.get("smaract") or {}).get("mcs2") or {}).get("saved_positions")) or {}
        return {str(k): v for k, v in saved.items() if isinstance(v, dict)}

    @staticmethod
    def _write_saved_positions(saved: dict[str, dict[str, float]]) -> None:
        path = _config_path()
        data = read_yaml(path)
        node = data.get("smaract", {}) if isinstance(data.get("smaract"), dict) else {}
        mcs2_node = node.get("mcs2", {}) if isinstance(node.get("mcs2"), dict) else {}
        mcs2_node["saved_positions"] = saved
        node["mcs2"] = mcs2_node
        data["smaract"] = node
        write_yaml(path, data)

    def _refresh_saved_combo(self, select: str | None = None) -> None:
        current = select or self._saved_combo.currentText()
        self._saved_combo.clear()
        names = list(self._read_saved_positions())
        self._saved_combo.addItems(names)
        if current in names:
            self._saved_combo.setCurrentText(current)
        self._delete_saved_btn.setEnabled(bool(names))

    def _on_save_positions(self) -> None:
        if self._controller is None:
            return

        positions: dict[str, float] = {}
        for row in self._axis_rows:
            if not row._has_sensor:
                continue
            pos = self._controller.get_position(row._axis)
            if pos is not None:
                positions[f"axis{row._axis}"] = float(pos)

        if not positions:
            QMessageBox.warning(self, "Error", "No axis positions available to save.")
            return

        name, ok = QInputDialog.getText(
            self, "Save Position", "Name:", text=self._saved_combo.currentText()
        )
        name = name.strip()
        if not ok or not name:
            return

        try:
            saved = self._read_saved_positions()
            if name in saved:
                reply = QMessageBox.question(
                    self, "Overwrite", f"Overwrite saved position '{name}'?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
            saved[name] = positions
            self._write_saved_positions(saved)
            self._refresh_saved_combo(select=name)
            pretty = ", ".join(f"{k}={v / _UM:.3f} µm" for k, v in positions.items())
            self._log_message(f"Saved position '{name}': {pretty}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save position: {e}")

    def _on_delete_saved_position(self) -> None:
        name = self._saved_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Delete", f"Delete saved position '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            saved = self._read_saved_positions()
            saved.pop(name, None)
            self._write_saved_positions(saved)
            self._refresh_saved_combo()
            self._log_message(f"Deleted saved position '{name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete position: {e}")

    def _on_goto_saved_positions(self) -> None:
        if self._controller is None:
            return

        name = self._saved_combo.currentText()
        saved = self._read_saved_positions().get(name)
        if not saved:
            QMessageBox.information(self, "SmarAct", "No saved position selected.")
            return

        self._log_message(f"Going to saved position '{name}'…")
        for row in self._axis_rows:
            if not row._has_sensor:
                continue
            value = saved.get(f"axis{row._axis}")
            if value is None:
                continue
            try:
                self._controller.move_to(row._axis, float(value), blocking=False)
                row._last_target = float(value)
                self._log_message(f"Axis {row._axis}: moving to {float(value) / _UM:.3f} µm…")
            except Exception as e:
                self._log_message(f"Axis {row._axis}: failed to move to saved position: {e}")

    def closeEvent(self, event) -> None:
        if self._controller is not None:
            self._on_deactivate()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SmarActStageWindow()
    window.show()
    sys.exit(app.exec_())
