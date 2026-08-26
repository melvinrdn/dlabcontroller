from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QMessageBox,
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import ROOT, get_config
from dlab.hardware.wrappers.smaract_controller import SmarActAxis, SmarActController
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.config_utils import cfg_get
from dlab.utils.yaml_utils import read_yaml, write_yaml


def _config_path():
    return ROOT / "config" / "config.yaml"


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

        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._update_position)

        self._init_ui()
        if has_sensor:
            self._poll.start()

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
        self._target_edit.setPlaceholderText("Position (m)")
        self._target_edit.setValidator(QDoubleValidator(-1e3, 1e3, 9, self))
        self._target_edit.setFixedWidth(120)
        layout.addWidget(self._target_edit)

        self._move_btn = QPushButton("Move To")
        self._move_btn.clicked.connect(self._on_move)
        layout.addWidget(self._move_btn)

        self._current_edit = QLineEdit()
        self._current_edit.setPlaceholderText("Current")
        self._current_edit.setFixedWidth(120)
        self._current_edit.setReadOnly(True)
        layout.addWidget(self._current_edit)

        layout.addStretch(1)

        if not self._has_sensor:
            self._home_btn.setEnabled(False)
            self._target_edit.setEnabled(False)
            self._move_btn.setEnabled(False)
            self._current_edit.setPlaceholderText("No sensor")

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(f"Axis {self._axis}: {msg}", source="SmarAct")

    def _update_position(self) -> None:
        try:
            pos = self._controller.get_position(self._axis)
            if pos is not None:
                self._current_edit.setText(f"{pos:.3e}")
        except Exception as e:
            self._poll.stop()
            self._log_message(f"Position read failed: {e}")

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
            self._controller.move_to(self._axis, value, blocking=False)
            self._log_message(f"Moving to {value:.3e} m…")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Axis {self._axis}: move failed: {e}")
            self._log_message(f"Move failed: {e}")

    def stop_polling(self) -> None:
        self._poll.stop()


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

        positions_row = QHBoxLayout()
        self._save_positions_btn = QPushButton("Save Positions")
        self._save_positions_btn.setEnabled(False)
        self._save_positions_btn.clicked.connect(self._on_save_positions)
        positions_row.addWidget(self._save_positions_btn)

        self._goto_saved_positions_btn = QPushButton("Go to Saved Positions")
        self._goto_saved_positions_btn.setEnabled(False)
        self._goto_saved_positions_btn.clicked.connect(self._on_goto_saved_positions)
        positions_row.addWidget(self._goto_saved_positions_btn)

        positions_row.addStretch(1)
        main.addLayout(positions_row)

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

        path = _config_path()
        data = read_yaml(path)
        node = data.get("smaract", {}) if isinstance(data.get("smaract"), dict) else {}
        mcs2_node = node.get("mcs2", {}) if isinstance(node.get("mcs2"), dict) else {}
        mcs2_node["saved_positions"] = positions
        node["mcs2"] = mcs2_node
        data["smaract"] = node

        try:
            write_yaml(path, data)
            self._log_message(f"Saved positions: {positions}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save positions: {e}")

    def _on_goto_saved_positions(self) -> None:
        if self._controller is None:
            return

        data = read_yaml(_config_path())
        saved = (((data.get("smaract") or {}).get("mcs2") or {}).get("saved_positions")) or {}
        if not saved:
            QMessageBox.information(self, "SmarAct", "No saved positions found.")
            return

        for row in self._axis_rows:
            if not row._has_sensor:
                continue
            value = saved.get(f"axis{row._axis}")
            if value is None:
                continue
            try:
                self._controller.move_to(row._axis, float(value), blocking=False)
                self._log_message(f"Axis {row._axis}: moving to saved position {float(value):.3e} m…")
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
