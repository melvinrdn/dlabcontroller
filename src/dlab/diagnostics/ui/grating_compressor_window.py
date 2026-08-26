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
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import get_config
from dlab.hardware.wrappers.zaber_controller import ZaberBinaryController
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.position_poller import PositionPoller

REGISTRY_KEY = "stage:zaber:grating_compressor"


class GratingCompressorWindow(QWidget):
    """Control window for Zaber grating compressor stage."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grating Compressor")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._stage: Optional[ZaberBinaryController] = None

        # Load configuration
        cfg = get_config() or {}
        zcfg = cfg.get("zaber", {}) or {}
        self._port = str(zcfg.get("port", "COM4"))
        self._baud = int(zcfg.get("baud", 9600))
        rng = zcfg.get("range", {}) or {}
        self._range_min = float(rng.get("min", 0.0))
        self._range_max = float(rng.get("max", 50.0))

        self._init_ui()

        self._poller = PositionPoller(
            get_position=lambda: self._stage.get_position() if self._stage else None,
            target_edit=self._cur_pos_edit,
            log=self._log_message,
            parent=self,
        )

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Connection group
        conn_box = QGroupBox("Connection")
        cl = QHBoxLayout(conn_box)

        self._port_edit = QLineEdit(self._port)
        self._port_edit.setFixedWidth(120)
        self._baud_edit = QLineEdit(str(self._baud))
        self._baud_edit.setFixedWidth(80)
        self._baud_edit.setValidator(QDoubleValidator(0, 1e9, 0, self))

        self._activate_btn = QPushButton("Activate")
        self._deactivate_btn = QPushButton("Deactivate")
        self._deactivate_btn.setEnabled(False)

        self._activate_btn.clicked.connect(self._on_activate)
        self._deactivate_btn.clicked.connect(self._on_deactivate)

        cl.addWidget(QLabel("Port:"))
        cl.addWidget(self._port_edit)
        cl.addWidget(QLabel("Baud:"))
        cl.addWidget(self._baud_edit)
        cl.addStretch(1)
        cl.addWidget(self._activate_btn)
        cl.addWidget(self._deactivate_btn)
        main.addWidget(conn_box)

        # Motion group
        mot_box = QGroupBox("Motion (mm)")
        ml = QVBoxLayout(mot_box)

        # Absolute move row
        row1 = QHBoxLayout()
        self._target_abs_edit = QLineEdit()
        self._target_abs_edit.setPlaceholderText("Absolute position (mm)")
        self._move_abs_btn = QPushButton("Move To")
        self._move_abs_btn.setEnabled(False)
        self._move_abs_btn.clicked.connect(self._on_move_abs)
        row1.addWidget(self._target_abs_edit, 1)
        row1.addWidget(self._move_abs_btn)
        ml.addLayout(row1)

        # Relative move row
        row2 = QHBoxLayout()
        self._step_rel_edit = QLineEdit("0.001")
        self._step_rel_edit.setFixedWidth(100)
        self._rel_minus_btn = QPushButton("− Step")
        self._rel_plus_btn = QPushButton("+ Step")
        self._rel_minus_btn.setEnabled(False)
        self._rel_plus_btn.setEnabled(False)
        self._rel_minus_btn.clicked.connect(lambda: self._on_move_rel(sign=-1))
        self._rel_plus_btn.clicked.connect(lambda: self._on_move_rel(sign=+1))
        row2.addWidget(QLabel("Step (mm):"))
        row2.addWidget(self._step_rel_edit)
        row2.addStretch(1)
        row2.addWidget(self._rel_minus_btn)
        row2.addWidget(self._rel_plus_btn)
        ml.addLayout(row2)

        # Control buttons row
        row3 = QHBoxLayout()
        self._home_btn = QPushButton("Home")
        self._home_btn.setEnabled(False)
        self._ident_btn = QPushButton("Identify")
        self._ident_btn.setEnabled(False)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._home_btn.clicked.connect(self._on_home)
        self._ident_btn.clicked.connect(self._on_ident)
        self._stop_btn.clicked.connect(self._on_stop)
        row3.addWidget(self._home_btn)
        row3.addWidget(self._ident_btn)
        row3.addWidget(self._stop_btn)
        ml.addLayout(row3)

        # Position display row
        row4 = QHBoxLayout()
        self._cur_pos_edit = QLineEdit()
        self._cur_pos_edit.setReadOnly(True)
        self._cur_pos_edit.setPlaceholderText("—")
        self._cur_pos_edit.setFixedWidth(140)
        row4.addWidget(QLabel("Current:"))
        row4.addWidget(self._cur_pos_edit)
        row4.addStretch(1)
        row4.addWidget(QLabel(f"Limits: {self._range_min:g} … {self._range_max:g} mm"))
        ml.addLayout(row4)

        main.addWidget(mot_box)
        main.addStretch(1)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="Compressor")

    # -------------------------------------------------------------------------
    # UI state management
    # -------------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._deactivate_btn.setEnabled(enabled)
        self._move_abs_btn.setEnabled(enabled)
        self._rel_minus_btn.setEnabled(enabled)
        self._rel_plus_btn.setEnabled(enabled)
        self._home_btn.setEnabled(enabled)
        self._ident_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(enabled)

    # -------------------------------------------------------------------------
    # Stage control
    # -------------------------------------------------------------------------

    def _on_activate(self) -> None:
        if self._stage:
            return

        port = self._port_edit.text().strip() or self._port
        try:
            baud = int(float(self._baud_edit.text().strip()))
        except Exception:
            baud = self._baud

        try:
            stg = ZaberBinaryController(
                port=port,
                baud_rate=baud,
                range_min=self._range_min,
                range_max=self._range_max,
            )
            stg.activate(homing=False)
            self._stage = stg
            self._log_message(f"Activated on {port} @ {baud}.")

            # Register in device registry
            try:
                prev = REGISTRY.get(REGISTRY_KEY)
                if prev is not None and prev is not self._stage:
                    REGISTRY.unregister(REGISTRY_KEY)
            except Exception:
                pass
            REGISTRY.register(REGISTRY_KEY, self._stage)

            self._set_controls_enabled(True)
            self._activate_btn.setEnabled(False)
            self._port_edit.setEnabled(False)
            self._baud_edit.setEnabled(False)
            self._poller.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Activation failed: {e}")
            self._log_message(f"Activation failed: {e}")
            self._stage = None

    def _on_deactivate(self) -> None:
        try:
            if self._stage:
                try:
                    self._stage.disable()
                finally:
                    try:
                        REGISTRY.unregister(REGISTRY_KEY)
                    except Exception:
                        pass
            self._log_message("Deactivated.")
        finally:
            self._stage = None
            self._set_controls_enabled(False)
            self._activate_btn.setEnabled(True)
            self._port_edit.setEnabled(True)
            self._baud_edit.setEnabled(True)
            self._poller.stop()

    def _on_home(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self._stage.home(blocking=False)
            self._log_message("Homing…")
            self._poller.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Home failed: {e}")
            self._log_message(f"Home failed: {e}")

    def _on_ident(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self._stage.identify()
            self._log_message("Identify.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Identify failed: {e}")
            self._log_message(f"Identify failed: {e}")

    def _on_stop(self) -> None:
        if not self._stage:
            return
        try:
            self._stage._ensure().stop()
            self._log_message("Stop requested.")
        except Exception as e:
            self._log_message(f"Stop failed: {e}")

    def _on_move_abs(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        t = self._target_abs_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Enter a target position (mm).")
            return

        try:
            val = float(t)
            val = max(self._range_min, min(self._range_max, val))
            self._stage.move_to(val, blocking=False)
            self._log_message(f"Move to {val:.3f} mm …")
            self._poller.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
            self._log_message(f"Move failed: {e}")

    def _on_move_rel(self, sign: int) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        try:
            step = float(self._step_rel_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid step.")
            return

        try:
            cur = self._stage.get_position() or 0.0
            tgt = cur + sign * step
            tgt = max(self._range_min, min(self._range_max, tgt))
            self._stage.move_to(tgt, blocking=False)
            self._log_message(f"Move to {tgt:.3f} mm …")
            self._poller.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Relative move failed: {e}")
            self._log_message(f"Relative move failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._stage:
            self._on_deactivate()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = GratingCompressorWindow()
    window.show()
    sys.exit(app.exec_())
