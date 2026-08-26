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
from dlab.hardware.wrappers.piezojena_controller import NV40
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.position_poller import PositionPoller


class PiezoJenaBlock(QGroupBox):
    """Self-contained control block for a single PiezoJena NV40 stage.

    Each block handles its own connection, motion controls, and registry
    entry independently.
    """

    def __init__(
        self,
        instance_id: str,
        label: str,
        default_port: str,
        log_panel: LogPanel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent)
        self._instance_id = instance_id
        self._label = label
        self._registry_key = f"stage:piezojena:{instance_id}"
        self._default_port = default_port
        self._log = log_panel
        self._stage: Optional[NV40] = None
        self._range_min, self._range_max = NV40.get_voltage_limits()

        self._init_ui()

        self._poller = PositionPoller(
            get_position=lambda: self._stage.get_position() if self._stage else None,
            target_edit=self._cur_pos_edit,
            log=self._log_message,
            parent=self,
        )

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Connection row
        conn = QHBoxLayout()
        self._port_edit = QLineEdit(self._default_port)
        self._port_edit.setFixedWidth(120)

        self._activate_btn = QPushButton("Activate")
        self._deactivate_btn = QPushButton("Deactivate")
        self._deactivate_btn.setEnabled(False)

        self._activate_btn.clicked.connect(self._on_activate)
        self._deactivate_btn.clicked.connect(self._on_deactivate)

        conn.addWidget(QLabel("Port:"))
        conn.addWidget(self._port_edit)
        conn.addStretch(1)
        conn.addWidget(self._activate_btn)
        conn.addWidget(self._deactivate_btn)
        outer.addLayout(conn)

        # Absolute move row
        row_abs = QHBoxLayout()
        self._target_abs_edit = QLineEdit()
        self._target_abs_edit.setPlaceholderText("Absolute V")
        self._target_abs_edit.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))
        self._move_abs_btn = QPushButton("Move To")
        self._move_abs_btn.setEnabled(False)
        self._move_abs_btn.clicked.connect(self._on_move_abs)
        row_abs.addWidget(self._target_abs_edit, 1)
        row_abs.addWidget(self._move_abs_btn)
        outer.addLayout(row_abs)

        # Relative move row
        row_rel = QHBoxLayout()
        self._step_rel_edit = QLineEdit("0.1")
        self._step_rel_edit.setFixedWidth(100)
        self._step_rel_edit.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))
        self._rel_minus_btn = QPushButton("− Step")
        self._rel_plus_btn = QPushButton("+ Step")
        self._rel_minus_btn.setEnabled(False)
        self._rel_plus_btn.setEnabled(False)
        self._rel_minus_btn.clicked.connect(lambda: self._on_move_rel(sign=-1))
        self._rel_plus_btn.clicked.connect(lambda: self._on_move_rel(sign=+1))
        row_rel.addWidget(QLabel("Step (V):"))
        row_rel.addWidget(self._step_rel_edit)
        row_rel.addStretch(1)
        row_rel.addWidget(self._rel_minus_btn)
        row_rel.addWidget(self._rel_plus_btn)
        outer.addLayout(row_rel)

        # Control buttons row
        row_ctrl = QHBoxLayout()
        self._force_remote_btn = QPushButton("Re-enable Remote")
        self._force_remote_btn.setEnabled(False)
        self._force_remote_btn.clicked.connect(self._on_force_remote)
        row_ctrl.addWidget(self._force_remote_btn)
        row_ctrl.addStretch(1)
        outer.addLayout(row_ctrl)

        # Position display row
        row_pos = QHBoxLayout()
        self._cur_pos_edit = QLineEdit()
        self._cur_pos_edit.setReadOnly(True)
        self._cur_pos_edit.setPlaceholderText("—")
        self._cur_pos_edit.setFixedWidth(140)
        row_pos.addWidget(QLabel("Current:"))
        row_pos.addWidget(self._cur_pos_edit)
        row_pos.addStretch(1)
        row_pos.addWidget(
            QLabel(f"Limits: {self._range_min:g} … {self._range_max:g} V")
        )
        outer.addLayout(row_pos)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source=f"PiezoJena/{self._instance_id}")

    # -------------------------------------------------------------------------
    # UI state management
    # -------------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._deactivate_btn.setEnabled(enabled)
        self._move_abs_btn.setEnabled(enabled)
        self._rel_minus_btn.setEnabled(enabled)
        self._rel_plus_btn.setEnabled(enabled)
        self._force_remote_btn.setEnabled(enabled)

    # -------------------------------------------------------------------------
    # Stage control
    # -------------------------------------------------------------------------

    def _on_activate(self) -> None:
        if self._stage:
            return

        port = self._port_edit.text().strip() or self._default_port

        try:
            stg = NV40(port=port, closed_loop=False)
            self._stage = stg
            self._log_message(f"Activated on {port}.")

            # Register in device registry
            try:
                prev = REGISTRY.get(self._registry_key)
                if prev is not None and prev is not self._stage:
                    REGISTRY.unregister(self._registry_key)
            except Exception:
                pass
            REGISTRY.register(self._registry_key, self._stage)

            self._set_controls_enabled(True)
            self._activate_btn.setEnabled(False)
            self._port_edit.setEnabled(False)

            # Read initial position
            try:
                p = self._stage.get_position()
                self._cur_pos_edit.setText(f"{p:.3f}")
            except Exception:
                self._cur_pos_edit.setText("—")

            self._poller.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"[{self._label}] Activation failed: {e}"
            )
            self._log_message(f"Activation failed: {e}")
            self._stage = None

    def _on_deactivate(self) -> None:
        try:
            if self._stage:
                try:
                    self._stage.set_remote_control(False)
                finally:
                    try:
                        REGISTRY.unregister(self._registry_key)
                    except Exception:
                        pass
            self._log_message("Deactivated.")
        finally:
            self._stage = None
            self._poller.stop()
            self._set_controls_enabled(False)
            self._activate_btn.setEnabled(True)
            self._port_edit.setEnabled(True)
            self._cur_pos_edit.setText("—")

    def _on_force_remote(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", f"[{self._label}] Stage not activated.")
            return
        try:
            self._stage.set_remote_control(True)
            self._log_message("Remote mode re-enabled.")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"[{self._label}] Remote enable failed: {e}"
            )
            self._log_message(f"Remote enable failed: {e}")

    def _on_move_abs(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", f"[{self._label}] Stage not activated.")
            return

        t = self._target_abs_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Enter a voltage.")
            return

        try:
            val = float(t)
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid number.")
            return

        val = max(self._range_min, min(self._range_max, val))

        try:
            self._stage.set_position(val)
            self._cur_pos_edit.setText(f"{val:.3f}")
            self._log_message(f"Move to {val:.3f} V.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"[{self._label}] Move failed: {e}")
            self._log_message(f"Move failed: {e}")

    def _on_move_rel(self, sign: int) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", f"[{self._label}] Stage not activated.")
            return

        try:
            step = float(self._step_rel_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid step.")
            return

        try:
            try:
                cur = self._stage.get_position()
            except Exception as e:
                self._log_message(f"Position read failed, assuming 0.0: {e}")
                cur = 0.0
            tgt = cur + sign * step
            tgt = max(self._range_min, min(self._range_max, tgt))

            self._stage.set_position(tgt)
            self._cur_pos_edit.setText(f"{tgt:.3f}")
            self._log_message(f"Move to {tgt:.3f} V.")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"[{self._label}] Relative move failed: {e}"
            )
            self._log_message(f"Relative move failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def deactivate(self) -> None:
        """Public entry point for cleanup (called by parent window)."""
        if self._stage:
            self._on_deactivate()


class PiezoJenaStageWindow(QWidget):
    """Control window hosting two PiezoJena NV40 stage."""

    _INSTANCES = (
        ("nv40_2w", "ω/2ω Stabilisation", "nv40_2w", "COMPORT"),
        ("nv40_3w", "ω/3ω Stabilisation", "nv40_3w", "COMPORT"),
    )

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PiezoJena Stages")
        self.setAttribute(Qt.WA_DeleteOnClose)

        cfg = get_config() or {}
        pcfg = cfg.get("piezojena", {}) or {}

        main = QVBoxLayout(self)
        self._blocks: list[PiezoJenaBlock] = []

        for inst_id, label, cfg_key, fallback_port in self._INSTANCES:
            sub_cfg = pcfg.get(cfg_key, {}) or {}
            port = str(sub_cfg.get("port", fallback_port))

            block = PiezoJenaBlock(
                instance_id=inst_id,
                label=label,
                default_port=port,
                log_panel=log_panel,
                parent=self,
            )
            self._blocks.append(block)
            main.addWidget(block)

        main.addStretch(1)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        for block in self._blocks:
            block.deactivate()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = PiezoJenaStageWindow()
    window.show()
    sys.exit(app.exec_())
