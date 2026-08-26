from __future__ import annotations

import sys

try:
    # Must run before PyQt5 is imported: once Qt's own DLLs are loaded into
    # the process, loading SmarActCTL.dll afterward fails with WinError 1114
    # (DllMain init failure), regardless of ctypes search mode.
    from dlab.hardware.wrappers.smaract_controller import preload_dll

    preload_dll()
except Exception as e:
    print(f"SmarAct DLL preload failed (SmarAct tab will be unavailable): {e}")

from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dlab.boot import ROOT, bootstrap
from dlab.utils.log_panel import LogPanel


class DlabControllerWindow(QMainWindow):
    """Main launcher window for lab instrument control."""

    def __init__(self, log_panel: LogPanel):
        super().__init__()
        self._log = log_panel
        self._windows: dict[str, QWidget | None] = {
            "andor": None,
            "avaspec_w": None,
            "avaspec_2w": None,
            "avaspec_3w": None,
            "powermeter": None,
            "stage_control": None,
            "slm": None,
            "scan": None,
            "phase_lock_2w": None,
            "phase_lock_3w": None,
            "lumenera": None,
        }
        self._daheng_windows: dict[str, QWidget] = {}
        self._camera_controls: dict[str, QSpinBox] = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("DlabControllerWindow")
        self.resize(300, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        detector_buttons = [
            ("Open Andor", self._open_andor),
            ("Open Lumenera SP402S", self._open_lumenera),
            ("Open Avaspec ω", self._open_avaspec_w),
            ("Open Avaspec 2ω", self._open_avaspec_2w),
            ("Open Avaspec 3ω", self._open_avaspec_3w),
            ("Open Powermeter", self._open_powermeter),
        ]
        stage_buttons = [
            ("Open Stage Control", self._open_stage_control),
            ("Open SLM", self._open_slm),
        ]
        other_buttons = [
            ("Open Scan Panel", self._open_scan),
            ("Open Phase Lock ω/2ω", self._open_phase_lock_2w),
            ("Open Phase Lock ω/3ω", self._open_phase_lock_3w),
        ]

        detectors_group, detectors_layout = self._build_group("Detectors", detector_buttons)
        for i, name in enumerate(
            ["DahengCam_1", "DahengCam_2", "DahengCam_3"], start=1
        ):
            self._add_daheng_control(detectors_layout, name, default_index=i)
        main_layout.addWidget(detectors_group)

        stages_group, _ = self._build_group("Stages", stage_buttons)
        main_layout.addWidget(stages_group)

        other_group, _ = self._build_group("Other", other_buttons)
        main_layout.addWidget(other_group)

        main_layout.addStretch(1)

        # Log toggle button
        self._log_button = QPushButton("Show Log")
        self._log_button.clicked.connect(self._toggle_log)
        main_layout.addWidget(self._log_button)

    def _build_group(
        self, title: str, buttons: list[tuple[str, object]]
    ) -> tuple[QGroupBox, QVBoxLayout]:
        """Build a QGroupBox containing one button per (label, callback) pair."""
        group = QGroupBox(title)
        layout = QVBoxLayout()
        for label, callback in buttons:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            layout.addWidget(btn)
        group.setLayout(layout)
        return group, layout

    def _add_daheng_control(self, layout: QVBoxLayout, name: str, default_index: int):
        """Add a Daheng camera control group with index spinbox."""
        box = QGroupBox(name)
        h_layout = QHBoxLayout()

        spinbox = QSpinBox()
        spinbox.setRange(1, 5)
        spinbox.setValue(default_index)

        h_layout.addWidget(QLabel("Index:"))
        h_layout.addWidget(spinbox)

        button = QPushButton("Open")
        button.clicked.connect(
            lambda _, n=name, sb=spinbox: self._open_daheng(n, sb.value())
        )
        h_layout.addWidget(button)

        box.setLayout(h_layout)
        layout.addWidget(box)
        self._camera_controls[name] = spinbox

    def _toggle_log(self):
        if self._log.isVisible():
            self._log.hide()
            self._log_button.setText("Show Log")
        else:
            self._log.show()
            self._log_button.setText("Hide Log")

    # -------------------------------------------------------------------------
    # Generic window management
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Close every open sub-window so their hardware disconnects cleanly."""
        for win in list(self._windows.values()) + list(self._daheng_windows.values()):
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass
        super().closeEvent(event)

    def _open_window(
        self,
        key: str,
        window_class: type,
        display_name: str,
        *args,
        use_destroyed_signal: bool = False,
        **kwargs,
    ):
        """
        Open a window if not already open, or bring it to front.

        Args:
            key: Key in self._windows dict
            window_class: The window class to instantiate
            display_name: Human-readable name for logging
            use_destroyed_signal: Use 'destroyed' signal instead of 'closed'
            *args, **kwargs: Passed to window_class constructor
        """
        win = self._windows[key]

        if win is None:
            win = window_class(*args, **kwargs)
            signal = win.destroyed if use_destroyed_signal else win.closed
            signal.connect(
                lambda *a, k=key, name=display_name: self._on_window_closed(k, name)
            )
            self._windows[key] = win
            self._log.log("Window opened.", source=display_name)

        win.show()
        win.raise_()
        win.activateWindow()

    def _on_window_closed(self, key: str, display_name: str):
        """Handle window close event."""
        self._windows[key] = None
        self._log.log("Window closed.", source=display_name)

    # -------------------------------------------------------------------------
    # Individual window openers
    # -------------------------------------------------------------------------

    def _open_andor(self):
        from dlab.diagnostics.ui.andor_live_window import AndorLiveWindow

        self._open_window("andor", AndorLiveWindow, "Andor", self._log)
        
    def _open_lumenera(self):
        from dlab.diagnostics.ui.lumenera_live_window import LumeneraLiveWindow

        self._open_window(
            "lumenera",
            LumeneraLiveWindow,
            "Lumenera SP402S",
            camera_name="SP402S",
            fixed_index=1,
            log_panel=self._log,
        )

    def _open_avaspec_w(self):
        from dlab.diagnostics.ui.avaspec_live_window import AvaspecLiveWindow

        self._open_window(
            "avaspec_w",
            AvaspecLiveWindow,
            "Avaspec ω",
            instance_id="w",
            log_panel=self._log,
        )

    def _open_avaspec_2w(self):
        from dlab.diagnostics.ui.avaspec_live_window import AvaspecLiveWindow

        self._open_window(
            "avaspec_2w",
            AvaspecLiveWindow,
            "Avaspec ω/2ω",
            instance_id="2w",
            log_panel=self._log,
        )

    def _open_avaspec_3w(self):
        from dlab.diagnostics.ui.avaspec_live_window import AvaspecLiveWindow

        self._open_window(
            "avaspec_3w",
            AvaspecLiveWindow,
            "Avaspec ω/3ω",
            instance_id="3w",
            log_panel=self._log,
        )

    def _open_slm(self):
        from dlab.diagnostics.ui.slm_window import SlmWindow

        self._open_window("slm", SlmWindow, "SLM", self._log)

    def _open_stage_control(self):
        from dlab.diagnostics.ui.stage_control_window import StageControlWindow

        self._open_window(
            "stage_control", StageControlWindow, "Stage Control", self._log
        )

    def _open_scan(self):
        from dlab.diagnostics.ui.scans.scan_window import ScanWindow

        self._open_window("scan", ScanWindow, "Scan", log_panel=self._log)

    def _open_powermeter(self):
        from dlab.diagnostics.ui.powermeter_live_window import PowermeterLiveWindow

        self._open_window("powermeter", PowermeterLiveWindow, "Powermeter", self._log)

    def _open_phase_lock_2w(self):
        from dlab.diagnostics.ui.phase_lock_window import AvaspecPhaseLockWindow

        self._open_window(
            "phase_lock_2w",
            AvaspecPhaseLockWindow,
            "Phase Lock ω/2ω",
            instance_id="2w",
            log_panel=self._log,
        )

    def _open_phase_lock_3w(self):
        from dlab.diagnostics.ui.phase_lock_window import AvaspecPhaseLockWindow

        self._open_window(
            "phase_lock_3w",
            AvaspecPhaseLockWindow,
            "Phase Lock ω/3ω",
            instance_id="3w",
            log_panel=self._log,
        )

    # -------------------------------------------------------------------------
    # Daheng camera windows (multiple instances)
    # -------------------------------------------------------------------------

    def _open_daheng(self, camera_name: str, index: int):
        from dlab.diagnostics.ui.daheng_live_window import DahengLiveWindow

        if camera_name in self._daheng_windows:
            self._log.log("Window already open.", source=camera_name)
            win = self._daheng_windows[camera_name]
            win.show()
            win.raise_()
            win.activateWindow()
            return

        win = DahengLiveWindow(
            camera_name=camera_name, fixed_index=index, log_panel=self._log
        )
        win.closed.connect(lambda name=camera_name: self._on_daheng_window_closed(name))
        self._daheng_windows[camera_name] = win
        win.show()
        win.raise_()
        win.activateWindow()
        self._log.log("Window opened.", source=camera_name)

    def _on_daheng_window_closed(self, camera_name: str):
        self._daheng_windows.pop(camera_name, None)
        self._log.log("Window closed.", source=camera_name)


def main():
    CFG = bootstrap(ROOT / "config" / "config.yaml")
    app = QApplication(sys.argv)

    log_panel = LogPanel()

    window = DlabControllerWindow(log_panel)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
