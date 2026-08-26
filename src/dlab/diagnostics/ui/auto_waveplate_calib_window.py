from __future__ import annotations

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QMessageBox
)

from dlab.boot import ROOT, get_config
from dlab.hardware.wrappers.thorlabs_controller import ThorlabsController
from dlab.hardware.wrappers.powermeter_controller import PowermeterController

logger = logging.getLogger("dlab.ui.AutoWaveplateCalibWindow")

def _ressources_root() -> Path:
    cfg = get_config() or {}
    rel = (cfg.get("paths", {}) or {}).get("ressources", "ressources")
    return (ROOT / rel).resolve()


def _calib_dir(waveplate_name: str) -> Path:
    return _ressources_root() / "calibration" / waveplate_name


# ---------- worker ----------
class AutoAttCalibWorker(QObject):
    measurement_updated = pyqtSignal(float, float)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        motor_id: int,
        powermeter_id: str,
        wavelength_nm: float,
        angles_deg: Iterable[float],
        output_dir: Path,
        stabilization_s: float = 1.0,
        home_on_start: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.motor_id = motor_id
        self.powermeter_id = powermeter_id
        self.wavelength_nm = wavelength_nm
        self.angles_deg = list(float(a) for a in angles_deg)
        self.output_dir = Path(output_dir)
        self.stabilization_s = float(stabilization_s)
        self.home_on_start = bool(home_on_start)
        self.abort = False

        self._motor: ThorlabsController | None = None
        self._pm: PowermeterController | None = None

    @staticmethod
    def _unique_path(base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem, ext = base_path.stem, base_path.suffix
        i = 2
        while True:
            p = base_path.with_name(f"{stem}-{i}{ext}")
            if not p.exists():
                return p
            i += 1

    def _open_hardware(self) -> None:
        # Motor
        self._motor = ThorlabsController(self.motor_id)
        self._motor.activate(homing=self.home_on_start)
        self.log_signal.emit(f"Motor {self.motor_id} ready (home={self.home_on_start}).")

        # Powermeter
        self._pm = PowermeterController(self.powermeter_id)
        self._pm.activate()
        self._pm.set_wavelength(self.wavelength_nm)
        self.log_signal.emit(
            f"Powermeter {self.powermeter_id} ready at {self.wavelength_nm:.1f} nm."
        )

    def _close_hardware(self) -> None:
        try:
            if self._pm:
                self._pm.deactivate()
        except Exception as e:
            self.log_signal.emit(f"Powermeter close error: {e}")
        try:
            if self._motor:
                self._motor.disable()
        except Exception as e:
            self.log_signal.emit(f"Motor close error: {e}")
        self._pm = None
        self._motor = None

    def run(self) -> None:

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y_%m_%d")
        out_path = self._unique_path(self.output_dir / f"calib_{ts}.txt")

        try:
            self._open_hardware()
        except Exception as e:
            msg = f"Hardware init failed: {e}"
            self.log_signal.emit(msg)
            logger.exception(msg)
            self.finished.emit("")
            return

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                # Header
                f.write(f"# Date: {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"# MotorID: {self.motor_id}\n")
                f.write(f"# PowermeterID: {self.powermeter_id}\n")
                f.write(f"# Wavelength_nm: {self.wavelength_nm:.3f}\n")
                f.write(f"# Stabilization_s: {self.stabilization_s:.3f}\n")
                f.write("Angle_deg\tPower_W\n")

                for angle in self.angles_deg:
                    if self.abort:
                        self.log_signal.emit("Calibration aborted.")
                        break

                    # Move and settle
                    try:
                        assert self._motor is not None
                        self._motor.move_to(float(angle), blocking=True)
                    except Exception as e:
                        self.log_signal.emit(f"Move to {angle:.2f}° failed: {e}")
                        continue

                    time.sleep(self.stabilization_s)

                    try:
                        assert self._pm is not None
                        power_w = float(self._pm.read_power())
                    except Exception as e:
                        self.log_signal.emit(f"Power read failed at {angle:.2f}°: {e}")
                        power_w = float("nan")

                    self.measurement_updated.emit(float(angle), power_w)
                    self.log_signal.emit(f"Angle {angle:.2f}° → {power_w:.6e} W")
                    f.write(f"{angle:.6f}\t{power_w:.9f}\n")

            self.log_signal.emit(f"Saved calibration to {out_path.as_posix()}")
            logger.info("Calibration saved to %s", out_path)
        except Exception as e:
            msg = f"Calibration error: {e}"
            self.log_signal.emit(msg)
            logger.exception(msg)
        finally:
            self._close_hardware()
            self.finished.emit(out_path.as_posix())


class AutoWaveplateCalibWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Auto Attenuation Calibration")
        self._thread: QThread | None = None
        self._worker: AutoAttCalibWorker | None = None

        self._build_ui()
        self._angles: list[float] = []
        self._powers: list[float] = []

    def _build_ui(self) -> None:
        central = QWidget(self); self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QVBoxLayout()

        self.motor_id_edit = QLineEdit("83837725")
        self.pm_id_edit = QLineEdit("USB0::0x1313::0x8078::P0045634::INSTR")
        self.wavelength_edit = QLineEdit("1030.0")
        self.stab_edit = QLineEdit("1.0")
        self.wp_name_edit = QLineEdit("wp1")
        self.start_deg_edit = QLineEdit("0")
        self.end_deg_edit = QLineEdit("90")
        self.npts_edit = QLineEdit("91")
        self.home_chk = QPushButton("Home at start: ON")
        self.home_chk.setCheckable(True)
        self.home_chk.setChecked(True)
        self.home_chk.clicked.connect(
            lambda: self.home_chk.setText(f"Home at start: {'ON' if self.home_chk.isChecked() else 'OFF'}")
        )

        def add_row(label: str, w: QLineEdit | QPushButton):
            r = QHBoxLayout()
            r.addWidget(QLabel(label)); r.addWidget(w)
            left.addLayout(r)

        add_row("Motor ID:", self.motor_id_edit)
        add_row("Powermeter VISA:", self.pm_id_edit)
        add_row("Wavelength (nm):", self.wavelength_edit)
        add_row("Stabilization (s):", self.stab_edit)
        add_row("Waveplate name:", self.wp_name_edit)
        add_row("Angle start (°):", self.start_deg_edit)
        add_row("Angle end (°):", self.end_deg_edit)
        add_row("Points:", self.npts_edit)
        add_row("", self.home_chk)

        self.path_label = QLabel("")
        left.addWidget(QLabel("Output folder:"))
        left.addWidget(self.path_label)
        self._update_out_path()
        self.wp_name_edit.textChanged.connect(self._update_out_path)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.clicked.connect(self._abort)
        self.abort_btn.setEnabled(False)

        left.addWidget(self.start_btn)
        left.addWidget(self.abort_btn)
        left.addStretch(1)
        main.addLayout(left, 1)

        right = QVBoxLayout()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Angle (°)"); self.ax.set_ylabel("Power (W)")
        self.line, = self.ax.plot([], [], "bo-")
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        right.addWidget(QLabel("Log:")); right.addWidget(self.log_text)

        main.addLayout(right, 2)

    def _update_out_path(self) -> None:
        wp = (self.wp_name_edit.text() or "waveplate").strip()
        self.path_label.setText(_calib_dir(wp).as_posix())

    # Run
    def _start(self) -> None:
        try:
            motor_id = int(self.motor_id_edit.text())
            pm_id = self.pm_id_edit.text().strip()
            wl = float(self.wavelength_edit.text())
            stab = float(self.stab_edit.text())
            wp = (self.wp_name_edit.text() or "waveplate").strip()
            a0 = float(self.start_deg_edit.text())
            a1 = float(self.end_deg_edit.text())
            n = int(self.npts_edit.text())
            if n < 2:
                raise ValueError("Points must be ≥ 2.")
            if stab < 0:
                raise ValueError("Stabilization must be ≥ 0.")
        except Exception as e:
            QMessageBox.critical(self, "Invalid input", str(e))
            return

        angles = np.linspace(a0, a1, n, dtype=float)

        # reset series
        self._angles, self._powers = [], []
        self.line.set_data([], [])
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

        out_dir = _calib_dir(wp)

        # thread + worker
        self._thread = QThread(self)
        self._worker = AutoAttCalibWorker(
            motor_id=motor_id,
            powermeter_id=pm_id,
            wavelength_nm=wl,
            angles_deg=angles,
            output_dir=out_dir,
            stabilization_s=stab,
            home_on_start=self.home_chk.isChecked(),
        )
        self._worker.moveToThread(self._thread)

        # signals
        self._thread.started.connect(self._worker.run)
        self._worker.measurement_updated.connect(self._on_point)
        self._worker.log_signal.connect(self._log)
        self._worker.finished.connect(self._finished)

        # cleanup on thread end
        self._thread.finished.connect(self._thread.deleteLater)

        # go
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._thread.start()
        self._log("Calibration started…")

    def _abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.abort_btn.setEnabled(False)

    # slots
    def _on_point(self, angle: float, power_w: float) -> None:
        self._angles.append(angle); self._powers.append(power_w)
        self.line.set_data(self._angles, self._powers)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        logger.info(msg)

    def _finished(self, out_path: str) -> None:
        if out_path:
            self._log(f"Finished. Saved to {out_path}")
        else:
            self._log("Finished with errors.")
        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

        # stop thread
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None


if __name__ == "__main__":
    app = QApplication([])
    win = AutoWaveplateCalibWindow()
    win.show()
    app.exec_()
