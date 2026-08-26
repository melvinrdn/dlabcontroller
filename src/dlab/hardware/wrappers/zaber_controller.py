from __future__ import annotations
from typing import Optional

from zaber_motion import Library, Units
from zaber_motion.binary import Connection, Device


class ZaberNotActivatedError(RuntimeError):
    """Error Management"""


class ZaberBinaryController:

    def __init__(self, port: str, baud_rate: int = 9600,
                 range_min: float = 0.0, range_max: float = 50.0):
        self.port = port
        self.baud_rate = baud_rate
        self.conn: Optional[Connection] = None
        self.device: Optional[Device] = None

        self.range_min = range_min
        self.range_max = range_max

    def _ensure(self) -> Device:
        if self.device is None:
            raise ZaberNotActivatedError("Stage not activated. Call activate() first.")
        return self.device

    def activate(self, homing: bool = True) -> None:
        Library.enable_device_db_store()
        self.conn = Connection.open_serial_port(self.port, baud_rate=self.baud_rate)
        devices = self.conn.detect_devices()
        if not devices:
            self.disable()
            raise ZaberNotActivatedError(f"No Zaber devices found on {self.port}@{self.baud_rate}")
        self.device = devices[0]
        if homing:
            try:
                self.home()
            except Exception:
                self.disable()
                raise

    def home(self, blocking: bool = True) -> None:
        dev = self._ensure()
        dev.home()
        if blocking:
            dev.wait_until_idle()

    def move_to(self, position: float, blocking: bool = True) -> None:
        dev = self._ensure()
        pos = max(self.range_min, min(self.range_max, float(position)))
        dev.move_absolute(pos, Units.LENGTH_MILLIMETRES)
        if blocking:
            dev.wait_until_idle()

    def get_position(self) -> Optional[float]:
        if self.device is None:
            return None
        return float(self.device.get_position(Units.LENGTH_MILLIMETRES))

    def identify(self) -> None:
        self._ensure().identify()

    def disable(self) -> None:
        try:
            if self.device is not None:
                self.device.stop()
        finally:
            try:
                if self.conn is not None:
                    self.conn.close()
            finally:
                self.device = None
                self.conn = None
