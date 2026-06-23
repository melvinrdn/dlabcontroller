from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_APT_DLL_DIR = Path(__file__).resolve().parent.parent / "drivers" / "thorlabs_driver"
os.add_dll_directory(str(_APT_DLL_DIR))
os.environ["PATH"] = str(_APT_DLL_DIR) + os.pathsep + os.environ["PATH"]

import thorlabs_apt as apt  # noqa: E402


class ThorlabsNotActivatedError(RuntimeError):
    """Raised when an operation requires an activated motor."""


class ThorlabsController:

    def __init__(self, motor_id: int) -> None:
        self.motor_id: int = motor_id
        self.motor: Optional[apt.Motor] = None

    def activate(self, homing: bool = True) -> None:
        """Instantiate the motor handle and optionally home."""
        self.motor = apt.Motor(self.motor_id)
        if homing:
            self.home()

    def _ensure(self) -> apt.Motor:
        """Return the motor handle or raise if not activated."""
        if self.motor is None:
            raise ThorlabsNotActivatedError("Motor not activated. Call activate() first.")
        return self.motor

    def home(self, blocking: bool = True) -> None:
        """Send the motor to its home position."""
        self._ensure().move_home(blocking=blocking)

    def move_to(self, position: float, blocking: bool = True) -> None:
        """Move the motor to an absolute position (device units)."""
        self._ensure().move_to(position, blocking=blocking)

    def get_position(self) -> Optional[float]:
        """Return the current position, or None if not activated."""
        return self.motor.position if self.motor is not None else None

    def identify(self) -> None:
        """Flash the device LED to identify the unit."""
        self._ensure().identify()

    def disable(self) -> None:
        """Disable the motor and drop the handle."""
        if self.motor is not None:
            try:
                if hasattr(self.motor, "disable"):
                    self.motor.disable()
            finally:
                self.motor = None
