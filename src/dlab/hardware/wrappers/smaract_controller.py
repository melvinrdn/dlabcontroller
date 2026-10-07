from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Optional

import pylablib
from pylablib.devices import SmarAct

from dlab.boot import ROOT
from dlab.utils.config_utils import cfg_get


class SmarActNotActivatedError(RuntimeError):
    """Raised when an operation requires an activated controller."""


_LOAD_WITH_ALTERED_SEARCH_PATH = 0x8


def _configure_dll_path() -> Path:
    driver_path = str(cfg_get("paths.drivers_smaract", "src/dlab/hardware/drivers/smaract_driver"))
    dll_dir = (ROOT / driver_path).resolve()
    pylablib.par["devices/dlls/smaract_mcs2"] = str(dll_dir)

    # ctypes' default DLL search mode (Python's hardened default since 3.8) fails
    # to load SmarActCTL.dll with a WinError 1114 (DllMain init failure) on some
    # machines; LOAD_WITH_ALTERED_SEARCH_PATH (classic search order) works.
    # Pre-load it that way (letting it pull in its co-located SmarActIO.dll /
    # SmarActLog.dll dependencies itself) so pylablib's own load call just
    # reuses the already resident module instead of retrying with the default
    # (broken) mode.
    dll_path = dll_dir / "SmarActCTL.dll"
    if dll_path.exists():
        ctypes.WinDLL(str(dll_path), winmode=_LOAD_WITH_ALTERED_SEARCH_PATH)
    return dll_dir


def preload_dll() -> None:
    """Force-load SmarActCTL.dll now, so import-time issues fail fast at startup."""
    _configure_dll_path()
    SmarAct.get_mcs2_SDK_version()


class SmarActController:
    """Controls one SmarAct MCS2 unit and all of its axes over a single shared connection.

    The MCS2 only accepts one open connection at a time, so unlike the
    single-axis Thorlabs/Zaber controllers, one instance here manages the
    whole device and every method takes an explicit axis index.
    """

    def __init__(self, locator: str):
        self.locator = locator
        self.stage: Optional[SmarAct.MCS2] = None

    @staticmethod
    def list_devices() -> list[str]:
        """List locators of connected SmarAct MCS2 devices."""
        return SmarAct.list_msc2_devices()

    def _ensure(self) -> SmarAct.MCS2:
        if self.stage is None:
            raise SmarActNotActivatedError("Controller not activated. Call activate() first.")
        return self.stage

    def activate(self) -> None:
        _configure_dll_path()
        self.stage = SmarAct.MCS2(self.locator)

    @property
    def naxes(self) -> int:
        return int(self._ensure().naxes)

    def has_sensor(self, axis: int) -> bool:
        return "sensor_present" in self._ensure().get_status(axis)

    def home(self, axis: int, blocking: bool = True) -> None:
        self._ensure().home(axis, sync=blocking)

    def move_to(self, axis: int, position: float, blocking: bool = True) -> None:
        stage = self._ensure()
        stage.move_to(position, axis=axis)
        if blocking:
            stage.wait_move(axis)

    def get_position(self, axis: int) -> Optional[float]:
        if self.stage is None:
            return None
        return self.stage.get_position(axis)

    def is_moving(self, axis: int) -> bool:
        return bool(self._ensure().is_moving(axis))

    def disable(self) -> None:
        if self.stage is not None:
            try:
                self.stage.stop("all")
            finally:
                try:
                    self.stage.close()
                finally:
                    self.stage = None


class SmarActAxis:
    """Single-axis view over a shared SmarActController.

    Exposes the no-axis-argument interface (move_to/get_position/home) used
    elsewhere in the app (Thorlabs, Zaber, scan tabs), so one axis can be
    registered and driven like any other stage.
    """

    def __init__(self, controller: SmarActController, axis: int):
        self.controller = controller
        self.axis = axis

    def move_to(self, position: float, blocking: bool = True) -> None:
        self.controller.move_to(self.axis, position, blocking=blocking)

    def get_position(self) -> Optional[float]:
        return self.controller.get_position(self.axis)

    def home(self, blocking: bool = True) -> None:
        self.controller.home(self.axis, blocking=blocking)
