from typing import Optional

import pyvisa
from ThorlabsPM100 import ThorlabsPM100


class PowermeterControllerError(Exception):
    """Errors raised by PowermeterController."""


class PowermeterController: 

    def __init__(self, powermeter_id: str) -> None:
        """
        Parameters
        ----------
        powermeter_id : str
            VISA resource string, e.g. 'USB0::0x1313::0x8078::P0045634::INSTR'
        """
        self.powermeter_id = powermeter_id
        self._rm: Optional[pyvisa.ResourceManager] = None
        self.instrument: Optional[pyvisa.resources.MessageBasedResource] = None
        self.power_meter: Optional[ThorlabsPM100] = None

    def activate(self) -> None:
        """Open VISA resource and initialize ThorlabsPM100."""
        try:
            self._rm = pyvisa.ResourceManager()
            self.instrument = self._rm.open_resource(self.powermeter_id)
            self.power_meter = ThorlabsPM100(inst=self.instrument)
            self.set_auto_range(1)  # default ON
        except Exception as e:
            self.power_meter = None
            self.instrument = None
            raise PowermeterControllerError(f"Failed to activate powermeter: {e}") from e

    def deactivate(self) -> None:
        """Close the VISA resource."""
        try:
            if self.instrument is not None:
                self.instrument.close()
        finally:
            self.instrument = None
            self.power_meter = None

    def read_power(self) -> float:
        """
        Trigger a measurement and return power in Watts (blocking).
        """
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        try:
            return float(self.power_meter.read)
        except Exception as e:
            raise PowermeterControllerError(f"read_power failed: {e}") from e

    def fetch_power(self) -> float:
        """
        Return the last measured power in Watts.
        """
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        try:
            return float(self.power_meter.fetch)
        except Exception as e:
            raise PowermeterControllerError(f"fetch_power failed: {e}") from e

    def set_auto_range(self, auto_range: int | bool) -> None:
        """
        Enable/disable autoranging.
        """
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        val = "ON" if bool(auto_range) else "OFF"
        try:
            self.power_meter.sense.power.dc.range.auto = val
        except Exception as e:
            raise PowermeterControllerError(f"set_auto_range failed: {e}") from e

    def set_avg(self, no_avg: int) -> None:
        """Set the averaging count."""
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        if not isinstance(no_avg, int) or no_avg <= 0:
            raise ValueError("Averaging count must be a positive integer")
        try:
            self.power_meter.sense.average.count = no_avg
        except Exception as e:
            raise PowermeterControllerError(f"set_avg failed: {e}") from e

    def set_wavelength(self, wavelength: float) -> None:
        """Set operating wavelength in nm."""
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        try:
            self.power_meter.sense.correction.wavelength = float(wavelength)
        except Exception as e:
            raise PowermeterControllerError(f"set_wavelength failed: {e}") from e

    def set_bandwidth(self, bandwidth: str) -> None:
        """
        Set bandwidth: "high" or "low".
        """
        if self.power_meter is None:
            raise PowermeterControllerError("Device not active")
        mode = (bandwidth or "").strip().lower()
        try:
            if mode == "high":
                self.power_meter.input.pdiode.filter.lpass.state = 0
            elif mode == "low":
                self.power_meter.input.pdiode.filter.lpass.state = 1
            else:
                raise ValueError("bandwidth must be 'high' or 'low'")
        except Exception as e:
            raise PowermeterControllerError(f"set_bandwidth failed: {e}") from e

