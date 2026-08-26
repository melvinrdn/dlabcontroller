import serial
import time

V_MIN = 0.0
V_MAX = 140.0


class NV40Error(RuntimeError):
    """Raised for NV40 protocol/configuration errors."""


class NV40:
    ERRORS = {
        'err,1': 'Unknown command',
        'err,2': 'Too many characters in the command',
        'err,3': 'Too many characters in the parameter',
        'err,4': 'Too many parameters',
        'err,5': 'Wrong character in parameter',
        'err,6': 'Wrong separator',
        'err,7': 'Position out of range',
    }
    def __init__(self, port, timeout=0.05, closed_loop=False):
        if closed_loop:
            raise NV40Error("Closed-loop mode is not supported on this NV40.")

        self.ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=timeout,
            write_timeout=0.05,
        )
        time.sleep(0.05)

        self.set_remote_control(True)
        self._send("ol")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def _send(self, cmd: str):
        self.ser.write((cmd + "\r").encode())

    def _query(self, cmd: str) -> str:
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\r").encode())
        ans = self.ser.readline().decode(errors="ignore").strip()
        if ans in self.ERRORS:
            raise NV40Error(self.ERRORS[ans])
        return ans

    def set_remote_control(self, enable=True):
        self._send("i1" if enable else "i0")

    def set_closed_loop(self, enable=True):
        if enable:
            raise NV40Error("Closed loop not supported.")
        self._send("ol")

    def set_position(self, value: float):
        value = max(V_MIN, min(V_MAX, value))
        self._send(f"wr,{value:.3f}")

    def get_position(self) -> float:
        ans = self._query("rd")
        try:
            return float(ans.split(",")[1])
        except (IndexError, ValueError) as e:
            raise NV40Error(f"Unparseable position reply {ans!r}") from e

    @classmethod
    def get_voltage_limits(cls):
        return V_MIN, V_MAX


if __name__ == "__main__":
        dev = NV40("COM6")

        import numpy as np
        for v in np.arange(10, 11, 0.1):
            dev.set_position(v)
            time.sleep(0.05)
            print(v, dev.get_position())

        dev.close()
