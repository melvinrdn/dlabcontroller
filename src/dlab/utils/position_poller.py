from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QLineEdit


class PositionPoller(QObject):
    """Periodically reads a device position and displays it in a QLineEdit.

    Stops itself and logs once if the getter raises, instead of retrying
    every tick against a connection that's already gone.
    """

    def __init__(
        self,
        get_position: Callable[[], Optional[float]],
        target_edit: QLineEdit,
        log: Callable[[str], None] | None = None,
        interval_ms: int = 200,
        fmt: str = "{:.3f}",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_position = get_position
        self._target_edit = target_edit
        self._log = log or (lambda msg: None)
        self._fmt = fmt
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        try:
            pos = self._get_position()
        except Exception as e:
            self._timer.stop()
            self._log(f"Position read failed: {e}")
            return
        if pos is not None:
            self._target_edit.setText(self._fmt.format(pos))
