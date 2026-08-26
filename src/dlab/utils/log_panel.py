import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class LogPanel(QWidget):
    """A floating window for centralized timestamped log messages."""

    def __init__(self, time_fmt: str = "%H:%M:%S", max_lines: int = 500):
        super().__init__()
        self._time_fmt = time_fmt
        self._max_lines = max_lines
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Log")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.resize(500, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def log(self, message: str, source: str | None = None):
        """Append a timestamped message, optionally with a source prefix."""
        timestamp = datetime.datetime.now().strftime(self._time_fmt)
        if source:
            self._text.append(f"[{timestamp}] [{source}] {message}")
        else:
            self._text.append(f"[{timestamp}] {message}")
        self._trim_if_needed()

    def logger(self, source: str):
        """Return a callable that logs with a fixed source prefix."""
        return lambda message: self.log(message, source)

    def _trim_if_needed(self):
        """Remove old lines if exceeding max_lines."""
        doc = self._text.document()
        if doc and doc.blockCount() > self._max_lines:
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 50)
            cursor.removeSelectedText()
