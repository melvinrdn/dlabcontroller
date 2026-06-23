from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget

from dlab.utils.log_panel import LogPanel


class ScanWindow(QMainWindow):
    """Main window containing various scan tabs."""

    closed = pyqtSignal()

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scan")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._init_ui()

    def _init_ui(self) -> None:
        # Lazy imports to avoid circular imports
        from dlab.diagnostics.ui.scans.grid_scan_tab import GridScanTab
        from dlab.diagnostics.ui.scans.two_color_scan_tab import TwoColorScanTab
        from dlab.diagnostics.ui.scans.m2_measurement_tab import M2Tab
        from dlab.diagnostics.ui.scans.grating_compressor_scan_tab import GCScanTab
        from dlab.diagnostics.ui.scans.temporal_overlap_scan_tab import TOverlapTab


        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._tabs.addTab(GridScanTab(log_panel=self._log), "Grid Scan")
        self._tabs.addTab(TwoColorScanTab(log_panel=self._log), "Two-Color Scan")
        self._tabs.addTab(M2Tab(log_panel=self._log), "M² Scan")
        self._tabs.addTab(GCScanTab(log_panel=self._log), "Grating Compressor Scan")
        self._tabs.addTab(TOverlapTab(log_panel=self._log), "Temporal Overlap Scan")
        

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ScanWindow()
    window.show()
    sys.exit(app.exec_())