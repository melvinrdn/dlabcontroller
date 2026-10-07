from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QWidget

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
        from dlab.diagnostics.ui.scans.grating_compressor_scan_tab import GCScanTab
        from dlab.diagnostics.ui.scans.temporal_overlap_scan_tab import TOverlapTab
        from dlab.diagnostics.ui.scans.intensity_ladder_scan_tab import IntensityLadderScanTab

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._tabs.addTab(GridScanTab(log_panel=self._log), "Grid Scan")
        self._tabs.addTab(TwoColorScanTab(log_panel=self._log), "Two-Color Scan")
        self._tabs.addTab(GCScanTab(log_panel=self._log), "Grating Compressor Scan")
        self._tabs.addTab(TOverlapTab(log_panel=self._log), "Temporal Overlap Scan")
        self._tabs.addTab(IntensityLadderScanTab(log_panel=self._log), "Intensity Ladder Scan")


    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _running_tabs(self) -> list[tuple[str, QWidget]]:
        """Tabs whose scan QThread is still alive.

        Every tab (grid/two-color/M2/grating-compressor/temporal-overlap)
        follows the same self._thread/self._worker convention, so this
        works generically without each tab needing to expose anything.
        """
        running = []
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            thread = getattr(tab, "_thread", None)
            if thread is not None and thread.isRunning():
                running.append((self._tabs.tabText(i), tab))
        return running

    def closeEvent(self, event) -> None:
        running = self._running_tabs()
        if running:
            names = ", ".join(name for name, _ in running)
            reply = QMessageBox.question(
                self, "Scan running",
                f"A scan is still running on: {names}.\n\n"
                "Closing now will abort it. Abort and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

            for _, tab in running:
                worker = getattr(tab, "_worker", None)
                if worker is not None:
                    worker.abort = True
            for _, tab in running:
                thread = getattr(tab, "_thread", None)
                if thread is not None:
                    thread.quit()
                    thread.wait(5000)

        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ScanWindow()
    window.show()
    sys.exit(app.exec_())