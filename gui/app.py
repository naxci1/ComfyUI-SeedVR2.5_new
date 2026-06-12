#!/usr/bin/env python3
"""
SeedVR2 GUI – Entry point.
Run directly:  python gui/app.py
Or as package: python -m gui.app
"""

from __future__ import annotations

import platform
import sys
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

# Windows: bind this process to a unique AppUserModelID so Windows shows the
# correct taskbar icon instead of the generic Python interpreter icon.
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "naxci1.seedvr.upscaler.25"
        )
    except Exception:
        pass

try:
    from gui.styles import DARK_STYLESHEET
    from gui.main_window import MainWindow
except ImportError:
    from styles import DARK_STYLESHEET  # type: ignore[no-redef]
    from main_window import MainWindow  # type: ignore[no-redef]


def main() -> int:
    # High-DPI support (relevant on Windows with display scaling)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("1Click SeedVR2.5 ver. 1.7b (by Naxci1)")
    app.setOrganizationName("SeedVR2")
    app.setStyleSheet(DARK_STYLESHEET)

    def _show_uncaught_exception(exc_type, exc_value, exc_tb) -> None:
        trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Unhandled Error")
        box.setText("A critical error occurred, but the application will remain open.")
        box.setDetailedText(trace)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        copy_btn = box.addButton("Copy to Clipboard", QMessageBox.ButtonRole.ActionRole)
        box.exec()
        if box.clickedButton() == copy_btn:
            QApplication.clipboard().setText(trace)

    sys.excepthook = _show_uncaught_exception

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
