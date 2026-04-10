#!/usr/bin/env python3
"""
SeedVR2 GUI – Entry point.
Run directly:  python gui/app.py
Or as package: python -m gui.app
"""

from __future__ import annotations

import platform
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

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
    app.setApplicationName("SeedVR2.5 Upscaler by HB2k")
    app.setOrganizationName("SeedVR2")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
