"""Circular loading spinner (12 fading arc segments at ~60fps)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import Colors

_SEGMENTS = 12


class Spinner(QWidget):
    """A small indeterminate circular spinner."""

    def __init__(self, size: int = 28, parent=None) -> None:
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._rotate)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _rotate(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)

        radius = min(self.width(), self.height()) / 2 - 2
        inner = radius * 0.5
        for i in range(_SEGMENTS):
            alpha = int(255 * (i + 1) / _SEGMENTS)
            color = QColor(Colors.ACCENT)
            color.setAlpha(alpha)
            pen = QPen(color, 2.2, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(0, int(inner), 0, int(radius))
            painter.rotate(360 / _SEGMENTS)
        painter.end()
