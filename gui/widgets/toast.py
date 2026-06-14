"""Non-blocking slide-in toast notifications."""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QPoint,
    Qt,
    QTimer,
    QRectF,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from ..theme import Anim, Colors, Dims, Fonts

_LEVELS = {
    "info": (Colors.ACCENT, "ⓘ"),
    "success": (Colors.SUCCESS, "✓"),
    "warning": (Colors.WARNING, "⚠"),
    "error": (Colors.DANGER, "✕"),
}


class Toast(QWidget):
    """A single toast that slides in from top-center and auto-dismisses."""

    def __init__(self, parent: QWidget, message: str, level: str = "info", duration: int = 3000) -> None:
        super().__init__(parent)
        self._level = level if level in _LEVELS else "info"
        self._message = message
        self._duration = duration
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setMinimumHeight(44)

        self._label = QLabel(message, self)
        self._label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-family: '{Fonts.FAMILY_PRIMARY}';"
            f" font-size: {Fonts.SIZE_BODY}px; background: transparent;"
        )
        self._label.move(48, 12)
        self._label.adjustSize()

        width = max(220, self._label.width() + 72)
        self.resize(width, 44)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setDuration(Anim.NORMAL)
        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(Anim.SLOW)
        self._slide.setEasingCurve(QEasingCurve.OutCubic)

    # ---------------------------------------------------------------- show
    def _start(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        start_x = (parent.width() - self.width()) // 2
        start_y = -self.height()
        end_y = 24
        self.move(start_x, start_y)
        QWidget.show(self)
        self.raise_()

        self._slide.setStartValue(QPoint(start_x, start_y))
        self._slide.setEndValue(QPoint(start_x, end_y))
        self._slide.start()

        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

        QTimer.singleShot(self._duration, self._dismiss)

    def _dismiss(self) -> None:
        self._fade.stop()
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.deleteLater)
        self._fade.start()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = Dims.CORNER_RADIUS_MD

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Colors.BG_LIGHTER))
        painter.drawRoundedRect(rect, radius, radius)

        accent_color, icon = _LEVELS[self._level]
        # Left accent bar.
        painter.setBrush(QColor(accent_color))
        painter.drawRoundedRect(QRectF(0, 0, 4, rect.height()), 2, 2)

        # Icon.
        painter.setPen(QColor(accent_color))
        font = painter.font()
        font.setPointSize(Fonts.SIZE_H2)
        painter.setFont(font)
        painter.drawText(QRectF(16, 0, 28, rect.height()), Qt.AlignCenter, icon)
        painter.end()

    # ---------------------------------------------------------------- static
    @staticmethod
    def show(parent: QWidget, message: str, level: str = "info", duration: int = 3000) -> "Toast":
        toast = Toast(parent, message, level, duration)
        toast._start()
        return toast
