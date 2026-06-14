"""3D-styled QPushButton rendered entirely with QPainter."""

from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, Qt, QRectF
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QPushButton

from ..theme import Anim, Colors, Dims, Fonts


_VARIANTS = {
    "default": (Colors.SURFACE_HOVER, Colors.SURFACE, Colors.TEXT_PRIMARY),
    "primary": (Colors.ACCENT_HOVER, Colors.ACCENT, Colors.WHITE),
    "danger": (Colors.DANGER_HOVER, Colors.DANGER, Colors.WHITE),
    "success": (Colors.SUCCESS_HOVER, Colors.SUCCESS, Colors.SUCCESS_TEXT),
    "ghost": ("#00000000", "#00000000", Colors.TEXT_PRIMARY),
}


class Button3D(QPushButton):
    """A push button with a painted beveled 3D body, drop shadow and animations."""

    def __init__(self, text: str = "", variant: str = "default", parent=None) -> None:
        super().__init__(text, parent)
        self._variant = variant if variant in _VARIANTS else "default"
        self._brightness = 0.0  # 0 = idle, 1 = hover
        self._press_offset = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(Dims.BUTTON_HEIGHT_MD)
        # Paint everything ourselves: strip the global QSS for this button.
        self.setStyleSheet("background: transparent; border: none;")
        self.setFlat(True)

        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setColor(QColor(0, 0, 0, 150))
        self._shadow.setBlurRadius(12)
        self._shadow.setOffset(0, 3)
        if self._variant != "ghost":
            self.setGraphicsEffect(self._shadow)

        self._bright_anim = QPropertyAnimation(self, b"brightness", self)
        self._bright_anim.setDuration(Anim.NORMAL)
        self._bright_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._press_anim = QPropertyAnimation(self, b"pressOffset", self)
        self._press_anim.setDuration(Anim.FAST)
        self._press_anim.setEasingCurve(QEasingCurve.OutCubic)

    # ---------------------------------------------------------------- props
    def get_brightness(self) -> float:
        return self._brightness

    def set_brightness(self, value: float) -> None:
        self._brightness = value
        if self._variant != "ghost":
            blur = 12 + 6 * value
            off = 3 + 3 * value
            self._shadow.setBlurRadius(blur)
            self._shadow.setOffset(0, off)
        self.update()

    brightness = Property(float, get_brightness, set_brightness)

    def get_press_offset(self) -> float:
        return self._press_offset

    def set_press_offset(self, value: float) -> None:
        self._press_offset = value
        if self._variant != "ghost":
            self._shadow.setOffset(0, max(0.0, 3 + 3 * self._brightness - value * 3))
        self.update()

    pressOffset = Property(float, get_press_offset, set_press_offset)

    # ---------------------------------------------------------------- events
    def enterEvent(self, event) -> None:  # noqa: N802
        self._animate(self._bright_anim, 1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._animate(self._bright_anim, 0.0)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._animate(self._press_anim, 1.0)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._animate(self._press_anim, 0.0)
        super().mouseReleaseEvent(event)

    @staticmethod
    def _animate(anim: QPropertyAnimation, end: float) -> None:
        anim.stop()
        anim.setEndValue(end)
        anim.start()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        top_color, bottom_color, text_color = _VARIANTS[self._variant]
        top = QColor(top_color)
        bottom = QColor(bottom_color)

        radius = Dims.CORNER_RADIUS_MD
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        rect.translate(0, 2 * self._press_offset)

        if not self.isEnabled():
            top = QColor(Colors.BG_MEDIUM)
            bottom = QColor(Colors.BG_MEDIUM)
            text_color = Colors.TEXT_MUTED

        if self._variant == "ghost":
            if self._brightness > 0.0:
                hov = QColor(Colors.SURFACE_HOVER)
                hov.setAlphaF(0.6 * self._brightness)
                path = QPainterPath()
                path.addRoundedRect(rect, radius, radius)
                painter.fillPath(path, hov)
        else:
            # Brighten on hover.
            factor = 1.0 + 0.12 * self._brightness
            top = self._scale(top, factor)
            bottom = self._scale(bottom, factor)

            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, top)
            gradient.setColorAt(1.0, bottom)

            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            painter.fillPath(path, gradient)

            # Top highlight line (1px white at 10% opacity) for 3D top edge.
            highlight = QColor(255, 255, 255, 26)
            painter.setPen(QPen(highlight, 1))
            painter.drawLine(
                rect.left() + radius,
                rect.top() + 1,
                rect.right() - radius,
                rect.top() + 1,
            )

        # Text + icon.
        painter.setPen(QColor(text_color))
        font = painter.font()
        font.setFamily(Fonts.FAMILY_PRIMARY)
        font.setPointSize(Fonts.SIZE_BODY)
        font.setWeight(QtWeight(Fonts.WEIGHT_MEDIUM))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self.text())
        painter.end()

    @staticmethod
    def _scale(color: QColor, factor: float) -> QColor:
        return QColor(
            min(255, int(color.red() * factor)),
            min(255, int(color.green() * factor)),
            min(255, int(color.blue() * factor)),
            color.alpha(),
        )


def QtWeight(weight: int):
    """Map a numeric CSS weight to a QFont.Weight value."""
    from PySide6.QtGui import QFont

    mapping = {
        300: QFont.Light,
        400: QFont.Normal,
        500: QFont.Medium,
        600: QFont.DemiBold,
        700: QFont.Bold,
    }
    return mapping.get(weight, QFont.Normal)
