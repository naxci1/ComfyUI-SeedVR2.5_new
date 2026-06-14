"""Progress bar with shimmer sweep, smooth value transitions and ETA overlay."""

from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, Qt, QRectF
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import Anim, Colors, Dims, Fonts


class AnimatedProgressBar(QWidget):
    """A custom progress bar with a looping shimmer and animated fill."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value = 0.0          # displayed value (animated) 0..100
        self._target = 0.0
        self._shimmer = 0.0        # 0..1 sweep position
        self._eta = ""
        self.setMinimumHeight(22)

        self._value_anim = QPropertyAnimation(self, b"value", self)
        self._value_anim.setDuration(400)
        self._value_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._shimmer_anim = QPropertyAnimation(self, b"shimmer", self)
        self._shimmer_anim.setDuration(2000)
        self._shimmer_anim.setStartValue(0.0)
        self._shimmer_anim.setEndValue(1.0)
        self._shimmer_anim.setLoopCount(-1)
        self._shimmer_anim.start()

    # ---------------------------------------------------------------- props
    def get_value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, value))
        self.update()

    value = Property(float, get_value, set_value)

    def get_shimmer(self) -> float:
        return self._shimmer

    def set_shimmer(self, value: float) -> None:
        self._shimmer = value
        self.update()

    shimmer = Property(float, get_shimmer, set_shimmer)

    # ---------------------------------------------------------------- api
    def setValue(self, value: float, eta: str = "") -> None:  # noqa: N802
        self._target = max(0.0, min(100.0, value))
        self._eta = eta
        self._value_anim.stop()
        self._value_anim.setStartValue(self._value)
        self._value_anim.setEndValue(self._target)
        self._value_anim.start()

    def reset(self) -> None:
        self._value_anim.stop()
        self._value = 0.0
        self._target = 0.0
        self._eta = ""
        self.update()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        radius = Dims.CORNER_RADIUS_SM
        rect = QRectF(self.rect())

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Colors.SCRUB_TRACK))
        painter.drawRoundedRect(rect, radius, radius)

        frac = self._value / 100.0
        if frac > 0:
            fill_w = rect.width() * frac
            fill_rect = QRectF(rect.left(), rect.top(), fill_w, rect.height())

            gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            gradient.setColorAt(0.0, QColor(Colors.SUCCESS))
            gradient.setColorAt(1.0, QColor(Colors.ACCENT))
            painter.save()
            path_rect = QRectF(rect)
            painter.setClipRect(fill_rect)
            painter.setBrush(gradient)
            painter.drawRoundedRect(path_rect, radius, radius)

            # Shimmer sweep.
            sweep_x = fill_rect.left() + self._shimmer * fill_rect.width()
            shimmer_grad = QLinearGradient(sweep_x - 40, 0, sweep_x + 40, 0)
            transparent = QColor(255, 255, 255, 0)
            bright = QColor(255, 255, 255, 55)
            shimmer_grad.setColorAt(0.0, transparent)
            shimmer_grad.setColorAt(0.5, bright)
            shimmer_grad.setColorAt(1.0, transparent)
            painter.setBrush(shimmer_grad)
            painter.drawRect(fill_rect)
            painter.restore()

        # Percentage + ETA text overlay.
        painter.setPen(QColor(Colors.TEXT_PRIMARY))
        font = painter.font()
        font.setFamily(Fonts.FAMILY_PRIMARY)
        font.setPointSize(Fonts.SIZE_SMALL)
        painter.setFont(font)
        label = f"{int(round(self._value))}%"
        if self._eta:
            label += f"  •  ETA {self._eta}"
        painter.drawText(rect, Qt.AlignCenter, label)
        painter.end()
