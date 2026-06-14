"""Dual progress bar: Total (phase) + Process (within current phase)."""

from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, Qt, QRectF
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..theme import Anim, Colors, Dims, Fonts


class _SingleBar(QWidget):
    """Internal animated bar used for each section."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._shimmer = 0.0
        self._eta = ""
        self.setMinimumHeight(20)

        self._value_anim = QPropertyAnimation(self, b"value", self)
        self._value_anim.setDuration(400)
        self._value_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._shimmer_anim = QPropertyAnimation(self, b"shimmer", self)
        self._shimmer_anim.setDuration(2000)
        self._shimmer_anim.setStartValue(0.0)
        self._shimmer_anim.setEndValue(1.0)
        self._shimmer_anim.setLoopCount(-1)
        self._shimmer_anim.start()

    def get_value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, value))
        self.update()

    value = Property(float, get_value, set_value)

    def get_shimmer(self) -> float:
        return self._shimmer

    def set_shimmer(self, v: float) -> None:
        self._shimmer = v
        self.update()

    shimmer = Property(float, get_shimmer, set_shimmer)

    def setValue(self, value: float, eta: str = "") -> None:  # noqa: N802
        target = max(0.0, min(100.0, value))
        self._eta = eta
        self._value_anim.stop()
        self._value_anim.setStartValue(self._value)
        self._value_anim.setEndValue(target)
        self._value_anim.start()

    def reset(self) -> None:
        self._value_anim.stop()
        self._value = 0.0
        self._eta = ""
        self.update()

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
            painter.setClipRect(fill_rect)
            painter.setBrush(gradient)
            painter.drawRoundedRect(rect, radius, radius)
            # Shimmer.
            sweep_x = fill_rect.left() + self._shimmer * fill_rect.width()
            sg = QLinearGradient(sweep_x - 40, 0, sweep_x + 40, 0)
            t = QColor(255, 255, 255, 0)
            b = QColor(255, 255, 255, 55)
            sg.setColorAt(0.0, t)
            sg.setColorAt(0.5, b)
            sg.setColorAt(1.0, t)
            painter.setBrush(sg)
            painter.drawRect(fill_rect)
            painter.restore()

        painter.setPen(QColor(Colors.TEXT_PRIMARY))
        font = painter.font()
        font.setFamily(Fonts.FAMILY_PRIMARY)
        font.setPointSize(Fonts.SIZE_SMALL)
        painter.setFont(font)
        label = f"{int(round(self._value))}%"
        if self._eta:
            label += f"  ETA {self._eta}"
        painter.drawText(rect, Qt.AlignCenter, label)
        painter.end()


class AnimatedProgressBar(QWidget):
    """Dual-section progress bar: Total (phases) and Process (current phase)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._total_label = QLabel("Total", self)
        self._total_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
        )
        self._total_label.setFixedWidth(38)
        self._total_bar = _SingleBar(self)

        self._process_label = QLabel("Process", self)
        self._process_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SMALL}px;"
        )
        self._process_label.setFixedWidth(52)
        self._process_bar = _SingleBar(self)

        layout.addWidget(self._total_label)
        layout.addWidget(self._total_bar, 2)
        layout.addWidget(self._process_label)
        layout.addWidget(self._process_bar, 3)

    # ---------------------------------------------------------------- api
    def setValue(self, value: float, eta: str = "") -> None:  # noqa: N802
        """Set the Process (current phase) progress."""
        self._process_bar.setValue(value, eta)

    def setTotalValue(self, value: float, eta: str = "") -> None:  # noqa: N802
        """Set the Total (phase) progress."""
        self._total_bar.setValue(value, eta)

    def reset(self) -> None:
        self._total_bar.reset()
        self._process_bar.reset()
