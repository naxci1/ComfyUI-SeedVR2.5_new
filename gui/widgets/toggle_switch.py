"""Animated iOS-style toggle switch with a text label."""

from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import Anim, Colors, Fonts


class ToggleSwitch(QWidget):
    """A compact animated toggle switch followed by an optional text label."""

    toggled = Signal(bool)

    _TRACK_W = 36
    _TRACK_H = 18
    _KNOB = 12

    def __init__(self, label: str = "", checked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._checked = checked
        self._label = label
        self._knob_pos = 1.0 if checked else 0.0  # 0..1 normalised travel
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(max(self._TRACK_H + 4, 22))
        self.setMinimumWidth(self._TRACK_W + 12 + (len(label) * 7 if label else 0))

        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(Anim.NORMAL)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    # ---------------------------------------------------------------- props
    def get_knob_pos(self) -> float:
        return self._knob_pos

    def set_knob_pos(self, value: float) -> None:
        self._knob_pos = value
        self.update()

    knobPos = Property(float, get_knob_pos, set_knob_pos)

    # ---------------------------------------------------------------- api
    def isChecked(self) -> bool:  # noqa: N802 (Qt convention)
        return self._checked

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        if checked == self._checked:
            return
        self._checked = checked
        self._anim.stop()
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()
        self.toggled.emit(checked)

    def setText(self, text: str) -> None:  # noqa: N802
        self._label = text
        self.update()

    def text(self) -> str:
        return self._label

    # ---------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.setChecked(not self._checked)
        super().mousePressEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        cy = self.height() / 2
        track = QRectF(0, cy - self._TRACK_H / 2, self._TRACK_W, self._TRACK_H)

        off = QColor(Colors.SURFACE_ACTIVE)
        on = QColor(Colors.ACCENT)
        track_color = self._mix(off, on, self._knob_pos)
        if not self.isEnabled():
            track_color.setAlphaF(0.4)
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track, self._TRACK_H / 2, self._TRACK_H / 2)

        margin = (self._TRACK_H - self._KNOB) / 2
        travel = self._TRACK_W - self._KNOB - 2 * margin
        knob_x = margin + self._knob_pos * travel
        knob = QRectF(knob_x, cy - self._KNOB / 2, self._KNOB, self._KNOB)
        painter.setBrush(QColor(Colors.WHITE))
        painter.drawEllipse(knob)

        if self._label:
            painter.setPen(QColor(Colors.TEXT_PRIMARY if self.isEnabled() else Colors.TEXT_MUTED))
            font = painter.font()
            font.setFamily(Fonts.FAMILY_PRIMARY)
            font.setPointSize(Fonts.SIZE_BODY)
            painter.setFont(font)
            text_rect = QRectF(self._TRACK_W + 8, 0, self.width() - self._TRACK_W - 8, self.height())
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._label)
        painter.end()

    @staticmethod
    def _mix(a: QColor, b: QColor, t: float) -> QColor:
        return QColor(
            int(a.red() + (b.red() - a.red()) * t),
            int(a.green() + (b.green() - a.green()) * t),
            int(a.blue() + (b.blue() - a.blue()) * t),
        )
