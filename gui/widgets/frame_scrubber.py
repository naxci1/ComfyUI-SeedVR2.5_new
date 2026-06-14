"""Horizontal frame scrubber bar with click-to-seek and drag-to-scrub."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF, QPoint
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import Colors, Dims, Fonts


class FrameScrubber(QWidget):
    """A frame navigation scrubber emitting ``frame_changed`` / ``frame_selected``."""

    frame_changed = Signal(int)
    frame_selected = Signal(int)

    _HANDLE = 14

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(Dims.SCRUBBER_HEIGHT)
        self.setMouseTracking(True)
        self._frame_count = 0
        self._current = 0
        self._dragging = False
        self._hover_x = -1

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(0, count)
        self._current = min(self._current, max(0, count - 1))
        self.update()

    def set_frame(self, idx: int) -> None:
        if self._frame_count <= 0:
            return
        self._current = max(0, min(idx, self._frame_count - 1))
        self.update()

    # ---------------------------------------------------------------- geometry
    def _track_rect(self) -> QRectF:
        m = self._HANDLE / 2 + 2
        cy = self.height() / 2
        return QRectF(m, cy - 2, self.width() - 2 * m, 4)

    def _frame_from_x(self, x: float) -> int:
        track = self._track_rect()
        if track.width() <= 0 or self._frame_count <= 1:
            return 0
        frac = (x - track.left()) / track.width()
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * (self._frame_count - 1)))

    def _x_from_frame(self, idx: int) -> float:
        track = self._track_rect()
        if self._frame_count <= 1:
            return track.left()
        return track.left() + track.width() * (idx / (self._frame_count - 1))

    # ---------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._frame_count > 0:
            self._dragging = True
            idx = self._frame_from_x(event.position().x())
            self.set_frame(idx)
            self.frame_changed.emit(idx)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._hover_x = event.position().x()
        if self._dragging and self._frame_count > 0:
            idx = self._frame_from_x(event.position().x())
            if idx != self._current:
                self.set_frame(idx)
                self.frame_changed.emit(idx)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.frame_selected.emit(self._current)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_x = -1
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._frame_count <= 0:
            return
        step = 1 if event.angleDelta().y() > 0 else -1
        self.set_frame(self._current + step)
        self.frame_changed.emit(self._current)
        event.accept()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = self._track_rect()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Colors.SCRUB_TRACK))
        painter.drawRoundedRect(track, 2, 2)

        handle_x = self._x_from_frame(self._current)
        # Filled (played) portion.
        played = QRectF(track.left(), track.top(), handle_x - track.left(), track.height())
        painter.setBrush(QColor(Colors.SCRUB_FILL))
        painter.drawRoundedRect(played, 2, 2)

        # Hover guide line.
        if self._hover_x >= 0:
            pen = QPen(QColor(Colors.TEXT_SECONDARY), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(self._hover_x), 2, int(self._hover_x), self.height() - 2)

        # Handle.
        cy = self.height() / 2
        painter.setPen(QPen(QColor(Colors.ACCENT), 2))
        painter.setBrush(QColor(Colors.TEXT_PRIMARY))
        painter.drawEllipse(QPoint(int(handle_x), int(cy)), self._HANDLE // 2, self._HANDLE // 2)

        # Drag tooltip with frame number.
        if self._dragging and self._frame_count > 0:
            label = str(self._current)
            painter.setPen(QColor(Colors.TEXT_PRIMARY))
            font = painter.font()
            font.setFamily(Fonts.FAMILY_MONO)
            font.setPointSize(Fonts.SIZE_TINY)
            painter.setFont(font)
            painter.drawText(
                QRectF(handle_x - 24, 0, 48, 14), Qt.AlignCenter, label
            )
        painter.end()
