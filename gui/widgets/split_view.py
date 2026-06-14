"""Split-view comparison widget with a draggable vertical divider.

Left of the divider shows the original; right shows the processed result.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..theme import Colors


class SplitViewWidget(QWidget):
    """Overlay comparison: drag the vertical line to reveal original vs processed."""

    _HANDLE_W = 10  # px half-width of the drag zone

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._original: Optional[QPixmap] = None
        self._processed: Optional[QPixmap] = None
        self._split_frac = 0.5  # 0..1, position of divider
        self._dragging = False
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    # ---------------------------------------------------------------- api
    def set_images(self, original: QPixmap, processed: QPixmap) -> None:
        self._original = original
        self._processed = processed
        self.update()

    def clear(self) -> None:
        self._original = None
        self._processed = None
        self.update()

    def has_images(self) -> bool:
        return self._original is not None and self._processed is not None

    # ---------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            split_x = int(self._split_frac * self.width())
            if abs(event.position().x() - split_x) <= self._HANDLE_W * 2:
                self._dragging = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        split_x = int(self._split_frac * self.width())
        near = abs(event.position().x() - split_x) <= self._HANDLE_W * 2
        self.setCursor(Qt.SplitHCursor if near or self._dragging else Qt.ArrowCursor)
        if self._dragging and self.width() > 0:
            frac = event.position().x() / self.width()
            self._split_frac = max(0.01, min(0.99, frac))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        super().mouseReleaseEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(Colors.PREVIEW_BG))

        if self._original is None and self._processed is None:
            painter.setPen(QColor(Colors.TEXT_SECONDARY))
            painter.drawText(self.rect(), Qt.AlignCenter, "No media loaded")
            painter.end()
            return

        w = self.width()
        h = self.height()
        split_x = int(self._split_frac * w)

        def _draw_scaled(pix: QPixmap, clip_left: int, clip_right: int) -> None:
            if pix is None or pix.isNull():
                return
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.save()
            painter.setClipRect(clip_left, 0, clip_right - clip_left, h)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(x, y, scaled)
            painter.restore()

        if self._original is not None:
            _draw_scaled(self._original, 0, split_x)
        if self._processed is not None:
            _draw_scaled(self._processed, split_x, w)

        # Divider line.
        pen = QPen(QColor(Colors.TEXT_PRIMARY), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)

        # Handle circle.
        painter.setBrush(QColor(Colors.TEXT_PRIMARY))
        painter.setPen(Qt.NoPen)
        cy = h // 2
        r = 8
        painter.drawEllipse(split_x - r, cy - r, 2 * r, 2 * r)

        # Labels.
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        label_pad = 8
        painter.setPen(QColor(Colors.TEXT_PRIMARY))
        if split_x > 60:
            painter.drawText(QRectF(label_pad, label_pad, split_x - label_pad * 2, 20),
                             Qt.AlignLeft | Qt.AlignTop, "Original")
        if w - split_x > 60:
            painter.drawText(QRectF(split_x + label_pad, label_pad, w - split_x - label_pad * 2, 20),
                             Qt.AlignLeft | Qt.AlignTop, "Processed")

        painter.end()
