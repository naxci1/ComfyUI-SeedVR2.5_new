"""Drag & drop import area for video and image files."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFileDialog, QWidget

from ..theme import Colors, Dims, Fonts

_VIDEO_FILTER = "Media (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.tif *.tiff *.exr *.dpx);;All files (*)"
_VALID_EXT = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx", ".bmp",
}


class DropZone(QWidget):
    """Dashed drop area; emits ``file_dropped(path)`` on drop or file selection."""

    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False

    # ---------------------------------------------------------------- dnd
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            self._hover = True
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and self._is_valid(path):
                event.acceptProposedAction()
                self.file_dropped.emit(path)
                return
        event.ignore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(self, "Import media", "", _VIDEO_FILTER)
            if path:
                self.file_dropped.emit(path)
        super().mousePressEvent(event)

    @staticmethod
    def _is_valid(path: str) -> bool:
        import os

        return os.path.splitext(path)[1].lower() in _VALID_EXT

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        radius = Dims.CORNER_RADIUS_LG

        bg = QColor(Colors.BG_MEDIUM)
        painter.setBrush(bg)
        if self._hover:
            border = QColor(Colors.ACCENT)
            pen = QPen(border, 2, Qt.SolidLine)
            glow = QColor(Colors.ACCENT)
            glow.setAlpha(30)
            painter.setBrush(glow)
        else:
            border = QColor(Colors.BORDER)
            pen = QPen(border, 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, radius, radius)

        icon_color = QColor(Colors.ACCENT if self._hover else Colors.TEXT_SECONDARY)
        # Icon (simple up-arrow into a tray).
        painter.setPen(QPen(icon_color, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        cx = rect.center().x()
        cy = rect.center().y() - 18
        painter.drawLine(cx, cy + 22, cx, cy - 6)
        painter.drawLine(cx, cy - 6, cx - 8, cy + 2)
        painter.drawLine(cx, cy - 6, cx + 8, cy + 2)

        painter.setPen(QColor(Colors.TEXT_PRIMARY))
        font = painter.font()
        font.setFamily(Fonts.FAMILY_PRIMARY)
        font.setPointSize(Fonts.SIZE_H2)
        painter.setFont(font)
        text_rect = QRectF(rect.left(), cy + 28, rect.width(), 24)
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, "Drop video or image here")

        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        font.setPointSize(Fonts.SIZE_SMALL)
        painter.setFont(font)
        sub_rect = QRectF(rect.left(), cy + 52, rect.width(), 20)
        painter.drawText(sub_rect, Qt.AlignHCenter | Qt.AlignTop, "or click to browse")
        painter.end()
