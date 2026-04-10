"""
SplitViewWidget – Topaz-style overlay comparison player.

Output video is the base layer (full frame).
Input video is drawn on top but pixel-clipped to the left of the draggable
divider, giving a true before/after overlay that stays perfectly frame-synced.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtMultimedia import QVideoFrame, QVideoSink
from PyQt6.QtWidgets import QWidget


def _to_renderable(img: QImage) -> QImage:
    """Convert *img* to ARGB32 if it is not already in a directly paintable format."""
    _ok = {
        QImage.Format.Format_ARGB32,
        QImage.Format.Format_ARGB32_Premultiplied,
        QImage.Format.Format_RGB32,
        QImage.Format.Format_RGB888,
    }
    if img.format() not in _ok:
        return img.convertToFormat(QImage.Format.Format_ARGB32)
    return img


class SplitViewWidget(QWidget):
    """
    Single-frame overlay comparison widget (Topaz Video AI style).

    * The **Output** video is rendered as the full base layer.
    * The **Input** video is clipped to the left of the divider, revealing
      the original on the left and the upscaled result on the right.
    * The divider can be dragged horizontally.

    Wire up your players in split mode::

        input_player.setVideoOutput(split_widget.input_sink)
        output_player.setVideoOutput(split_widget.output_sink)
    """

    _HANDLE_RADIUS = 10
    _HANDLE_HIT_MARGIN = 22  # pixels either side of divider that trigger drag

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._split_ratio: float = 0.5  # 0.0 = all output, 1.0 = all input
        self._dragging: bool = False
        self._input_image: Optional[QImage] = None
        self._output_image: Optional[QImage] = None

        self.input_sink = QVideoSink(self)
        self.output_sink = QVideoSink(self)
        self.input_sink.videoFrameChanged.connect(self._on_input_frame)
        self.output_sink.videoFrameChanged.connect(self._on_output_frame)

        self.setMinimumHeight(300)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Frame reception
    # ------------------------------------------------------------------

    def _on_input_frame(self, frame: QVideoFrame) -> None:
        if frame.isValid():
            img = frame.toImage()
            if not img.isNull():
                self._input_image = _to_renderable(img)
                self.update()

    def _on_output_frame(self, frame: QVideoFrame) -> None:
        if frame.isValid():
            img = frame.toImage()
            if not img.isNull():
                self._output_image = _to_renderable(img)
                self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        h = self.height()
        split_x = int(w * self._split_ratio)

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0d0d0d"))

        def _draw(img: QImage) -> None:
            """Scale *img* to fit the widget while keeping aspect ratio (letterbox)."""
            scaled = img.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = (w - scaled.width()) // 2
            y_off = (h - scaled.height()) // 2
            painter.drawImage(x_off, y_off, scaled)

        # Output – base layer (full frame)
        if self._output_image and not self._output_image.isNull():
            _draw(self._output_image)

        # Input – top layer, clipped to left of divider
        if self._input_image and not self._input_image.isNull() and split_x > 0:
            painter.save()
            painter.setClipRect(0, 0, split_x, h)
            _draw(self._input_image)
            painter.restore()

        # Divider line
        painter.setPen(QPen(QColor("#00b4d8"), 2))
        painter.drawLine(split_x, 0, split_x, h)

        # Handle circle
        hr = self._HANDLE_RADIUS
        cy = h // 2
        painter.setBrush(QBrush(QColor("#00b4d8")))
        painter.setPen(QPen(QColor("#005580"), 1))
        painter.drawEllipse(split_x - hr, cy - hr, hr * 2, hr * 2)

        # Labels
        lbl_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(lbl_font)
        painter.setPen(QColor("white"))
        if split_x > 70:
            painter.drawText(10, 24, "ORIGINAL")
        if split_x < w - 90:
            painter.drawText(split_x + 10, 24, "UPSCALED")

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction (drag the divider)
    # ------------------------------------------------------------------

    def _near_handle(self, x: float) -> bool:
        return abs(x - self.width() * self._split_ratio) < self._HANDLE_HIT_MARGIN

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._near_handle(event.position().x()):
            self._dragging = True
            self.setCursor(Qt.CursorShape.SizeHorCursor)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging:
            ratio = event.position().x() / max(1, self.width())
            self._split_ratio = max(0.0, min(1.0, ratio))
            self.update()
        elif self._near_handle(event.position().x()):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Static image comparison (used when input is an image, not a video)
    # ------------------------------------------------------------------

    def set_input_image(self, img: QImage) -> None:
        """Display a static QImage as the input (left/before) side."""
        if not img.isNull():
            self._input_image = _to_renderable(img)
            self.update()

    def set_output_image(self, img: QImage) -> None:
        """Display a static QImage as the output (right/after) side."""
        if not img.isNull():
            self._output_image = _to_renderable(img)
            self.update()
